import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
LOG2E = 1.4426950408889634
SCALE = HEAD_DIM_K ** -0.5


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
    },
)
def _gdn_prefill_kernel(H, Hg, dtype, accum_dtype):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")

    qk_shape = (batch_size, num_tokens, H, HEAD_DIM_K)
    k_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    output_shape = (batch_size, num_tokens, H, HEAD_DIM_V)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=dtype),
        k: T.Tensor(k_shape, dtype=dtype),
        v: T.Tensor(v_shape, dtype=dtype),
        g_cumsum: T.Tensor(gate_shape, dtype=accum_dtype),
        beta: T.Tensor(gate_shape, dtype=accum_dtype),
        A: T.Tensor(a_shape, dtype=dtype),
        initial_state: T.Tensor(state_shape, dtype=accum_dtype),
        output: T.Tensor(output_shape, dtype=dtype),
        final_state: T.Tensor(state_shape, dtype=accum_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(batch_size * H, threads=256) as (block_id,):
            bb = block_id // H
            hh = block_id % H
            hhg = hh // (H // Hg) if H != Hg else hh

           # State persists across chunks (FP32 for accuracy)
            state_frag = T.alloc_fragment((HEAD_DIM_K, HEAD_DIM_V), dtype=accum_dtype)
            state_bf16 = T.alloc_shared((HEAD_DIM_K, HEAD_DIM_V), dtype=dtype)

            # Input data in BF16
            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=dtype)
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)

            # Precomputed exp(g) to avoid redundant exp2 calls
            exp_g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            inv_exp_g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            beta_exp_g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)

            # Shared BF16 scratch for K_decay then W (reused, saves 16KB)
            kv_scratch_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            # Dedicated V_beta buffer (decouples from kv_scratch_shared)
            v_beta_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_V), dtype=dtype)

            v_new_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_V), dtype=dtype)
            scores_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=dtype)

            # Fragments (registers, FP32 accumulators)
            u_frag = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            out_chunk_frag = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            scores_frag = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)

            # Initialize state and BF16 copy
            for d1, d2 in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                state_frag[d1, d2] = initial_state[bb, hh, d1, d2]
            for d1, d2 in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                state_bf16[d1, d2] = T.cast(state_frag[d1, d2], dtype)

            for chunk_idx in T.serial(num_chunks):
                left = chunk_idx * CHUNK_SIZE
                right = left + CHUNK_SIZE

                # Load chunk data with tail handling
                if right <= num_tokens:
                    if chunk_idx == 0:
                        T.async_copy(q[bb, left:right, hh, 0:HEAD_DIM_K], q_shared)
                        T.async_copy(k[bb, left:right, hhg, 0:HEAD_DIM_K], k_shared)
                        T.async_copy(A[bb, left:right, hh, 0:CHUNK_SIZE], a_shared)
                        T.async_copy(v[bb, left:right, hh, 0:HEAD_DIM_V], v_beta_shared)
                    for i in T.Parallel(CHUNK_SIZE):
                        g_shared[i] = g_cumsum[bb, left + i, hh]
                        beta_shared[i] = beta[bb, left + i, hh]
                else:
                    for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                        if left + i < num_tokens:
                            q_shared[i, d] = q[bb, left + i, hh, d]
                            k_shared[i, d] = k[bb, left + i, hhg, d]
                        else:
                            q_shared[i, d] = 0
                            k_shared[i, d] = 0
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        if left + i < num_tokens:
                            a_shared[i, j] = A[bb, left + i, hh, j]
                        else:
                            a_shared[i, j] = 0
                    for i in T.Parallel(CHUNK_SIZE):
                        if left + i < num_tokens:
                            g_shared[i] = g_cumsum[bb, left + i, hh]
                            beta_shared[i] = beta[bb, left + i, hh]
                        else:
                            # Pad g with last valid cumsum so g_last is correct
                            g_shared[i] = g_cumsum[bb, num_tokens - 1, hh]
                            beta_shared[i] = 0

                # Precompute exp(g) once per chunk
                for i in T.Parallel(CHUNK_SIZE):
                    exp_g_shared[i] = T.exp2(g_shared[i] * LOG2E)
                    inv_exp_g_shared[i] = T.exp2(-g_shared[i] * LOG2E)
                    beta_exp_g_shared[i] = beta_shared[i] * exp_g_shared[i]

                # Wait for async global memory loads (non-tail case only)
                if right <= num_tokens:
                    T.ptx_wait_group(0)

                # K_decay = K * beta * exp(g)  -> BF16 (for GEMM with A which is BF16)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    kv_scratch_shared[i, d] = T.cast(
                        T.cast(k_shared[i, d], accum_dtype)
                        * beta_exp_g_shared[i],
                        dtype,
                    )

                # K_state = K_decay @ state (replaces W=A@K_decay, saves 1 GEMM)
                T.gemm(kv_scratch_shared, state_bf16, u_frag, clear_accum=True)
                T.copy(u_frag, kv_scratch_shared)

                # Fused V*beta - K_state (non-tail: V in shared; saves 1 element-wise loop)
                if right <= num_tokens:
                    for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                        kv_scratch_shared[i, d] = T.cast(
                            T.cast(v_beta_shared[i, d], accum_dtype) * beta_shared[i]
                            - T.cast(kv_scratch_shared[i, d], accum_dtype),
                            dtype,
                        )
                else:
                    for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                        if left + i < num_tokens:
                            kv_scratch_shared[i, d] = T.cast(
                                T.cast(v[bb, left + i, hh, d], accum_dtype) * beta_shared[i]
                                - T.cast(kv_scratch_shared[i, d], accum_dtype),
                                dtype,
                            )
                        else:
                            kv_scratch_shared[i, d] = 0

                # V_new = A @ (V_beta - K_state)
                T.gemm(a_shared, kv_scratch_shared, u_frag, clear_accum=True)
                T.copy(u_frag, v_new_shared)

                # scores = Q @ K^T  (moved earlier for ILP overlap)
                T.gemm(q_shared, k_shared, scores_frag, transpose_B=True, clear_accum=True)

                # Q @ state (u_frag is now free, reuse it)
                T.gemm(q_shared, state_bf16, u_frag, clear_accum=True)
                for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if i >= j:
                        scores_shared[i, j] = T.cast(
                            scores_frag[i, j] * exp_g_shared[i] * inv_exp_g_shared[j],
                            dtype,
                        )
                    else:
                        scores_shared[i, j] = 0

                # output_in_chunk = scores @ V_new  (BF16 x BF16 -> FP32 frag)
                T.gemm(scores_shared, v_new_shared, out_chunk_frag, clear_accum=True)

                # State update prep
                g_last = g_shared[CHUNK_SIZE - 1]
                exp_g_last = exp_g_shared[CHUNK_SIZE - 1]

                # Fused output write + K_decay_last computation (same dims, independent)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    if left + i < num_tokens:
                        output[bb, left + i, hh, d] = T.cast(
                            SCALE * (exp_g_shared[i] * u_frag[i, d] + out_chunk_frag[i, d]),
                            dtype,
                        )
                    kv_scratch_shared[i, d] = T.cast(
                        T.cast(k_shared[i, d], accum_dtype)
                        * exp_g_last * inv_exp_g_shared[i],
                        dtype,
                    )

                # Prefetch next chunk's data (overlaps with state_update GEMM + state update)
                next_left = (chunk_idx + 1) * CHUNK_SIZE
                next_right = next_left + CHUNK_SIZE
                if next_right <= num_tokens:
                    T.async_copy(q[bb, next_left:next_right, hh, 0:HEAD_DIM_K], q_shared)
                    T.async_copy(k[bb, next_left:next_right, hhg, 0:HEAD_DIM_K], k_shared)
                    T.async_copy(A[bb, next_left:next_right, hh, 0:CHUNK_SIZE], a_shared)
                    T.async_copy(v[bb, next_left:next_right, hh, 0:HEAD_DIM_V], v_beta_shared)

                # Scale state by exp_g_last (in registers, no shared mem traffic)
                for d1, d2 in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                    state_frag[d1, d2] = state_frag[d1, d2] * exp_g_last
                # state += K_decay_last^T @ V_new (accumulate onto pre-scaled state)
                T.gemm(
                    kv_scratch_shared, v_new_shared, state_frag,
                    transpose_A=True, clear_accum=False,
                )
                # Refresh state_bf16 from state_frag
                for d1, d2 in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                    state_bf16[d1, d2] = T.cast(state_frag[d1, d2], dtype)

            # Write final state
            for d1, d2 in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                final_state[bb, hh, d1, d2] = state_frag[d1, d2]

    return kernel


def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_tokens, num_heads_qk, _ = q.shape
    num_heads_v = v.shape[2]
    num_chunks = tilelang.cdiv(num_tokens, CHUNK_SIZE)

    # Expand Q heads for GVA (one Q/K head maps to multiple V heads)
    if num_heads_qk != num_heads_v:
        q = q.repeat_interleave(num_heads_v // num_heads_qk, dim=2)

    if initial_state is None:
        initial_state = torch.zeros(
            (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        initial_state = initial_state.to(torch.float32)

    output = torch.empty(
        (batch_size, num_tokens, num_heads_v, HEAD_DIM_V),
        dtype=torch.bfloat16,
        device=q.device,
    )
    final_state = torch.empty(
        (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
        dtype=torch.float32,
        device=q.device,
    )

    kernel = _gdn_prefill_kernel(
        num_heads_v,
        num_heads_qk,
        dtype="bfloat16",
        accum_dtype="float32",
    )
    kernel(
        q, k, v, g_cumsum, beta, A,
        initial_state, output, final_state,
        num_chunks,
    )
    return output, final_state
