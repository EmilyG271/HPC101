// Main task: optimize the MoE forward pass.
//
// Correctness-repaired version of the student's original AMX/AVX-512 design:
//   router -> Top-K -> per-token W8A8 quantization -> expert grouping
//   -> AMX INT8 GEMM -> SwiGLU -> requantization -> AMX down projection.

#include "moe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif
// R128: cache omp_get_num_procs (0.24us/call) — called every moe_forward.
static inline int get_num_procs_cached() {
    static int cached = omp_get_num_procs();
    return cached;
}

#if defined(_OPENMP) && defined(__linux__)
// R62: Bind threads to physical cores and pack them close together
// to improve L2/L3 locality for weight-streaming kernels.
__attribute__((constructor))
static void tune_omp_affinity() {
    setenv("OMP_PROC_BIND", "close", 0);
    setenv("OMP_PLACES", "cores", 0);
}
#endif
// R104: Pin S1 persistent worker threads to specific cores for cache locality
#if defined(__linux__)
#include <sched.h>
static void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
}
#else
static void pin_thread_to_core(int) {}
#endif



#if defined(__AMX_INT8__) && defined(__AMX_TILE__) && defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif
#endif

// -----------------------------------------------------------------------------
// Routing scratch buffers
// -----------------------------------------------------------------------------

// Router 只保留最终 Top-K 结果；logit/affinity 在 token 局部产生并立即消费，
// 不再写入完整 [N,E] 中间矩阵。
static int g_topk_indices[MAX_NUM_TOKENS][MAX_TOP_K];
static float g_topk_weights[MAX_NUM_TOKENS][MAX_TOP_K];
static int g_expert_token_count[MAX_NUM_EXPERTS];
static int g_expert_token_list[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];
// 分发阶段直接保存路由权重，专家回写时无需再次扫描 K。
static float g_expert_token_weight[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];

// -----------------------------------------------------------------------------
// Compact runtime-layout activation/output buffers
//
// These are flat on purpose. A declaration such as [MAX_NUM_TOKENS][MAX_D_FF]
// has a physical row stride of MAX_D_FF, which is wrong when the runtime H is
// smaller. AMX must see the actual compact [M, K] / [M, N] row strides.
// -----------------------------------------------------------------------------

alignas(64) static int8_t g_quantized_x[MAX_NUM_TOKENS * MAX_D_MODEL];
static float g_x_scale[MAX_NUM_TOKENS];

alignas(64) static int32_t g_shared_gate_out[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int32_t g_shared_up_out[MAX_NUM_TOKENS * MAX_D_FF];
static float g_shared_gated_fp32[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int8_t
    g_shared_quantized_gated[MAX_NUM_TOKENS * MAX_D_FF];
static float g_shared_gated_scale[MAX_NUM_TOKENS];
alignas(64) static int32_t
    g_shared_down_out[MAX_NUM_TOKENS * MAX_D_MODEL];

alignas(64) static int8_t g_amx_A[MAX_NUM_TOKENS * MAX_D_MODEL];
alignas(64) static int32_t g_gate_out[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int32_t g_up_out[MAX_NUM_TOKENS * MAX_D_FF];
static float g_gated_fp32[MAX_NUM_TOKENS * MAX_D_FF];
alignas(64) static int8_t
    g_quantized_gated[MAX_NUM_TOKENS * MAX_D_FF];
static float g_gated_scale[MAX_NUM_TOKENS];
alignas(64) static int32_t g_down_out[MAX_NUM_TOKENS * MAX_D_MODEL];

// -----------------------------------------------------------------------------
// Prepacked expert weights
//
// Logical multiplication is A[M,K] * B[K,N]. The framework stores each weight
// matrix output-major as W[N,K], so preprocess transposes and VNNI-packs it as
// B_pack[K/4, N*4]:
//   B_pack[k/4][4*n + k%4] = W[n][k].
// -----------------------------------------------------------------------------

alignas(64) static int8_t
    g_packed_gate[MAX_NUM_EXPERTS][MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_up[MAX_NUM_EXPERTS][MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_down[MAX_NUM_EXPERTS][MAX_D_FF * MAX_D_MODEL];

static float g_scale_gate[MAX_NUM_EXPERTS];
static float g_scale_up[MAX_NUM_EXPERTS];
static float g_scale_down[MAX_NUM_EXPERTS];
alignas(64) static int32_t g_sum_gate[MAX_NUM_EXPERTS][MAX_D_FF];
alignas(64) static int32_t g_sum_up[MAX_NUM_EXPERTS][MAX_D_FF];
alignas(64) static int32_t g_sum_down[MAX_NUM_EXPERTS][MAX_D_MODEL];

alignas(64) static int8_t
    g_packed_shared_gate[MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_shared_up[MAX_D_MODEL * MAX_D_FF];
alignas(64) static int8_t
    g_packed_shared_down[MAX_D_FF * MAX_D_MODEL];
static float g_scale_shared_gate;
static float g_scale_shared_up;
static float g_scale_shared_down;
alignas(64) static int32_t g_sum_shared_gate[MAX_D_FF];
alignas(64) static int32_t g_sum_shared_up[MAX_D_FF];
alignas(64) static int32_t g_sum_shared_down[MAX_D_MODEL];

// R126: S1-only blocked VNNI layout. Each 16-output block stores all K/4
// groups contiguously, so VPDPBUSD produces 16 final output channels directly
// and avoids a horizontal reduction for every scalar output.
static constexpr int S1_PACK_E = 16;
static constexpr int S1_PACK_D = 256;
static constexpr int S1_PACK_H = 128;
alignas(64) static int8_t
    g_s1_gate[S1_PACK_E][S1_PACK_D * S1_PACK_H];
alignas(64) static int8_t
    g_s1_up[S1_PACK_E][S1_PACK_D * S1_PACK_H];
alignas(64) static int8_t
    g_s1_down[S1_PACK_E][S1_PACK_D * S1_PACK_H];
alignas(64) static int8_t g_s1_shared_gate[S1_PACK_D * S1_PACK_H];
alignas(64) static int8_t g_s1_shared_up[S1_PACK_D * S1_PACK_H];
alignas(64) static int8_t g_s1_shared_down[S1_PACK_D * S1_PACK_H];

#if defined(_OPENMP) && defined(__AVX512VNNI__)
static constexpr int S2_SPMD_D = 1024;
static constexpr int S2_SPMD_H = 512;
static constexpr int S2_SPMD_E = 16;
static constexpr int S2_SPMD_THREADS = 8;
static constexpr int S2_SPMD_SLOTS = S2_SPMD_E + 1;
static constexpr int S2_SPMD_H_SHARD = S2_SPMD_H / S2_SPMD_THREADS;
static constexpr int S2_SPMD_D_SHARD = S2_SPMD_D / S2_SPMD_THREADS;
static constexpr size_t S2_SPMD_GU_SHARD_ELEMS =
    (size_t)S2_SPMD_H_SHARD * S2_SPMD_D;
static constexpr size_t S2_SPMD_DOWN_SHARD_ELEMS =
    (size_t)S2_SPMD_D_SHARD * S2_SPMD_H;

static int8_t* g_s2_spmd_gate_w = nullptr;
static int8_t* g_s2_spmd_up_w = nullptr;
static int8_t* g_s2_spmd_down_w = nullptr;
static bool g_s2_spmd_ready = false;

static void ensure_s2_spmd_pool();
#endif

// R10: Router 权重 FP16 存储（带宽降 2x，cvt 后 FP32 计算精度无损）
alignas(64) static uint16_t g_router_f16[MAX_NUM_EXPERTS * MAX_D_MODEL];

// R12c: INT8 router weights for AMX GEMM (large-E path)
alignas(64) static int8_t g_router_int8[MAX_NUM_EXPERTS * MAX_D_MODEL];
alignas(64) static int8_t g_packed_router[MAX_D_MODEL * MAX_NUM_EXPERTS];
static float g_scale_router[MAX_NUM_EXPERTS];
// R134: 128 × per-expert router row-sum (for INT8 VNNI routing correction)
static int32_t g_sum_router_128[MAX_NUM_EXPERTS];
// R137: interleaved VNNI-broadcast router for S1 single-ZMM routing.
// Layout: [4-input group g][16 outputs x 4 bytes] (64 B/group). One VPDPBUSD
// per group computes all 16 expert logits in one ZMM lane each, eliminating
// the 16 zmm_hsum_i32 horizontal reductions the generic kernel needed.
alignas(64) static int8_t g_router_vnni[16 * MAX_D_MODEL];
alignas(64) static int32_t g_router_logits[MAX_NUM_TOKENS * MAX_NUM_EXPERTS];

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
alignas(64) static thread_local unsigned char g_tile_cfg[64];
// Tile 配置属于线程的扩展寄存器状态；缓存当前行数可跳过重复 LDTILECFG。
static thread_local int g_last_tile_rows = -1;
#endif
static bool g_amx_runtime_enabled = false;

// 每线程独立 GEMM scratch（专家并行用）
#if defined(_OPENMP)
struct ExpertScratch {
    int8_t*  A = nullptr;
    int32_t* gate_out = nullptr;
    int32_t* up_out = nullptr;
    float*   gated_fp32 = nullptr;
    int8_t*  quantized_gated = nullptr;
    float*   gated_scale = nullptr;
    int32_t* down_out = nullptr;
    size_t   cap = 0;
    bool     amx_perm = false;

    static void* xalloc(size_t n) {
        void* p = nullptr;
        size_t sz = (n + 63) & ~((size_t)63);
        if (posix_memalign(&p, 64, sz) != 0) return nullptr;
        return p;
    }
    void ensure(size_t max_rows, size_t D, size_t H) {
        size_t maxdim = (D > H) ? D : H;
        size_t need = max_rows * maxdim;
        if (need <= cap) return;
        free_all();
        A               = (int8_t*)  xalloc(need);
        gate_out        = (int32_t*) xalloc(need * 4);
        up_out          = (int32_t*) xalloc(need * 4);
        gated_fp32      = (float*)   xalloc(need * 4);
        quantized_gated = (int8_t*)  xalloc(need);
        down_out        = (int32_t*) xalloc(need * 4);
        gated_scale     = (float*)   xalloc(max_rows * 4);
        cap = need;
    }
    void free_all() {
        free(A); free(gate_out); free(up_out); free(gated_fp32);
        free(quantized_gated); free(down_out); free(gated_scale);
        A = nullptr; gate_out = nullptr; up_out = nullptr;
        gated_fp32 = nullptr; quantized_gated = nullptr;
        down_out = nullptr; gated_scale = nullptr;
        cap = 0;
    }
    ~ExpertScratch() { free_all(); }
};
static thread_local ExpertScratch tl_scratch;
#endif

// 小批量保持串行，避免 OpenMP 团队创建/调度开销影响 S1、S2。
static constexpr int OMP_TOKEN_THRESHOLD = 64;

// R95: Custom horizontal sum of 16 int32 lanes via VPADDD (lat 1) + PSHUFD,
// replacing _mm512_reduce_add_epi32 which uses PHADDD (lat 7, tput 0.5).
// Lower latency lets compiler overlap reduction with next iteration's VPDPBUSDs.
static inline int32_t zmm_hsum_i32(__m512i v) {
    __m256i lo = _mm512_castsi512_si256(v);
    __m256i hi = _mm512_extracti32x8_epi32(v, 1);
    __m256i s1 = _mm256_add_epi32(lo, hi);
    __m128i s2 = _mm256_castsi256_si128(s1);
    __m128i s3 = _mm256_extracti128_si256(s1, 1);
    __m128i s4 = _mm_add_epi32(s2, s3);
    __m128i s5 = _mm_shuffle_epi32(s4, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i s6 = _mm_add_epi32(s4, s5);
    __m128i s7 = _mm_shuffle_epi32(s6, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i s8 = _mm_add_epi32(s6, s7);
    return _mm_cvtsi128_si32(s8);
}

/**
 * @brief 将框架的输出通道优先 INT8 权重转置并打包为 AMX TDPBSSD 所需布局。
 *
 * 功能：框架中的权重按 W[N][K] 连续存储，而矩阵乘逻辑需要 B[K][N]。
 * 本函数在转置的同时，将 K 维每连续 4 个有符号 INT8 元素组成一个 4-byte
 * 点积小组，最终得到 B_pack[K/4][N*4]。打包后的地址映射为：
 *     B_pack[k/4][4*n + k%4] = W[n][k]。
 *
 * 实现思路：外层遍历 K/4 个四元素条带，中层遍历输出列 n，内层复制该列
 * 在 K 维上的 4 个连续元素。该操作只在 preprocess() 中执行一次，前向阶段
 * 可以直接顺序加载打包权重，避免每次矩阵乘时临时转置或跨行收集。
 *
 * 向量化情况：本函数本身未显式使用 AVX-512/AMX；它属于 AMX 计算前的数据
 * 预处理。真正的矩阵计算由 amx_matmul_packed() 使用 AMX TDPBSSD 完成。
 *
 * @param src    原始权重，逻辑形状为 [N, K]，元素类型为 int8_t。
 * @param packed 输出缓冲区，逻辑形状为 [K/4, N*4]。
 * @param K      归约维度，实验保证为 64 的倍数。
 * @param N      输出维度，实验保证为 64 的倍数。
 */
static void pack_output_major_weight(const int8_t* src, int8_t* packed,
                                     int K, int N) {
    // src is [N][K], while logical B is [K][N]. K is guaranteed to be a
    // multiple of 64 by the driver, hence also a multiple of 4.
    for (int k4 = 0; k4 < K / 4; ++k4) {
        for (int n = 0; n < N; ++n) {
            for (int r = 0; r < 4; ++r) {
                packed[(size_t)k4 * (N * 4) + 4 * n + r] =
                    src[(size_t)n * K + 4 * k4 + r];
            }
        }
    }
}

static void pack_s1_output_blocks(const int8_t* src, int8_t* packed,
                                  int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 16) {
        int8_t* block = packed + (size_t)n0 * K;
        for (int k4 = 0; k4 < K / 4; ++k4) {
            int8_t* dst = block + (size_t)k4 * 64;
            for (int n = 0; n < 16; ++n) {
                std::memcpy(dst + 4 * n,
                            src + (size_t)(n0 + n) * K + 4 * k4, 4);
            }
        }
    }
}

#if defined(_OPENMP) && defined(__AVX512VNNI__)
static void pack_s2_spmd_weight(const int8_t* src, int8_t* packed,
                                int K, int N) {
    const int rows_per_thread = N / S2_SPMD_THREADS;
    const size_t shard_elems = (size_t)rows_per_thread * K;
    for (int participant = 0; participant < S2_SPMD_THREADS; ++participant) {
        pack_s1_output_blocks(
            src + (size_t)participant * rows_per_thread * K,
            packed + (size_t)participant * shard_elems,
            K, rows_per_thread);
    }
}
#endif

static void compute_output_major_row_sums(const int8_t* src, int32_t* sums,
                                          int K, int N) {
    for (int n = 0; n < N; ++n) {
        int32_t sum = 0;
        const int8_t* row = src + (size_t)n * K;
        for (int k = 0; k < K; ++k) sum += (int32_t)row[k];
        sums[n] = sum * 128;
    }
}

/**
 * @brief 为当前进程/线程申请使用 Intel AMX Tile 数据状态的权限。
 *
 * 功能：Linux 内核采用扩展状态按需授权机制。程序在第一次执行 AMX Tile 指令
 * 之前，需要通过 arch_prctl 请求 XFEATURE_XTILEDATA 权限，否则可能在执行
 * LDTILECFG、TILELOADD 或 TDPBSSD 时触发异常。
 *
 * 实现思路：仅当编译器同时定义 __AMX_INT8__ 和 __AMX_TILE__ 时启用 AMX
 * 代码。Linux 下调用 syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
 * XFEATURE_XTILEDATA)；其他 AMX 编译平台直接返回 true；未启用 AMX 的构建
 * 返回 false，使后续矩阵乘自动使用标量回退实现。
 *
 * 向量化情况：不执行数值向量化；该函数负责 AMX 运行环境初始化和能力选择。
 *
 * @return true 表示可以进入 AMX kernel；false 表示使用标量回退路径。
 */
static bool request_amx_permission() {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
#if defined(__linux__)
    // Linux requires each process/thread to request permission for XTILEDATA
    // before executing the first AMX instruction.
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM,
                   XFEATURE_XTILEDATA) == 0;
#else
    return true;
#endif
#else
    return false;
#endif
}

/**
 * @brief 在计时开始前预打包全部路由专家和共享专家的 INT8 权重。
 *
 * 功能：读取 MoEWeights 中已经完成 W8 量化的 Gate、Up、Down 权重，将它们
 * 转换为 AMX TDPBSSD 可直接消费的 VNNI 风格布局，并保存框架提供的每矩阵
 * FP32 反量化 scale。原始权重不会被覆盖，满足实验对 preprocess() 的要求。
 *
 * 实现思路：
 * 1. Gate/Up 的原始形状是 [H,D]，按 K=D、N=H 打包；
 * 2. Down 的原始形状是 [D,H]，按 K=H、N=D 打包；
 * 3. 每个专家分别保存 s_gate、s_up、s_down；
 * 4. 共享专家执行同样处理；
 * 5. 最后申请 AMX 权限，决定前向阶段使用 AMX 还是标量回退 kernel。
 *
 * 向量化情况：本函数不直接执行向量矩阵运算；其数据重排专门服务于后续 AMX
 * 分块矩阵乘。预打包只执行一次，避免在每轮 moe_forward_optimized() 中重复。
 *
 * @param w 模型权重及运行时维度。函数只读权重内容，不原地修改权重数组。
 */
void preprocess(MoEWeights& w) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int E = w.num_experts;
    const size_t expert_elems = (size_t)D * H;

    // The framework has already quantized all expert weights to int8 and
    // provides their dequantization scales. preprocess only repacks them.
    for (int e = 0; e < E; ++e) {
        pack_output_major_weight(w.w_gate + (size_t)e * expert_elems,
                                 g_packed_gate[e], D, H);
        pack_output_major_weight(w.w_up + (size_t)e * expert_elems,
                                 g_packed_up[e], D, H);
        pack_output_major_weight(w.w_down + (size_t)e * expert_elems,
                                 g_packed_down[e], H, D);
        compute_output_major_row_sums(
            w.w_gate + (size_t)e * expert_elems, g_sum_gate[e], D, H);
        compute_output_major_row_sums(
            w.w_up + (size_t)e * expert_elems, g_sum_up[e], D, H);
        compute_output_major_row_sums(
            w.w_down + (size_t)e * expert_elems, g_sum_down[e], H, D);
        if (D == S1_PACK_D && H == S1_PACK_H && E <= S1_PACK_E) {
            pack_s1_output_blocks(
                w.w_gate + (size_t)e * expert_elems, g_s1_gate[e], D, H);
            pack_s1_output_blocks(
                w.w_up + (size_t)e * expert_elems, g_s1_up[e], D, H);
            pack_s1_output_blocks(
                w.w_down + (size_t)e * expert_elems, g_s1_down[e], H, D);
        }
        g_scale_gate[e] = w.s_gate[e];
        g_scale_up[e] = w.s_up[e];
        g_scale_down[e] = w.s_down[e];
    }

    pack_output_major_weight(w.sh_gate, g_packed_shared_gate, D, H);
    pack_output_major_weight(w.sh_up, g_packed_shared_up, D, H);
    pack_output_major_weight(w.sh_down, g_packed_shared_down, H, D);
    compute_output_major_row_sums(
        w.sh_gate, g_sum_shared_gate, D, H);
    compute_output_major_row_sums(
        w.sh_up, g_sum_shared_up, D, H);
    compute_output_major_row_sums(
        w.sh_down, g_sum_shared_down, H, D);
    if (D == S1_PACK_D && H == S1_PACK_H && E <= S1_PACK_E) {
        pack_s1_output_blocks(w.sh_gate, g_s1_shared_gate, D, H);
        pack_s1_output_blocks(w.sh_up, g_s1_shared_up, D, H);
        pack_s1_output_blocks(w.sh_down, g_s1_shared_down, H, D);
    }
#if defined(_OPENMP) && defined(__AVX512VNNI__)
    g_s2_spmd_ready = false;
    if (D == S2_SPMD_D && H == S2_SPMD_H && E == S2_SPMD_E &&
        w.top_k == MAX_TOP_K) {
        const size_t gu_total =
            (size_t)S2_SPMD_THREADS * S2_SPMD_SLOTS *
            S2_SPMD_GU_SHARD_ELEMS;
        const size_t down_total =
            (size_t)S2_SPMD_THREADS * S2_SPMD_SLOTS *
            S2_SPMD_DOWN_SHARD_ELEMS;
        if (!g_s2_spmd_gate_w)
            g_s2_spmd_gate_w = static_cast<int8_t*>(
                std::aligned_alloc(64, gu_total));
        if (!g_s2_spmd_up_w)
            g_s2_spmd_up_w = static_cast<int8_t*>(
                std::aligned_alloc(64, gu_total));
        if (!g_s2_spmd_down_w)
            g_s2_spmd_down_w = static_cast<int8_t*>(
                std::aligned_alloc(64, down_total));
        if (g_s2_spmd_gate_w && g_s2_spmd_up_w && g_s2_spmd_down_w) {
            for (int expert = 0; expert < E; ++expert) {
                const size_t gu_slot =
                    (size_t)expert * S2_SPMD_THREADS * S2_SPMD_GU_SHARD_ELEMS;
                const size_t down_slot =
                    (size_t)expert * S2_SPMD_THREADS * S2_SPMD_DOWN_SHARD_ELEMS;
                pack_s2_spmd_weight(
                    w.w_gate + (size_t)expert * expert_elems,
                    g_s2_spmd_gate_w + gu_slot, D, H);
                pack_s2_spmd_weight(
                    w.w_up + (size_t)expert * expert_elems,
                    g_s2_spmd_up_w + gu_slot, D, H);
                pack_s2_spmd_weight(
                    w.w_down + (size_t)expert * expert_elems,
                    g_s2_spmd_down_w + down_slot, H, D);
            }
            const size_t shared_gu_slot =
                (size_t)S2_SPMD_E * S2_SPMD_THREADS * S2_SPMD_GU_SHARD_ELEMS;
            const size_t shared_down_slot =
                (size_t)S2_SPMD_E * S2_SPMD_THREADS * S2_SPMD_DOWN_SHARD_ELEMS;
            pack_s2_spmd_weight(
                w.sh_gate, g_s2_spmd_gate_w + shared_gu_slot, D, H);
            pack_s2_spmd_weight(
                w.sh_up, g_s2_spmd_up_w + shared_gu_slot, D, H);
            pack_s2_spmd_weight(
                w.sh_down, g_s2_spmd_down_w + shared_down_slot, H, D);
            g_s2_spmd_ready = true;
            ensure_s2_spmd_pool();
        }
    }
#endif
    g_scale_shared_gate = w.sh_s_gate;
    g_scale_shared_up = w.sh_s_up;
    g_scale_shared_down = w.sh_s_down;

    // R10: router 权重 FP32->FP16（带宽降 2x）
    for (int e = 0; e < E; ++e) {
        const float* rw = w.w_router + (size_t)e * D;
        __m256i* wf = reinterpret_cast<__m256i*>(g_router_f16 + (size_t)e * D);
        for (int d = 0; d < D; d += 16) {
            __m256i h = _mm512_cvtps_ph(_mm512_loadu_ps(rw + d), _MM_FROUND_CUR_DIRECTION);
            _mm256_storeu_si256(wf + d / 16, h);
        }
    }

    // R12c: INT8 router weights for AMX GEMM (large-E path)
    for (int e = 0; e < E; ++e) {
        const float* rw = w.w_router + (size_t)e * D;
        float max_abs = 0.0f;
#if defined(__AVX512F__)
        {
            __m512 vmax = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
                __m512 v = _mm512_loadu_ps(rw + d);
                v = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), v);
                vmax = _mm512_max_ps(vmax, v);
            }
            max_abs = _mm512_reduce_max_ps(vmax);
        }
#else
        for (int d = 0; d < D; ++d)
            max_abs = std::max(max_abs, fabsf(rw[d]));
#endif
        g_scale_router[e] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv_scale = 1.0f / g_scale_router[e];
        int8_t* ri = g_router_int8 + (size_t)e * D;
#if defined(__AVX512F__)
        {
            const __m512 inv_s = _mm512_set1_ps(inv_scale);
            const __m512 lo = _mm512_set1_ps(-127.0f);
            const __m512 hi = _mm512_set1_ps(127.0f);
            for (int d = 0; d < D; d += 16) {
                __m512 v = _mm512_mul_ps(_mm512_loadu_ps(rw + d), inv_s);
                v = _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                v = _mm512_min_ps(hi, _mm512_max_ps(lo, v));
                const __m512i vi = _mm512_cvtps_epi32(v);
                const __m128i v8 = _mm512_cvtepi32_epi8(vi);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(ri + d), v8);
            }
        }
#else
        for (int d = 0; d < D; ++d) {
            long q = lrintf(rw[d] * inv_scale);
            q = std::max(-127L, std::min(127L, q));
            ri[d] = (int8_t)q;
        }
#endif
    }
    pack_output_major_weight(g_router_int8, g_packed_router, D, E);
    // R134: precompute 128 × row-sum for INT8 VNNI routing
    compute_output_major_row_sums(g_router_int8, g_sum_router_128, D, E);
    // R137: build interleaved VNNI-broadcast router for S1 single-ZMM routing.
    if (E <= 16) {
        for (int g = 0; g < D / 4; ++g) {
            int8_t* dst = g_router_vnni + (size_t)g * 64;
            for (int e = 0; e < E; ++e)
                std::memcpy(dst + (size_t)e * 4,
                             g_router_int8 + (size_t)e * D + g * 4, 4);
            for (int e = E; e < 16; ++e)
                std::memset(dst + (size_t)e * 4, 0, 4);
        }
    }

    g_amx_runtime_enabled = request_amx_permission();
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    g_last_tile_rows = -1;
#endif
}

// -----------------------------------------------------------------------------
// Router
// -----------------------------------------------------------------------------

/**
 * @brief 将一个专家候选插入当前 token 的有序 Top-K 列表。
 *
 * 专家按编号递增到达，且只有 score 严格更大时才前移，因此相同选择分数时
 * 自动保持较小专家编号优先。affinity 与带 bias 的选择分数同时保存，最终
 * 权重归一化只使用不含 bias 的 affinity。
 */
static inline void insert_routing_candidate(
    int expert, float affinity, float score, int K,
    int* best_idx, float* best_affinity, float* best_score) {
    int pos = K;
    for (int k = 0; k < K; ++k) {
        if (best_idx[k] < 0 || score > best_score[k]) {
            pos = k;
            break;
        }
    }
    if (pos == K) return;

    for (int k = K - 1; k > pos; --k) {
        best_idx[k] = best_idx[k - 1];
        best_affinity[k] = best_affinity[k - 1];
        best_score[k] = best_score[k - 1];
    }
    best_idx[pos] = expert;
    best_affinity[pos] = affinity;
    best_score[pos] = score;
}

/**
 * @brief 融合计算 Router logit、Sigmoid、Top-K 选择和路由权重归一化。
 *
 * 原实现先写完整 g_logits[N,E]，再写 g_affinity[N,E]，随后多次重新读取。
 * 当前实现对每个 token 在局部完成全部 Router 流程，只保留最终 K 个专家，
 * 从而消除两个大型中间矩阵以及相应的缓存/TLB流量。
 *
 * AVX-512 路径仍一次并行计算4个专家的点积并复用 x 向量；4个 logit 归约后
 * 按专家编号顺序计算精确 expf/Sigmoid，并立即插入局部 Top-K。
 */
#if defined(__AVX512F__)
// AVX-512 exp() via Cephes range-reduction + degree-6 polynomial. |x|<88, rel err<1e-7.
static inline __m512 exp512_ps(__m512 x) {
    const __m512 inv_ln2 = _mm512_set1_ps(1.4426950408889634f);
    const __m512 ln2     = _mm512_set1_ps(0.6931471805599453f);
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
    __m512 fx = _mm512_mul_ps(x, inv_ln2);
    __m512 n  = _mm512_cvtepi32_ps(_mm512_cvtps_epi32(fx));
    __m512 r  = _mm512_fnmadd_ps(n, ln2, x);
    __m512i ip = _mm512_add_epi32(_mm512_cvttps_epi32(n), _mm512_set1_epi32(127));
    __m512 pow2n = _mm512_castsi512_ps(_mm512_slli_epi32(ip, 23));
    // 7-order Taylor: exp(r)=1+r+r^2/2+...+r^6/720, r in [-0.347,0.347], err<1e-9
    __m512 pp = _mm512_set1_ps(1.3888888889E-3f);
    pp = _mm512_fmadd_ps(r, pp, _mm512_set1_ps(8.3333333333E-3f));
    pp = _mm512_fmadd_ps(r, pp, _mm512_set1_ps(4.1666666667E-2f));
    pp = _mm512_fmadd_ps(r, pp, _mm512_set1_ps(1.6666666667E-1f));
    pp = _mm512_fmadd_ps(r, pp, _mm512_set1_ps(5.0E-1f));
    pp = _mm512_fmadd_ps(r, pp, _mm512_set1_ps(1.0f));
    __m512 exp_r = _mm512_fmadd_ps(r, pp, _mm512_set1_ps(1.0f));
    return _mm512_mul_ps(pow2n, exp_r);
}
static inline __m512 silu512_ps(__m512 v) {
    __m512 neg = _mm512_xor_ps(v, _mm512_set1_ps(-0.0f));
    __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp512_ps(neg));
    // R107: rcp14 replaces div — 14-bit reciprocal (rel err < 6e-5) is far
    // below INT8 quantization tolerance (1/128 ~ 0.008). Latency 4+4=8c vs div 13-15c.
    // Unlike R98b (rcp14+NR, 3 instr), this uses fewer instructions than the original.
    return _mm512_mul_ps(v, _mm512_rcp14_ps(denom));
}
#endif

static inline void compute_routing_token(int t, const float* x, const MoEWeights& w,
                                         int D, int E, int K) {
        const float* x_t = x + (size_t)t * D;
        int best_idx[MAX_TOP_K] = {-1, -1, -1, -1};
        float best_affinity[MAX_TOP_K] = {};
        float best_score[MAX_TOP_K] = {};

#if defined(__AVX512F__)
        // R10: w_router 存 FP16（带宽降 2x），加载后 cvt FP32 再 FP32 fmadd（精度无损）
        int e = 0;
        for (; e + 3 < E; e += 4) {
            const __m256i* r0 = reinterpret_cast<const __m256i*>(g_router_f16 + (size_t)(e + 0) * D);
            const __m256i* r1 = reinterpret_cast<const __m256i*>(g_router_f16 + (size_t)(e + 1) * D);
            const __m256i* r2 = reinterpret_cast<const __m256i*>(g_router_f16 + (size_t)(e + 2) * D);
            const __m256i* r3 = reinterpret_cast<const __m256i*>(g_router_f16 + (size_t)(e + 3) * D);
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
                const __m512 xv = _mm512_loadu_ps(x_t + d);
                acc0 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(_mm256_loadu_si256(r0 + d / 16)), acc0);
                acc1 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(_mm256_loadu_si256(r1 + d / 16)), acc1);
                acc2 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(_mm256_loadu_si256(r2 + d / 16)), acc2);
                acc3 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(_mm256_loadu_si256(r3 + d / 16)), acc3);
            }
            // R12b: vectorize 4 sigmoid values using exp512_ps (one polynomial
            // instead of 4 scalar expf calls, ~8x faster on this path)
            const float logits[4] = {
                _mm512_reduce_add_ps(acc0), _mm512_reduce_add_ps(acc1),
                _mm512_reduce_add_ps(acc2), _mm512_reduce_add_ps(acc3)};
            __m512 lv = _mm512_castps128_ps512(_mm_loadu_ps(logits));
            __m512 neg_lv = _mm512_xor_ps(lv, _mm512_set1_ps(-0.0f));
            __m512 exp_v = exp512_ps(neg_lv);
            __m512 one_v = _mm512_set1_ps(1.0f);
            __m512 aff_v = _mm512_rcp14_ps(_mm512_add_ps(one_v, exp_v));
            float affinities[4];
            _mm_storeu_ps(affinities, _mm512_castps512_ps128(aff_v));
            for (int j = 0; j < 4; ++j) {
                insert_routing_candidate(e + j, affinities[j], affinities[j] + w.bias[e + j], K, best_idx, best_affinity, best_score);
            }
        }
        for (; e < E; ++e) {
            const __m256i* r_e = reinterpret_cast<const __m256i*>(g_router_f16 + (size_t)e * D);
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
                const __m512 xv = _mm512_loadu_ps(x_t + d);
                acc = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(_mm256_loadu_si256(r_e + d / 16)), acc);
            }
            const float logit = _mm512_reduce_add_ps(acc);
            const float affinity = 1.0f / (1.0f + expf(-logit));
            insert_routing_candidate(e, affinity, affinity + w.bias[e], K, best_idx, best_affinity, best_score);
        }
#else
        for (int e = 0; e < E; ++e) {
            const float* r_e = w.w_router + (size_t)e * D;
            float logit = 0.0f;
            for (int d = 0; d < D; ++d) logit += r_e[d] * x_t[d];
            const float affinity = 1.0f / (1.0f + expf(-logit));
            insert_routing_candidate(e, affinity, affinity + w.bias[e], K, best_idx, best_affinity, best_score);
        }
#endif

        float affinity_sum = 0.0f;
        for (int k = 0; k < K; ++k) affinity_sum += best_affinity[k];
        for (int k = 0; k < K; ++k) {
            g_topk_indices[t][k] = best_idx[k];
            g_topk_weights[t][k] = best_affinity[k] / affinity_sum;
        }
}

// Forward decl: small_m_matmul_output_major is defined later
static void small_m_matmul_output_major(const int8_t* A, const int8_t* W,
    const int32_t* row_sums, int32_t* C, int M, int K, int N);

// R134: INT8 VNNI routing for single-token S1 (E<=16, D=256).
// Replaces FP16 FMA dot-products with VPDPBUSD, cutting ~1000 cycles.
// Uses the same quantized input the expert GEMM already consumes, so no
// extra quantization error beyond the INT8 router weights.
static inline void compute_routing_int8_vnni(int t, const float* x,
                                              const MoEWeights& w,
                                              int D, int E, int K) {
    const float x_scale = g_x_scale[t];
    int32_t logits_i32[16];
#if defined(__AVX512VNNI__) && defined(__AVX512F__)
    if (E == 16) {
        // R137: single-ZMM broadcast routing - one VPDPBUSD per 4 inputs
        // produces all 16 expert logits (one per ZMM lane), eliminating the
        // 16 zmm_hsum_i32 horizontal reductions of the generic kernel.
        const int8_t* a = g_quantized_x + (size_t)t * D;
        const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
        __m512i acc = _mm512_setzero_si512();
        for (int g = 0; g < D / 4; ++g) {
            uint32_t word;
            std::memcpy(&word, a + 4 * g, 4);
            __m512i av = _mm512_xor_si512(
                _mm512_set1_epi32((int32_t)word), sign_flip);
            acc = _mm512_dpbusd_epi32(
                acc, av, _mm512_loadu_si512(
                            reinterpret_cast<const __m512i*>(
                                g_router_vnni + (size_t)g * 64)));
        }
        acc = _mm512_sub_epi32(acc, _mm512_loadu_si512(g_sum_router_128));
        _mm512_storeu_si512(logits_i32, acc);
    } else
#endif
    {
        small_m_matmul_output_major(
            g_quantized_x + (size_t)t * D, g_router_int8,
            g_sum_router_128, logits_i32, 1, D, E);
    }

    int best_idx[MAX_TOP_K] = {-1, -1, -1, -1};
    float best_affinity[MAX_TOP_K] = {};
    float best_score[MAX_TOP_K] = {};

#if defined(__AVX512F__)
    if (E == 16) {
        const __m512 xs = _mm512_set1_ps(x_scale);
        const __m512 rs = _mm512_loadu_ps(g_scale_router);
        const __m512 one = _mm512_set1_ps(1.0f);
        __m512 lf = _mm512_cvtepi32_ps(_mm512_loadu_si512(logits_i32));
        lf = _mm512_mul_ps(lf, _mm512_mul_ps(xs, rs));
        __m512 neg = _mm512_xor_ps(lf, _mm512_set1_ps(-0.0f));
        __m512 aff = _mm512_rcp14_ps(_mm512_add_ps(one, exp512_ps(neg)));
        alignas(64) float affinities[16];
        _mm512_store_ps(affinities, aff);
        for (int e = 0; e < 16; ++e)
            insert_routing_candidate(e, affinities[e],
                affinities[e] + w.bias[e], K, best_idx, best_affinity, best_score);
    } else
#endif
    {
        for (int e = 0; e < E; ++e) {
            const float logit = (float)logits_i32[e] * x_scale * g_scale_router[e];
            const float affinity = 1.0f / (1.0f + expf(-logit));
            insert_routing_candidate(e, affinity, affinity + w.bias[e],
                K, best_idx, best_affinity, best_score);
        }
    }

    float affinity_sum = 0.0f;
    for (int k = 0; k < K; ++k) affinity_sum += best_affinity[k];
    for (int k = 0; k < K; ++k) {
        g_topk_indices[t][k] = best_idx[k];
        g_topk_weights[t][k] = best_affinity[k] / affinity_sum;
    }
}

#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("align-functions=64")
#endif
static inline void compute_routing(const float* x, const MoEWeights& w,
                                   int num_tokens, int D, int E, int K) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(num_tokens >= OMP_TOKEN_THRESHOLD)
#endif
    for (int t = 0; t < num_tokens; ++t) compute_routing_token(t, x, w, D, E, K);
}
#if defined(__GNUC__)
#pragma GCC pop_options
#endif

/**
 * @brief 将 token 按其 Top-K 路由结果聚拢到各个专家的连续 token 列表中。
 *
 * 功能：构建专家 token 列表，并同步保存每个分发项的路由权重和原 Top-K
 * 槽位。这既把逐 token 调用转换为逐专家 GEMM，也消除了专家回写时的 K 次搜索。
 *
 * 实现思路：先清零 E 个专家的计数器，再遍历所有 token 的 K 个专家编号，
 * 将 token ID 追加到对应专家列表。随后每个专家可一次 Gather 出紧凑 A 矩阵，
 * 使 AMX 在 M 维同时处理多个 token，提高权重复用率。
 *
 * 向量化情况：未使用 AVX-512/AMX。本阶段主要是间接索引、计数和离散写入，
 * 不适合直接向量化；它为后续 AMX GEMM 创造连续矩阵布局。
 *
 * @param num_tokens token 数量。
 * @param E          专家数量。
 * @param K          每个 token 选择的专家数量。
 */
static inline void dispatch_tokens_to_experts(int num_tokens, int E, int K) {
    std::memset(g_expert_token_count, 0, (size_t)E * sizeof(int));
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = g_topk_indices[t][k];
            const int slot = g_expert_token_count[e]++;
            g_expert_token_list[e][slot] = t;
            g_expert_token_weight[e][slot] = g_topk_weights[t][k];
        }
    }
}

// -----------------------------------------------------------------------------
// Per-token activation quantization
// -----------------------------------------------------------------------------

/**
 * @brief 将 FP32 输入按 token 独立量化为 INT8，并记录 per-token scale。
 *
 * 功能：对每个 token 计算 max_abs，得到 s_x=max_abs/127；全零 token 使用
 * scale=1。随后执行 x_q=round(x/s_x)，并将结果限制到 [-127,127]。
 * 输出写入紧凑布局 g_quantized_x[t*D+d]，scale 写入 g_x_scale[t]。
 *
 * 实现思路：每个 token 分两遍处理。第一遍求绝对值最大值，第二遍执行除法、
 * 最近偶数舍入、饱和和窄化。使用除法而非倒数近似乘法，以尽量贴近参考实现
 * 在重量化边界附近的数值行为。
 *
 * 向量化情况：
 * - AVX-512 路径：ZMM 每次处理 16 个 FP32；使用向量绝对值、最大值归约、
 *   除法、roundscale、clamp 和 FP32->INT32 转换；最后写成 16 个 INT8；
 * - 标量路径：使用 fabsf/lrintf 完成同一量化语义。
 *
 * @param x          输入 FP32 token，布局 [num_tokens,D]。
 * @param num_tokens token 数量。
 * @param D          每个 token 的元素数量。
 */
static inline void quantize_input_token(int t, const float* x, int D) {
        const float* x_t = x + (size_t)t * D;
        int8_t* q_t = g_quantized_x + (size_t)t * D;

#if defined(__AVX512F__)
        __m512 max_vec = _mm512_setzero_ps();
        for (int d = 0; d < D; d += 16) {
            const __m512 v = _mm512_loadu_ps(x_t + d);
            max_vec = _mm512_max_ps(max_vec, _mm512_abs_ps(v));
        }
        const float max_abs = _mm512_reduce_max_ps(max_vec);
#else
        float max_abs = 0.0f;
        for (int d = 0; d < D; ++d) {
            max_abs = std::max(max_abs, fabsf(x_t[d]));
        }
#endif

        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        g_x_scale[t] = scale;

#if defined(__AVX512F__)
        // R132: Multiply by reciprocal (1 cycle) instead of divide (14 cycles).
        const __m512 inv_scale_vec = _mm512_set1_ps(1.0f / scale);
        const __m512 lo = _mm512_set1_ps(-127.0f);
        const __m512 hi = _mm512_set1_ps(127.0f);
        for (int d = 0; d < D; d += 16) {
            __m512 q = _mm512_mul_ps(_mm512_loadu_ps(x_t + d), inv_scale_vec);
            q = _mm512_roundscale_ps(
                q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            q = _mm512_min_ps(hi, _mm512_max_ps(lo, q));
            const __m512i q_i32 = _mm512_cvtps_epi32(q);
            // 16×INT32 直接窄化为 16×INT8，避免临时数组和 16 次标量写回。
            const __m128i q_i8 = _mm512_cvtepi32_epi8(q_i32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_t + d), q_i8);
        }
#else
        for (int d = 0; d < D; ++d) {
            long q = lrintf(x_t[d] / scale);
            q = std::max(-127L, std::min(127L, q));
            q_t[d] = (int8_t)q;
        }
#endif
}

static inline void quantize_input(const float* x, int num_tokens, int D) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(num_tokens >= OMP_TOKEN_THRESHOLD)
#endif
    for (int t = 0; t < num_tokens; ++t) quantize_input_token(t, x, D);
}

/**
 * @brief 将一行 SwiGLU FP32 隐藏激活重新量化为 INT8。
 *
 * 功能：计算 s_h=max(abs(h))/127，并执行 h_q=round(h/s_h)。该步骤是 W8A8
 * 流水线中 Gate/Up 的 FP32 SwiGLU 与 Down INT8 矩阵乘之间的重量化边界。
 *
 * 实现思路：SwiGLU 生成 hidden 时已经同步统计 max_abs，本函数直接计算 scale
 * 并完成除法、最近偶数舍入、饱和及 INT32→INT8 窄化，避免再次扫描求最大值。
 * 返回的 s_h 用于 Down 输出反量化：output_fp32=output_int32*s_h*s_down。
 *
 * 向量化情况：AVX-512 路径每次处理 16 个 FP32，并用 VPMOVDB 风格窄化一次
 * 写出 16 个 INT8；非 AVX-512 构建保留 lrintf 标量回退。
 *
 * @param src     一行 FP32 隐藏激活。
 * @param dst     输出 INT8 隐藏激活。
 * @param length  隐藏维度 H。
 * @param max_abs SwiGLU 计算阶段得到的该行绝对值最大值。
 * @return        当前行的激活 scale s_h。
 */
static inline float quantize_hidden_row(const float* src, int8_t* dst,
                                        int length, float max_abs) {
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
#if defined(__AVX512F__)
    // R25: full-precision reciprocal (1/scale) computed once per row replaces
    // the per-tile _mm512_div_ps with _mm512_mul_ps (~2x faster). Unlike rcp14
    // (14-bit, caused RMSE blowup in R23) this keeps full float precision, so
    // the INT8 round-to-nearest result is unaffected.
    const float rcp_scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
    const __m512 rcp_vec = _mm512_set1_ps(rcp_scale);
    const __m512 lo = _mm512_set1_ps(-127.0f);
    const __m512 hi = _mm512_set1_ps(127.0f);
    for (int i = 0; i < length; i += 16) {
        __m512 q = _mm512_mul_ps(_mm512_loadu_ps(src + i), rcp_vec);
        q = _mm512_roundscale_ps(
            q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q = _mm512_min_ps(hi, _mm512_max_ps(lo, q));
        const __m512i q_i32 = _mm512_cvtps_epi32(q);
        const __m128i q_i8 = _mm512_cvtepi32_epi8(q_i32);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), q_i8);
    }
#else
    for (int i = 0; i < length; ++i) {
        long q = lrintf(src[i] / scale);
        q = std::max(-127L, std::min(127L, q));
        dst[i] = (int8_t)q;
    }
#endif
    return scale;
}

// -----------------------------------------------------------------------------
// Small-M kernels
//
// 当 M<=4 时，AMX 的 16 行 Tile 严重欠填充。这里直接读取框架原始的输出通道
// 优先权重 W[N,K]，用 AVX-512BW 将 signed INT8 扩展到 INT16，再用 VPMADDWD
// 对相邻两项求和并累加到 INT32。一次同时处理 4 个输出通道以复用 A。
// -----------------------------------------------------------------------------

static constexpr int SMALL_M_THRESHOLD = 4;
// 仅在每行计算量足够大时启用 AVX-512BW 小 M kernel；避免 S1/S4 中
// 矩阵较小时，符号扩展与水平归约开销超过欠填充 AMX 的成本。
static constexpr size_t SMALL_M_WORK_THRESHOLD = 262144;
// 当前实测融合 Gate/Up AMX 在 S3/S4 略慢，保留实现但暂时关闭。
static constexpr bool USE_FUSED_GATE_UP_AMX = true;

static inline bool should_use_small_m_kernel(int M, int K, int N) {
#if defined(__AVX512VNNI__)
    if (M == 1) return true;
#endif
    return M <= SMALL_M_THRESHOLD &&
           (size_t)K * (size_t)N >= SMALL_M_WORK_THRESHOLD;
}

#if defined(__AVX512VNNI__) && defined(__GNUC__)
static inline __m512i dpbusd_accumulate_mem(
    __m512i accumulator, __m512i activations, const int8_t* weights) {
    asm("vpdpbusd %2, %1, %0"
        : "+v"(accumulator)
        : "v"(activations),
          "m"(*reinterpret_cast<const __m512i*>(weights)));
    return accumulator;
}
#endif

static inline __m512i s1_broadcast_u8x4(const int8_t* input) {
    uint32_t word;
    std::memcpy(&word, input, sizeof(word));
    return _mm512_set1_epi32((int32_t)(word ^ 0x80808080u));
}

static inline void s1_blocked_gate_up(
    const int8_t* A, const int8_t* W_gate, const int8_t* W_up,
    const int32_t* sum_gate, const int32_t* sum_up,
    int32_t* C_gate, int32_t* C_up, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 16) {
        __m512i gate0 = _mm512_setzero_si512();
        __m512i gate1 = _mm512_setzero_si512();
        __m512i gate2 = _mm512_setzero_si512();
        __m512i gate3 = _mm512_setzero_si512();
        __m512i up0 = _mm512_setzero_si512();
        __m512i up1 = _mm512_setzero_si512();
        __m512i up2 = _mm512_setzero_si512();
        __m512i up3 = _mm512_setzero_si512();
        const int8_t* gate_block = W_gate + (size_t)n0 * K;
        const int8_t* up_block = W_up + (size_t)n0 * K;
        for (int k4 = 0; k4 < K / 4; k4 += 4) {
            __m512i activations = s1_broadcast_u8x4(A + 4 * k4);
            gate0 = dpbusd_accumulate_mem(
                gate0, activations, gate_block + (size_t)(k4 + 0) * 64);
            up0 = dpbusd_accumulate_mem(
                up0, activations, up_block + (size_t)(k4 + 0) * 64);
            activations = s1_broadcast_u8x4(A + 4 * (k4 + 1));
            gate1 = dpbusd_accumulate_mem(
                gate1, activations, gate_block + (size_t)(k4 + 1) * 64);
            up1 = dpbusd_accumulate_mem(
                up1, activations, up_block + (size_t)(k4 + 1) * 64);
            activations = s1_broadcast_u8x4(A + 4 * (k4 + 2));
            gate2 = dpbusd_accumulate_mem(
                gate2, activations, gate_block + (size_t)(k4 + 2) * 64);
            up2 = dpbusd_accumulate_mem(
                up2, activations, up_block + (size_t)(k4 + 2) * 64);
            activations = s1_broadcast_u8x4(A + 4 * (k4 + 3));
            gate3 = dpbusd_accumulate_mem(
                gate3, activations, gate_block + (size_t)(k4 + 3) * 64);
            up3 = dpbusd_accumulate_mem(
                up3, activations, up_block + (size_t)(k4 + 3) * 64);
        }
        __m512i gate = _mm512_add_epi32(
            _mm512_add_epi32(gate0, gate1), _mm512_add_epi32(gate2, gate3));
        __m512i up = _mm512_add_epi32(
            _mm512_add_epi32(up0, up1), _mm512_add_epi32(up2, up3));
        gate = _mm512_sub_epi32(gate, _mm512_load_si512(sum_gate + n0));
        up = _mm512_sub_epi32(up, _mm512_load_si512(sum_up + n0));
        _mm512_store_si512(C_gate + n0, gate);
        _mm512_store_si512(C_up + n0, up);
    }
}

static inline void s1_blocked_matmul(
    const int8_t* A, const int8_t* W, const int32_t* row_sums,
    int32_t* C, int K, int N) {
    for (int n0 = 0; n0 < N; n0 += 16) {
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();
        const int8_t* block = W + (size_t)n0 * K;
        for (int k4 = 0; k4 < K / 4; k4 += 4) {
            __m512i activations = s1_broadcast_u8x4(A + 4 * k4);
            acc0 = dpbusd_accumulate_mem(
                acc0, activations, block + (size_t)(k4 + 0) * 64);
            activations = s1_broadcast_u8x4(A + 4 * (k4 + 1));
            acc1 = dpbusd_accumulate_mem(
                acc1, activations, block + (size_t)(k4 + 1) * 64);
            activations = s1_broadcast_u8x4(A + 4 * (k4 + 2));
            acc2 = dpbusd_accumulate_mem(
                acc2, activations, block + (size_t)(k4 + 2) * 64);
            activations = s1_broadcast_u8x4(A + 4 * (k4 + 3));
            acc3 = dpbusd_accumulate_mem(
                acc3, activations, block + (size_t)(k4 + 3) * 64);
        }
        __m512i acc = _mm512_add_epi32(
            _mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
        acc = _mm512_sub_epi32(acc, _mm512_load_si512(row_sums + n0));
        _mm512_store_si512(C + n0, acc);
    }
}

static void small_m_gate_up_output_major(
    const int8_t* A, const int8_t* W_gate, const int8_t* W_up,
    const int32_t* sum_gate, const int32_t* sum_up,
    int32_t* C_gate, int32_t* C_up, int M, int K, int N) {
#if defined(__AVX512VNNI__)
    if (M == 1) {
        const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
        for (int n0 = 0; n0 < N; n0 += 4) {
            __m512i gate0 = _mm512_setzero_si512();
            __m512i gate1 = _mm512_setzero_si512();
            __m512i gate2 = _mm512_setzero_si512();
            __m512i gate3 = _mm512_setzero_si512();
            __m512i up0 = _mm512_setzero_si512();
            __m512i up1 = _mm512_setzero_si512();
            __m512i up2 = _mm512_setzero_si512();
            __m512i up3 = _mm512_setzero_si512();
            const int8_t* gate0_w = W_gate + (size_t)n0 * K;
            const int8_t* up0_w = W_up + (size_t)n0 * K;
            for (int k0 = 0; k0 < K; k0 += 64) {
                const __m512i a_u8 = _mm512_xor_si512(
                    _mm512_loadu_si512(A + k0), sign_flip);
                gate0 = dpbusd_accumulate_mem(gate0, a_u8, gate0_w + k0);
                up0 = dpbusd_accumulate_mem(up0, a_u8, up0_w + k0);
                gate1 = dpbusd_accumulate_mem(gate1, a_u8, gate0_w + K + k0);
                up1 = dpbusd_accumulate_mem(up1, a_u8, up0_w + K + k0);
                gate2 = dpbusd_accumulate_mem(gate2, a_u8, gate0_w + 2 * K + k0);
                up2 = dpbusd_accumulate_mem(up2, a_u8, up0_w + 2 * K + k0);
                gate3 = dpbusd_accumulate_mem(gate3, a_u8, gate0_w + 3 * K + k0);
                up3 = dpbusd_accumulate_mem(up3, a_u8, up0_w + 3 * K + k0);
            }
            C_gate[n0 + 0] = zmm_hsum_i32(gate0) - sum_gate[n0 + 0];
            C_up[n0 + 0] = zmm_hsum_i32(up0) - sum_up[n0 + 0];
            C_gate[n0 + 1] = zmm_hsum_i32(gate1) - sum_gate[n0 + 1];
            C_up[n0 + 1] = zmm_hsum_i32(up1) - sum_up[n0 + 1];
            C_gate[n0 + 2] = zmm_hsum_i32(gate2) - sum_gate[n0 + 2];
            C_up[n0 + 2] = zmm_hsum_i32(up2) - sum_up[n0 + 2];
            C_gate[n0 + 3] = zmm_hsum_i32(gate3) - sum_gate[n0 + 3];
            C_up[n0 + 3] = zmm_hsum_i32(up3) - sum_up[n0 + 3];
        }
        return;
    }
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int n0 = 0; n0 < N; n0 += 4) {
            __m512i gate_acc[4] = {
                _mm512_setzero_si512(), _mm512_setzero_si512(),
                _mm512_setzero_si512(), _mm512_setzero_si512()};
            __m512i up_acc[4] = {
                _mm512_setzero_si512(), _mm512_setzero_si512(),
                _mm512_setzero_si512(), _mm512_setzero_si512()};

            for (int k0 = 0; k0 < K; k0 += 32) {
                const __m256i a_i8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_row + k0));
                const __m512i a_i16 = _mm512_cvtepi8_epi16(a_i8);

                for (int j = 0; j < 4; ++j) {
                    const int8_t* wg = W_gate + (size_t)(n0 + j) * K + k0;
                    const int8_t* wu = W_up + (size_t)(n0 + j) * K + k0;
                    const __m512i wg_i16 = _mm512_cvtepi8_epi16(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(wg)));
                    const __m512i wu_i16 = _mm512_cvtepi8_epi16(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(wu)));
                    gate_acc[j] = _mm512_add_epi32(
                        gate_acc[j], _mm512_madd_epi16(a_i16, wg_i16));
                    up_acc[j] = _mm512_add_epi32(
                        up_acc[j], _mm512_madd_epi16(a_i16, wu_i16));
                }
            }

            for (int j = 0; j < 4; ++j) {
               C_gate[(size_t)m * N + n0 + j] =
                   zmm_hsum_i32(gate_acc[j]);
               C_up[(size_t)m * N + n0 + j] =
                   zmm_hsum_i32(up_acc[j]);
            }
        }
    }
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t gate_acc = 0;
            int32_t up_acc = 0;
            for (int k = 0; k < K; ++k) {
                const int32_t a = (int32_t)A[(size_t)m * K + k];
                gate_acc += a * (int32_t)W_gate[(size_t)n * K + k];
                up_acc += a * (int32_t)W_up[(size_t)n * K + k];
            }
            C_gate[(size_t)m * N + n] = gate_acc;
            C_up[(size_t)m * N + n] = up_acc;
        }
    }
#endif
}

static void small_m_matmul_output_major(
    const int8_t* A, const int8_t* W, const int32_t* row_sums, int32_t* C,
    int M, int K, int N) {
#if defined(__AVX512VNNI__)
    if (M == 1) {
        const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
        for (int n0 = 0; n0 < N; n0 += 8) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i acc4 = _mm512_setzero_si512();
            __m512i acc5 = _mm512_setzero_si512();
            __m512i acc6 = _mm512_setzero_si512();
            __m512i acc7 = _mm512_setzero_si512();
            const int8_t* w0 = W + (size_t)n0 * K;
            for (int k0 = 0; k0 < K; k0 += 64) {
                const __m512i a_u8 = _mm512_xor_si512(
                    _mm512_loadu_si512(A + k0), sign_flip);
                acc0 = dpbusd_accumulate_mem(acc0, a_u8, w0 + k0);
                acc1 = dpbusd_accumulate_mem(acc1, a_u8, w0 + K + k0);
                acc2 = dpbusd_accumulate_mem(acc2, a_u8, w0 + 2 * K + k0);
                acc3 = dpbusd_accumulate_mem(acc3, a_u8, w0 + 3 * K + k0);
                acc4 = dpbusd_accumulate_mem(acc4, a_u8, w0 + 4 * K + k0);
                acc5 = dpbusd_accumulate_mem(acc5, a_u8, w0 + 5 * K + k0);
                acc6 = dpbusd_accumulate_mem(acc6, a_u8, w0 + 6 * K + k0);
                acc7 = dpbusd_accumulate_mem(acc7, a_u8, w0 + 7 * K + k0);
            }
            C[n0 + 0] = zmm_hsum_i32(acc0) - row_sums[n0 + 0];
            C[n0 + 1] = zmm_hsum_i32(acc1) - row_sums[n0 + 1];
            C[n0 + 2] = zmm_hsum_i32(acc2) - row_sums[n0 + 2];
            C[n0 + 3] = zmm_hsum_i32(acc3) - row_sums[n0 + 3];
            C[n0 + 4] = zmm_hsum_i32(acc4) - row_sums[n0 + 4];
            C[n0 + 5] = zmm_hsum_i32(acc5) - row_sums[n0 + 5];
            C[n0 + 6] = zmm_hsum_i32(acc6) - row_sums[n0 + 6];
            C[n0 + 7] = zmm_hsum_i32(acc7) - row_sums[n0 + 7];
        }
        return;
    }
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int n0 = 0; n0 < N; n0 += 4) {
            __m512i acc[4] = {
                _mm512_setzero_si512(), _mm512_setzero_si512(),
                _mm512_setzero_si512(), _mm512_setzero_si512()};

            for (int k0 = 0; k0 < K; k0 += 32) {
                const __m256i a_i8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_row + k0));
                const __m512i a_i16 = _mm512_cvtepi8_epi16(a_i8);
                for (int j = 0; j < 4; ++j) {
                    const int8_t* w_row = W + (size_t)(n0 + j) * K + k0;
                    const __m512i w_i16 = _mm512_cvtepi8_epi16(
                        _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(w_row)));
                    acc[j] = _mm512_add_epi32(
                        acc[j], _mm512_madd_epi16(a_i16, w_i16));
                }
            }

            for (int j = 0; j < 4; ++j) {
               C[(size_t)m * N + n0 + j] =
                   zmm_hsum_i32(acc[j]);
            }
        }
    }
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += (int32_t)A[(size_t)m * K + k] *
                       (int32_t)W[(size_t)n * K + k];
            }
            C[(size_t)m * N + n] = acc;
        }
    }
#endif
}

// -----------------------------------------------------------------------------
// AMX INT8 GEMM. The scalar branch is a build/runtime fallback; the algorithm
// and packed weight layout are identical, which also makes local verification
// possible on machines or compilers without AMX.
// -----------------------------------------------------------------------------

/**
 * @brief 使用标量指令计算与 AMX 打包格式完全一致的 INT8 GEMM。
 *
 * 功能：计算 C[M,N] = A[M,K] * B[K,N]，INT8 乘积在 INT32 中累加。
 * B 不以普通二维矩阵保存，而是读取 preprocess() 生成的 B_pack 布局。
 *
 * 实现思路：三重循环遍历 m、n、k，通过
 *     B_pack[(k/4)*(N*4) + 4*n + (k&3)]
 * 还原逻辑 B[k,n]。每个 C 元素使用独立 int32_t 累加器，K 最大为 1024，
 * 最坏情况下仍处于 INT32 安全范围内。
 *
 * 向量化情况：没有手写 SIMD/AMX。它是编译器未启用 AMX或运行时 AMX权限
 * 不可用时的正确性回退路径，也用于在普通开发机上验证打包布局。
 *
 * @param A      紧凑布局的左矩阵 [M,K]。
 * @param B_pack AMX/VNNI 打包后的右矩阵 [K/4,N*4]。
 * @param C      INT32 输出矩阵 [M,N]。
 */
static void scalar_matmul_packed(const int8_t* A, const int8_t* B_pack,
                                 int32_t* C, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                const int8_t b =
                    B_pack[(size_t)(k / 4) * (N * 4) + 4 * n + (k & 3)];
                acc += (int32_t)A[(size_t)m * K + k] * (int32_t)b;
            }
            C[(size_t)m * N + n] = acc;
        }
    }
}

#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
/**
 * @brief 在 64 字节 TILECFG 内存结构中设置一个 Tile 的行数和每行字节数。
 *
 * 功能：写入 tileN.colsb 与 tileN.rows 字段。colsb 使用两个字节的小端格式，
 * rows 使用一个字节。调用者负责保证 rows<=16、bytes_per_row<=64。
 *
 * 实现思路：按照 Intel TILECFG 的固定字节偏移写入配置，不执行 Tile 指令；
 * 最终由 configure_amx_tiles() 调用 _tile_loadconfig 一次性加载到硬件。
 *
 * 向量化情况：不涉及数值向量化，属于 AMX 控制状态配置辅助函数。
 */
static inline void set_tile_shape(int tile, int rows, int bytes_per_row) {
    g_tile_cfg[16 + 2 * tile] = (unsigned char)(bytes_per_row & 0xff);
    g_tile_cfg[17 + 2 * tile] =
        (unsigned char)((bytes_per_row >> 8) & 0xff);
    g_tile_cfg[48 + tile] = (unsigned char)rows;
}

/**
 * @brief 配置融合 Gate/Up 与普通 Down 共用的 AMX Tile 形状。
 *
 * 功能：Tile0/1 分别作为 Gate/Up 的 INT32 累加器，Tile2 保存 A，Tile3/4
 * 保存 Gate/Up 的打包权重。普通 Down 复用 Tile0、Tile2、Tile3。
 * rows_m 用于支持 token 数不是 16 倍数时的 M 维尾块。
 *
 * 实现思路：先将 TILECFG 全部清零，设置 palette_id=1，再调用
 * set_tile_shape() 填写三个 Tile，最后执行 _tile_loadconfig。若当前线程已经
 * 使用相同 rows_m，则直接复用配置。每个 Tile 均满足最多 16 行、每行 64 字节限制。
 *
 * 向量化情况：使用 AMX Tile 配置指令，但不执行乘加；实际矩阵运算由
 * amx_matmul_packed() 中的 _tile_dpbssd 完成。
 *
 * @param rows_m 当前 M 分块的有效 token 行数，范围为 1..16。
 */
static inline void configure_amx_tiles(int rows_m) {
    // Gate/Up/Down 的 Tile K/N 分块均固定为 64/16；只有 M 尾块行数变化。
    // 如果当前线程已经加载相同行数的配置，直接复用硬件 Tile 状态。
    if (g_last_tile_rows == rows_m) return;

    // 一份公共配置同时服务融合 Gate/Up 和普通 Down：
    //   tile0: Gate/普通 C，[rows_m,16] int32
    //   tile1: Up C，[rows_m,16] int32
    //   tile2: A，[rows_m,64] int8
    //   tile3: Gate/普通 B，[16,16*4] int8
    //   tile4: Up B，[16,16*4] int8
    std::memset(g_tile_cfg, 0, sizeof(g_tile_cfg));
    g_tile_cfg[0] = 1;
    set_tile_shape(0, rows_m, 64);
    set_tile_shape(1, rows_m, 64);
    set_tile_shape(2, rows_m, 64);
    set_tile_shape(3, 16, 64);
    set_tile_shape(4, 16, 64);
    _tile_loadconfig(g_tile_cfg);
    g_last_tile_rows = rows_m;
}

/**
 * @brief 使用 Intel AMX TDPBSSD 执行有符号 INT8 GEMM，并在 INT32 中累加。
 *
 * 功能：计算 C[M,N] = A[M,K] * B[K,N]。A 和权重均为 signed INT8，因而
 * 使用 _tile_dpbssd，而不是无符号/有符号混合版本。K 维多个 64 元素分块
 * 在 Tile 0 中持续累加，完成后一次性写回对应的 C 子块。
 *
 * 实现思路：
 * 1. M 维按 16 行分块，尾块通过 rows_m 动态配置；
 * 2. N 维按 16 个 INT32 输出列分块；
 * 3. 每个输出块先 _tile_zero(0)；
 * 4. K 维每次加载 64 个 INT8，B 对应加载 16 行×64 字节的四元素打包块；
 * 5. _tile_dpbssd(0,1,2) 完成 4-way INT8 点积并累加；
 * 6. _tile_stored 将 [rows_m,16] INT32 结果写回紧凑 C 矩阵。
 *
 * 向量化情况：显式使用 Intel AMX-INT8，是本实现主要的矩阵计算加速路径。
 * 实验保证 K、N 均为 64 的倍数，因此不需要 K/N 掩码尾处理。
 *
 * @param A      左矩阵 [M,K]，紧凑 INT8 布局。
 * @param B_pack 右矩阵的 AMX 四元素打包布局 [K/4,N*4]。
 * @param C      输出矩阵 [M,N]，元素为 int32_t。
 */
static void amx_matmul_packed(const int8_t* A, const int8_t* B_pack,
                              int32_t* C, int M, int K, int N) {
    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 16;

    // The framework guarantees K and N are multiples of 64.
    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        const int rows_m = std::min(TILE_M, M - m0);
        configure_amx_tiles(rows_m);

        // R111: 2N tiling — process 2 N-tiles per iteration.
        // A tile (tile 2) is loaded once and reused for both N-tiles,
        // halving A-tile loads and issuing 2 dpbssd per k-iter for ILP.
        for (int n0 = 0; n0 < N; n0 += 2 * TILE_N) {
            if (n0 + 2 * TILE_N <= N) {
                _tile_zero(0);
                _tile_zero(1);
                for (int k0 = 0; k0 < K; k0 += TILE_K) {
                    _tile_loadd(2, A + (size_t)m0 * K + k0, K);
                    _tile_loadd(3,
                                B_pack + (size_t)(k0 / 4) * (N * 4) + n0 * 4,
                                N * 4);
                    _tile_loadd(4,
                                B_pack + (size_t)(k0 / 4) * (N * 4) + (n0 + TILE_N) * 4,
                                N * 4);
                    _tile_dpbssd(0, 2, 3);
                    _tile_dpbssd(1, 2, 4);
                }
                _tile_stored(0, C + (size_t)m0 * N + n0, N * 4);
                _tile_stored(1, C + (size_t)m0 * N + n0 + TILE_N, N * 4);
            } else {
                // Remainder: single N-tile (odd N, not hit by S3/S4)
            _tile_zero(0);
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                _tile_loadd(2, A + (size_t)m0 * K + k0, K);
                _tile_loadd(3,
                            B_pack + (size_t)(k0 / 4) * (N * 4) + n0 * 4,
                            N * 4);
                _tile_dpbssd(0, 2, 3);
            }
            _tile_stored(0, C + (size_t)m0 * N + n0, N * 4);
            }
        }
    }
}

// Gate 和 Up 具有相同的 A/M/K/N。融合后每个 K-block 只加载一次 A，并用
// tile0/tile1 维护两条独立累加链，分别消费 Gate/Up 的打包权重。
static void amx_matmul_gate_up_packed(
    const int8_t* A, const int8_t* B_gate_pack, const int8_t* B_up_pack,
    int32_t* C_gate, int32_t* C_up, int M, int K, int N) {
    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 16;

    for (int m0 = 0; m0 < M; m0 += TILE_M) {
        const int rows_m = std::min(TILE_M, M - m0);
        configure_amx_tiles(rows_m);

        for (int n0 = 0; n0 < N; n0 += TILE_N) {
            _tile_zero(0);
            _tile_zero(1);
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                _tile_loadd(2, A + (size_t)m0 * K + k0, K);
                _tile_loadd(
                    3,
                    B_gate_pack + (size_t)(k0 / 4) * (N * 4) + n0 * 4,
                    N * 4);
                _tile_loadd(
                    4,
                    B_up_pack + (size_t)(k0 / 4) * (N * 4) + n0 * 4,
                    N * 4);
                _tile_dpbssd(0, 2, 3);
                _tile_dpbssd(1, 2, 4);
            }
            _tile_stored(0, C_gate + (size_t)m0 * N + n0, N * 4);
            _tile_stored(1, C_up + (size_t)m0 * N + n0, N * 4);
        }
    }
}
#endif

/**
 * @brief 在 AMX kernel 与标量回退 kernel 之间进行统一分派。
 *
 * 功能：向上层隐藏平台差异，使共享专家和路由专家始终调用同一矩阵乘接口。
 * 若编译器启用了 AMX 且运行时权限申请成功，则进入 amx_matmul_packed()；
 * 否则进入 scalar_matmul_packed()，两条路径使用完全相同的 B_pack 布局。
 *
 * 实现思路：通过编译期宏剔除不受支持的 intrinsic，再通过
 * g_amx_runtime_enabled 做运行时选择，从而兼顾目标 Sapphire Rapids 性能和
 * 普通开发环境的可编译、可验证性。
 *
 * 向量化情况：本函数自身不计算；其 AMX 分支显式向量/矩阵化，回退分支为标量。
 */
static void matmul_packed(const int8_t* A, const int8_t* B_pack, int32_t* C,
                          int M, int K, int N) {
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (g_amx_runtime_enabled) {
        amx_matmul_packed(A, B_pack, C, M, K, N);
        return;
    }
#endif
    scalar_matmul_packed(A, B_pack, C, M, K, N);
}

// Gate/Up 统一分派：仅当 M<=4 且 K×N 足够大时使用 AVX-512BW 小 M
// kernel；其余情况使用实测更快的分离 Gate/Up AMX。融合实现保留供后续 A/B。
static void matmul_gate_up(
    const int8_t* A,
    const int8_t* B_gate_pack, const int8_t* B_up_pack,
    const int8_t* W_gate, const int8_t* W_up,
    const int32_t* sum_gate, const int32_t* sum_up,
    int32_t* C_gate, int32_t* C_up,
    int M, int K, int N) {
    if (should_use_small_m_kernel(M, K, N)) {
        small_m_gate_up_output_major(
            A, W_gate, W_up, sum_gate, sum_up, C_gate, C_up, M, K, N);
        return;
    }
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    if (USE_FUSED_GATE_UP_AMX && g_amx_runtime_enabled && M <= 16) {
        amx_matmul_gate_up_packed(
            A, B_gate_pack, B_up_pack, C_gate, C_up, M, K, N);
        return;
    }
#endif
    // 实测分离 Gate/Up 在 S3/S4 更快；matmul_packed 会自行选择 AMX/标量。
    matmul_packed(A, B_gate_pack, C_gate, M, K, N);
    matmul_packed(A, B_up_pack, C_up, M, K, N);
}

// Down/普通单矩阵分派：M<=4 且 K×N 达到阈值时读取原始 W[N,K]；
// 否则保持打包 AMX 路径。
static void matmul_small_m_or_packed(
    const int8_t* A, const int8_t* B_pack, const int8_t* W,
    const int32_t* row_sums,
    int32_t* C, int M, int K, int N) {
    if (should_use_small_m_kernel(M, K, N)) {
        small_m_matmul_output_major(A, W, row_sums, C, M, K, N);
        return;
    }
    matmul_packed(A, B_pack, C, M, K, N);
}

// -----------------------------------------------------------------------------
// Shared and routed experts
// -----------------------------------------------------------------------------

/**
 * @brief 计算所有 token 都必须经过的共享 SwiGLU 专家，并累加到残差输出。
 *
 * 功能：依次完成共享专家的 Gate 投影、Up 投影、FP32 SwiGLU、隐藏激活重新
 * 量化、Down 投影和反量化累加。共享专家路由权重固定为 1。
 *
 * 实现思路：
 * 1. M<=4 且 K×N 足够大时使用 AVX-512BW；否则分离执行 Gate/Up AMX；
 * 2. 使用输入 scale 与权重 scale 将 INT32 投影反量化为 FP32；
 * 3. 计算 SiLU(gate)*up，并逐 token 重新量化为 INT8；
 * 4. 批量执行 INT8 Down GEMM；
 * 5. 乘以 s_h*s_down，并与输入 x 融合写入 y，省去单独残差 memcpy。
 * 所有矩阵缓冲区均使用运行时 D/H 的紧凑行跨度，避免 MAX_* 二维数组造成
 * 的物理 stride 错误。
 *
 * 向量化情况：常规 Gate/Up/Down 使用 AMX；仅对计算量足够大的小 M 使用
 * AVX-512BW 专用路径。Token 级 SwiGLU/输出合并使用 OpenMP，hidden 重量化
 * 和 Down 反量化使用 AVX-512；N<64 时自动保持串行。
 */
static void compute_shared_expert(const float* x, const MoEWeights& w,
                                  float* y, int num_tokens, int D, int H) {
    // R22: fuse 4 shared-expert OMP regions into 1 parallel{} to cut fork/join
    // overhead. VTune (S3): libgomp = 22.5% of CPU, top hotspot. Each omp for
    // retains an implicit barrier, preserving Gate/Up -> SwiGLU -> Down -> dequant
    // data dependencies. if(N>=64) serializes small batches with 1 thread (no
    // fork/join), avoiding code duplication and icache pressure.
    static thread_local bool shared_amx_perm = false;
#pragma omp parallel if(num_tokens >= OMP_TOKEN_THRESHOLD)
    {
        if (g_amx_runtime_enabled && !shared_amx_perm)
            shared_amx_perm = request_amx_permission();

        // --- Gate/Up GEMM (16-token blocks) ---
#pragma omp for schedule(static)
        for (int t0 = 0; t0 < num_tokens; t0 += 16) {
            const int m = (num_tokens - t0 < 16) ? (num_tokens - t0) : 16;
            matmul_gate_up(
                g_quantized_x + (size_t)t0 * D,
                g_packed_shared_gate, g_packed_shared_up,
                w.sh_gate, w.sh_up,
                g_sum_shared_gate, g_sum_shared_up,
                g_shared_gate_out + (size_t)t0 * H,
                g_shared_up_out + (size_t)t0 * H,
                m, D, H);
        }

        // --- SwiGLU + quantize ---
#pragma omp for schedule(static)
        for (int t = 0; t < num_tokens; ++t) {

        const float scale_g = g_scale_shared_gate * g_x_scale[t];
        const float scale_u = g_scale_shared_up * g_x_scale[t];
        float* h_row = g_shared_gated_fp32 + (size_t)t * H;
        int8_t* hq_row = g_shared_quantized_gated + (size_t)t * H;
        float max_abs = 0.0f;

        // 生成 SwiGLU hidden 的同时统计 max_abs，避免量化函数再次扫描一遍。
#if defined(__AVX512F__)
        {
            const __m512 sg = _mm512_set1_ps(scale_g);
            const __m512 su = _mm512_set1_ps(scale_u);
            const int32_t* go = g_shared_gate_out + (size_t)t * H;
            const int32_t* uo = g_shared_up_out   + (size_t)t * H;
            __m512 vmax = _mm512_setzero_ps();
            for (int h = 0; h < H; h += 16) {
                __m512 gate = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(go + h)), sg);
                __m512 up   = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(uo + h)), su);
                __m512 value = _mm512_mul_ps(silu512_ps(gate), up);
                _mm512_storeu_ps(h_row + h, value);
                vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value));
            }
            max_abs = _mm512_reduce_max_ps(vmax);
        }
#else
        for (int h = 0; h < H; ++h) {
            const float gate = (float)g_shared_gate_out[(size_t)t * H + h] * scale_g;
            const float up = (float)g_shared_up_out[(size_t)t * H + h] * scale_u;
            const float value = (gate / (1.0f + expf(-gate))) * up;
            h_row[h] = value;
            max_abs = std::max(max_abs, fabsf(value));
        }
#endif
        g_shared_gated_scale[t] =
            quantize_hidden_row(h_row, hq_row, H, max_abs);

        }

        // --- Down GEMM (16-token blocks) ---
#pragma omp for schedule(static)
        for (int t0 = 0; t0 < num_tokens; t0 += 16) {
            const int m = (num_tokens - t0 < 16) ? (num_tokens - t0) : 16;
            matmul_small_m_or_packed(
                g_shared_quantized_gated + (size_t)t0 * H,
                g_packed_shared_down, w.sh_down,
                g_sum_shared_down,
                g_shared_down_out + (size_t)t0 * D,
                m, H, D);
        }

        // --- Dequant + residual ---
#pragma omp for schedule(static) nowait
        for (int t = 0; t < num_tokens; ++t) {

        const float dequant = g_shared_gated_scale[t] * g_scale_shared_down;
        const float* x_t = x + (size_t)t * D;
        float* y_t = y + (size_t)t * D;
        const int32_t* out_t = g_shared_down_out + (size_t)t * D;
#if defined(__AVX512F__)
        const __m512 dequant_vec = _mm512_set1_ps(dequant);
        for (int d = 0; d < D; d += 16) {
            const __m512i out_i32 = _mm512_loadu_si512(out_t + d);
            const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
            const __m512 contribution = _mm512_mul_ps(out_fp32, dequant_vec);
            const __m512 x_vec = _mm512_loadu_ps(x_t + d);
            _mm512_storeu_ps(y_t + d, _mm512_add_ps(x_vec, contribution));
        }
#else
        for (int d = 0; d < D; ++d) {
            y_t[d] = x_t[d] + (float)out_t[d] * dequant;
        }
#endif

        }
    }
}

/**
 * @brief 执行完整的 DeepSeek-V3 风格 W8A8 MoE 前向计算。
 *
 * 功能：输出公式为
 *     y_t = x_t + shared_expert(x_t)
 *           + sum_{e in TopK(t)} gate[t,e] * expert_e(x_t)。
 * Router 使用 FP32；共享专家和路由专家的三个线性投影使用 INT8 权重、INT8
 * 激活和 INT32 累加；SiLU、scale 乘法以及最终专家加权在 FP32 中完成。
 *
 * 实现思路：
 * 1. 融合计算 Router logit、Sigmoid、Top-K 和路由权重；
 * 2. 输入按 token 量化；共享专家输出与残差 x 融合写入 y；
 * 4. N=1 时直接按 Top-K 计算路由专家并返回；
 * 5. N>1 时按专家分发并将命中 token Gather 到紧凑 g_amx_A；
 * 6. 对聚拢后的 token 批量执行 Gate/Up AMX GEMM、SwiGLU 和隐藏重量化；
 * 7. 执行 Down AMX GEMM，再按对应路由权重反量化并累加到原 token 输出。
 *
 * 向量化情况：
 * - Router 点积和输入量化在可用时使用 AVX-512；
 * - Gate/Up 使用融合 AMX；M<=4 的投影切换到 AVX-512BW 专用 kernel；
 * - Top-K、Sigmoid/SwiGLU 和分发保持标量；Down 反量化/加权使用 AVX-512；
 * - 未支持 AVX-512/AMX 的构建自动使用等价标量路径。
 *
 * @param x          FP32 输入，布局 [num_tokens,D]。
 * @param w          模型维度、Router、专家 INT8 权重及反量化 scale。
 * @param y          FP32 输出，布局 [num_tokens,D]。
 * @param num_tokens 当前批次 token 数量。
 */
/**
 * @brief 单 Token 路由专家快速路径。
 *
 * N=1 时无需构建专家计数/列表，也无需扫描全部 E 个专家或复制到 g_amx_A。
 * 直接按 Top-K 排名顺序计算被选中的专家；S1 直接调用分离 AMX，S2 仍保留
 * K×N 达阈值时的 AVX-512BW 小 M kernel。
 */
// R6: 单 token 路径按 K 专家并行。每线程用独立 tl_scratch(M=1)，
// 各专家把"反量化×路由权重"的输出写到独立 y_acc[k]，最后串行归约到 y。
// 复用 R5 的 thread_local ExpertScratch 与 AMX 权限缓存。
#if defined(_OPENMP)
alignas(64) static float g_single_yacc[MAX_TOP_K + 1][MAX_D_MODEL];

#if defined(__AVX512VNNI__)
alignas(64) static int32_t g_s2_gate[5][MAX_D_FF];
alignas(64) static int32_t g_s2_up[5][MAX_D_FF];
alignas(64) static float g_s2_hidden[5][MAX_D_FF];
alignas(64) static int8_t g_s2_hidden_q[5][MAX_D_FF];
static float g_s2_hidden_scale[5];
alignas(64) static int32_t g_s2_down[5][MAX_D_MODEL];

static inline __attribute__((aligned(64))) void s2_gate_up_range(
    const int8_t* xq, const int8_t* w_gate, const int8_t* w_up,
    const int32_t* sum_gate, const int32_t* sum_up, int32_t* gate_out,
    int32_t* up_out, int begin, int end) {
    const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    for (int n0 = begin; n0 < end; n0 += 4) {
        __m512i gate_acc[4];
        __m512i up_acc[4];
        for (int j = 0; j < 4; ++j) {
            gate_acc[j] = _mm512_setzero_si512();
            up_acc[j] = _mm512_setzero_si512();
        }
        for (int k0 = 0; k0 < 1024; k0 += 64) {
            const __m512i a_u8 = _mm512_xor_si512(
                _mm512_loadu_si512(xq + k0), sign_flip);
            for (int j = 0; j < 4; ++j) {
                const __m512i wg = _mm512_loadu_si512(
                    w_gate + (size_t)(n0 + j) * 1024 + k0);
                const __m512i wu = _mm512_loadu_si512(
                    w_up + (size_t)(n0 + j) * 1024 + k0);
                gate_acc[j] = _mm512_dpbusd_epi32(gate_acc[j], a_u8, wg);
                up_acc[j] = _mm512_dpbusd_epi32(up_acc[j], a_u8, wu);
            }
        }
       for (int j = 0; j < 4; ++j) {
           gate_out[n0 + j] = _mm512_reduce_add_epi32(gate_acc[j]) -
                              sum_gate[n0 + j];
           up_out[n0 + j] = _mm512_reduce_add_epi32(up_acc[j]) -
                            sum_up[n0 + j];
       }
    }
}

static inline __attribute__((aligned(64))) void s2_down_range(const int8_t* hq, const int8_t* w_down,
                                 const int32_t* row_sums, int32_t* down_out,
                                 int begin, int end) {
   const __m512i sign_flip = _mm512_set1_epi8((char)0x80);
    for (int n0 = begin; n0 < end; n0 += 4) {
        __m512i acc[4];
        for (int j = 0; j < 4; ++j) acc[j] = _mm512_setzero_si512();
        for (int k0 = 0; k0 < 512; k0 += 64) {
            const __m512i a_u8 = _mm512_xor_si512(
                _mm512_loadu_si512(hq + k0), sign_flip);
            for (int j = 0; j < 4; ++j) {
                const __m512i wv = _mm512_loadu_si512(
                    w_down + (size_t)(n0 + j) * 512 + k0);
                acc[j] = _mm512_dpbusd_epi32(acc[j], a_u8, wv);
            }
        }
       for (int j = 0; j < 4; ++j) {
           down_out[n0 + j] = _mm512_reduce_add_epi32(acc[j]) -
                               row_sums[n0 + j];
       }
    }
}

static __attribute__((aligned(64))) bool compute_single_token_s2_split(const float* x, const MoEWeights& w,
                                          float* y, int D, int H, int K) {
    constexpr int NUM_EXPERTS = 5;
    constexpr int NUM_CHUNKS = 3;
    constexpr int NUM_THREADS = NUM_EXPERTS * NUM_CHUNKS;
    if (D != 1024 || H != 512 || K != 4 || get_num_procs_cached() < NUM_THREADS)
        return false;

    const size_t expert_elems = (size_t)D * H;
    const int8_t* gate_w[NUM_EXPERTS] = {w.sh_gate};
    const int8_t* up_w[NUM_EXPERTS] = {w.sh_up};
    const int8_t* down_w[NUM_EXPERTS] = {w.sh_down};
    const int32_t* gate_sum[NUM_EXPERTS] = {g_sum_shared_gate};
    const int32_t* up_sum[NUM_EXPERTS] = {g_sum_shared_up};
    const int32_t* down_sum[NUM_EXPERTS] = {g_sum_shared_down};
    float gate_scale[NUM_EXPERTS] = {g_scale_shared_gate};
    float up_scale[NUM_EXPERTS] = {g_scale_shared_up};
    float down_scale[NUM_EXPERTS] = {g_scale_shared_down};
    float route_scale[NUM_EXPERTS] = {1.0f};
    for (int slot = 1; slot < NUM_EXPERTS; ++slot) {
        const int e = g_topk_indices[0][slot - 1];
        gate_w[slot] = w.w_gate + (size_t)e * expert_elems;
        up_w[slot] = w.w_up + (size_t)e * expert_elems;
        down_w[slot] = w.w_down + (size_t)e * expert_elems;
        gate_sum[slot] = g_sum_gate[e];
        up_sum[slot] = g_sum_up[e];
        down_sum[slot] = g_sum_down[e];
        gate_scale[slot] = g_scale_gate[e];
        up_scale[slot] = g_scale_up[e];
        down_scale[slot] = g_scale_down[e];
        route_scale[slot] = g_topk_weights[0][slot - 1];
    }

    const int8_t* xq = g_quantized_x;
    const float x_scale = g_x_scale[0];
    static constexpr int H_BOUNDS[NUM_CHUNKS + 1] = {0, 168, 336, 512};
    static constexpr int D_BOUNDS[NUM_CHUNKS + 1] = {0, 336, 672, 1024};
#pragma omp parallel num_threads(NUM_THREADS)
    {
        const int tid = omp_get_thread_num();
        const int slot = tid / NUM_CHUNKS;
        const int chunk = tid % NUM_CHUNKS;
        s2_gate_up_range(xq, gate_w[slot], up_w[slot], gate_sum[slot],
                         up_sum[slot], g_s2_gate[slot], g_s2_up[slot],
                         H_BOUNDS[chunk], H_BOUNDS[chunk + 1]);
#pragma omp barrier
        if (tid < NUM_EXPERTS) {
            float max_abs = 0.0f;
            const __m512 sg = _mm512_set1_ps(gate_scale[tid] * x_scale);
            const __m512 su = _mm512_set1_ps(up_scale[tid] * x_scale);
            __m512 vmax = _mm512_setzero_ps();
            for (int h = 0; h < H; h += 16) {
                const __m512 gate = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_loadu_si512(g_s2_gate[tid] + h)), sg);
                const __m512 up = _mm512_mul_ps(
                    _mm512_cvtepi32_ps(_mm512_loadu_si512(g_s2_up[tid] + h)), su);
                const __m512 value = _mm512_mul_ps(silu512_ps(gate), up);
                _mm512_storeu_ps(g_s2_hidden[tid] + h, value);
                vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(
                    _mm512_set1_ps(-0.0f), value));
            }
            max_abs = _mm512_reduce_max_ps(vmax);
            g_s2_hidden_scale[tid] = quantize_hidden_row(
                g_s2_hidden[tid], g_s2_hidden_q[tid], H, max_abs);
        }
#pragma omp barrier
        s2_down_range(g_s2_hidden_q[slot], down_w[slot], down_sum[slot],
                      g_s2_down[slot], D_BOUNDS[chunk],
                      D_BOUNDS[chunk + 1]);
    }

    // R107: Hoist dequant set1 out of d loop — scale is loop-invariant
    __m512 dequant_v[5];
    for (int slot = 0; slot < NUM_EXPERTS; ++slot)
        dequant_v[slot] = _mm512_set1_ps(
            g_s2_hidden_scale[slot] * down_scale[slot] * route_scale[slot]);
    for (int d = 0; d < D; d += 16) {
        __m512 acc = _mm512_loadu_ps(x + d);
        for (int slot = 0; slot < NUM_EXPERTS; ++slot) {
            acc = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_loadu_si512(g_s2_down[slot] + d)),
                dequant_v[slot], acc);
        }
        _mm512_storeu_ps(y + d, acc);
    }
    return true;
}

alignas(64) static float g_s2_spmd_local_max[S2_SPMD_THREADS][16];
static float g_s2_spmd_gate_scale[5];
static float g_s2_spmd_up_scale[5];
static float g_s2_spmd_down_scale[5];
static float g_s2_spmd_route_scale[5];
static float g_s2_spmd_quant_rcp[5];
static int g_s2_spmd_weight_slot[5];
static const float* g_s2_spmd_input = nullptr;
static float* g_s2_spmd_output = nullptr;

static std::thread g_s2_spmd_workers[S2_SPMD_THREADS - 1];
static std::atomic<uint32_t> g_s2_spmd_epoch{0};
static std::atomic<uint32_t> g_s2_spmd_done[S2_SPMD_THREADS - 1];
static std::atomic<int> g_s2_spmd_started{0};
static std::atomic<int> g_s2_spmd_barrier_count{0};
static std::atomic<uint32_t> g_s2_spmd_barrier_phase{0};
static bool g_s2_spmd_pool_started = false;

static inline void s2_spmd_barrier() {
    const uint32_t phase =
        g_s2_spmd_barrier_phase.load(std::memory_order_relaxed);
    if (g_s2_spmd_barrier_count.fetch_add(1, std::memory_order_acq_rel) ==
        S2_SPMD_THREADS - 1) {
        g_s2_spmd_barrier_count.store(0, std::memory_order_relaxed);
        g_s2_spmd_barrier_phase.fetch_add(1, std::memory_order_release);
    } else {
        while (g_s2_spmd_barrier_phase.load(std::memory_order_acquire) == phase)
            _mm_pause();
    }
}

static inline void s2_spmd_participant(int participant) {
    const int h_begin = participant * S2_SPMD_H_SHARD;
    const int d_begin = participant * S2_SPMD_D_SHARD;

    for (int slot = 0; slot < 5; ++slot) {
        const int weight_slot = g_s2_spmd_weight_slot[slot];
        const size_t packed_index =
            ((size_t)weight_slot * S2_SPMD_THREADS + participant) *
            S2_SPMD_GU_SHARD_ELEMS;
        const int32_t* gate_sum = weight_slot == S2_SPMD_E
            ? g_sum_shared_gate : g_sum_gate[weight_slot];
        const int32_t* up_sum = weight_slot == S2_SPMD_E
            ? g_sum_shared_up : g_sum_up[weight_slot];
        s1_blocked_gate_up(
            g_quantized_x, g_s2_spmd_gate_w + packed_index,
            g_s2_spmd_up_w + packed_index, gate_sum + h_begin,
            up_sum + h_begin, g_s2_gate[slot] + h_begin,
            g_s2_up[slot] + h_begin, S2_SPMD_D, S2_SPMD_H_SHARD);

        const __m512 gate_scale = _mm512_set1_ps(
            g_s2_spmd_gate_scale[slot] * g_x_scale[0]);
        const __m512 up_scale = _mm512_set1_ps(
            g_s2_spmd_up_scale[slot] * g_x_scale[0]);
        __m512 max_abs = _mm512_setzero_ps();
        for (int h = h_begin; h < h_begin + S2_SPMD_H_SHARD; h += 16) {
            const __m512 gate = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(g_s2_gate[slot] + h)),
                gate_scale);
            const __m512 up = _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(g_s2_up[slot] + h)),
                up_scale);
            const __m512 hidden = _mm512_mul_ps(silu512_ps(gate), up);
            _mm512_store_ps(g_s2_hidden[slot] + h, hidden);
            max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(hidden));
        }
        g_s2_spmd_local_max[participant][slot] =
            _mm512_reduce_max_ps(max_abs);
    }

    s2_spmd_barrier();
    if (participant == 0) {
        for (int slot = 0; slot < 5; ++slot) {
            float max_abs = g_s2_spmd_local_max[0][slot];
            for (int part = 1; part < S2_SPMD_THREADS; ++part)
                max_abs = std::max(max_abs, g_s2_spmd_local_max[part][slot]);
            g_s2_hidden_scale[slot] =
                max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            g_s2_spmd_quant_rcp[slot] =
                max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
        }
    }
    s2_spmd_barrier();

    const __m512 quant_lo = _mm512_set1_ps(-127.0f);
    const __m512 quant_hi = _mm512_set1_ps(127.0f);
    for (int slot = 0; slot < 5; ++slot) {
        const __m512 quant_rcp = _mm512_set1_ps(g_s2_spmd_quant_rcp[slot]);
        for (int h = h_begin; h < h_begin + S2_SPMD_H_SHARD; h += 16) {
            __m512 quantized = _mm512_mul_ps(
                _mm512_load_ps(g_s2_hidden[slot] + h), quant_rcp);
            quantized = _mm512_roundscale_ps(
                quantized, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            quantized = _mm512_min_ps(
                quant_hi, _mm512_max_ps(quant_lo, quantized));
            _mm_store_si128(
                reinterpret_cast<__m128i*>(g_s2_hidden_q[slot] + h),
                _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(quantized)));
        }
    }
    s2_spmd_barrier();

    for (int slot = 0; slot < 5; ++slot) {
        const int weight_slot = g_s2_spmd_weight_slot[slot];
        const size_t packed_index =
            ((size_t)weight_slot * S2_SPMD_THREADS + participant) *
            S2_SPMD_DOWN_SHARD_ELEMS;
        const int32_t* down_sum = weight_slot == S2_SPMD_E
            ? g_sum_shared_down : g_sum_down[weight_slot];
        s1_blocked_matmul(
            g_s2_hidden_q[slot], g_s2_spmd_down_w + packed_index,
            down_sum + d_begin, g_s2_down[slot] + d_begin,
            S2_SPMD_H, S2_SPMD_D_SHARD);
    }

    __m512 dequant[5];
    for (int slot = 0; slot < 5; ++slot) {
        dequant[slot] = _mm512_set1_ps(
            g_s2_hidden_scale[slot] * g_s2_spmd_down_scale[slot] *
            g_s2_spmd_route_scale[slot]);
    }
    for (int d = d_begin; d < d_begin + S2_SPMD_D_SHARD; d += 16) {
        __m512 result = _mm512_loadu_ps(g_s2_spmd_input + d);
        for (int slot = 0; slot < 5; ++slot) {
            result = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_load_si512(g_s2_down[slot] + d)),
                dequant[slot], result);
        }
        _mm512_storeu_ps(g_s2_spmd_output + d, result);
    }
}

static void s2_spmd_worker_loop(int participant) {
    uint32_t seen = g_s2_spmd_epoch.load(std::memory_order_relaxed);
    g_s2_spmd_started.fetch_add(1, std::memory_order_release);
    for (;;) {
        uint32_t current;
        while ((current = g_s2_spmd_epoch.load(std::memory_order_acquire)) == seen)
            _mm_pause();
        seen = current;
        s2_spmd_participant(participant);
        g_s2_spmd_done[participant - 1].store(current, std::memory_order_release);
    }
}

static void ensure_s2_spmd_pool() {
    if (g_s2_spmd_pool_started) return;
    g_s2_spmd_pool_started = true;
    for (int participant = 1; participant < S2_SPMD_THREADS; ++participant) {
        g_s2_spmd_workers[participant - 1] =
            std::thread(s2_spmd_worker_loop, participant);
        g_s2_spmd_workers[participant - 1].detach();
    }
    while (g_s2_spmd_started.load(std::memory_order_acquire) !=
           S2_SPMD_THREADS - 1)
        _mm_pause();
}

static bool compute_single_token_s2_spmd(
    const float* x, const MoEWeights& w, float* y, int D, int H, int K) {
    if (!g_s2_spmd_ready || D != S2_SPMD_D || H != S2_SPMD_H ||
        w.num_experts != S2_SPMD_E || K != MAX_TOP_K ||
        get_num_procs_cached() < S2_SPMD_THREADS)
        return false;

    g_s2_spmd_weight_slot[0] = S2_SPMD_E;
    g_s2_spmd_gate_scale[0] = g_scale_shared_gate;
    g_s2_spmd_up_scale[0] = g_scale_shared_up;
    g_s2_spmd_down_scale[0] = g_scale_shared_down;
    g_s2_spmd_route_scale[0] = 1.0f;
    for (int slot = 1; slot < 5; ++slot) {
        const int expert = g_topk_indices[0][slot - 1];
        g_s2_spmd_weight_slot[slot] = expert;
        g_s2_spmd_gate_scale[slot] = g_scale_gate[expert];
        g_s2_spmd_up_scale[slot] = g_scale_up[expert];
        g_s2_spmd_down_scale[slot] = g_scale_down[expert];
        g_s2_spmd_route_scale[slot] = g_topk_weights[0][slot - 1];
    }
    g_s2_spmd_input = x;
    g_s2_spmd_output = y;

    const uint32_t target =
        g_s2_spmd_epoch.fetch_add(1, std::memory_order_release) + 1;
    s2_spmd_participant(0);
    for (int worker = 0; worker < S2_SPMD_THREADS - 1; ++worker) {
        while (g_s2_spmd_done[worker].load(std::memory_order_acquire) != target)
            _mm_pause();
    }
    return true;
}
#endif

// R102: Persistent thread pool for S1 — eliminates OMP fork/join overhead.
// Each worker spin-waits on an epoch counter; main thread does task 0 in parallel.
static inline void s1_exec_task(
    const MoEWeights& w, int D, int H, int task,
    bool use_small_gate_up, bool use_small_down,
    ExpertScratch& scratch, float* y_acc) {
    const int8_t* xq = g_quantized_x;
    const float x_scale = g_x_scale[0];
    const size_t expert_elems = (size_t)D * H;
    const bool shared = task == 0;
    const int e = shared ? -1 : g_topk_indices[0][task - 1];
    const float route_weight = shared ? 1.0f : g_topk_weights[0][task - 1];
    const int8_t* w_gate_e = shared ? w.sh_gate
        : w.w_gate + (size_t)e * expert_elems;
    const int8_t* w_up_e = shared ? w.sh_up
        : w.w_up + (size_t)e * expert_elems;
    const int8_t* w_down_e = shared ? w.sh_down
        : w.w_down + (size_t)e * expert_elems;
    const int8_t* packed_gate_e = shared ? g_packed_shared_gate
        : g_packed_gate[e];
    const int8_t* packed_up_e = shared ? g_packed_shared_up
        : g_packed_up[e];
    const int8_t* packed_down_e = shared ? g_packed_shared_down
        : g_packed_down[e];
    const int32_t* sum_gate_e = shared ? g_sum_shared_gate : g_sum_gate[e];
    const int32_t* sum_up_e = shared ? g_sum_shared_up : g_sum_up[e];
    const int32_t* sum_down_e = shared ? g_sum_shared_down : g_sum_down[e];
    const float weight_scale_gate = shared ? g_scale_shared_gate : g_scale_gate[e];
    const float weight_scale_up = shared ? g_scale_shared_up : g_scale_up[e];
    const float weight_scale_down = shared ? g_scale_shared_down : g_scale_down[e];
    const bool use_s1_blocked = D == S1_PACK_D && H == S1_PACK_H;
    const int8_t* s1_gate_e = shared ? g_s1_shared_gate : g_s1_gate[e];
    const int8_t* s1_up_e = shared ? g_s1_shared_up : g_s1_up[e];
    const int8_t* s1_down_e = shared ? g_s1_shared_down : g_s1_down[e];

    if (use_s1_blocked) {
        s1_blocked_gate_up(
            xq, s1_gate_e, s1_up_e, sum_gate_e, sum_up_e,
            scratch.gate_out, scratch.up_out, D, H);
    } else if (use_small_gate_up) {
        small_m_gate_up_output_major(
            xq, w_gate_e, w_up_e,
            sum_gate_e, sum_up_e,
            scratch.gate_out, scratch.up_out, 1, D, H);
    } else {
        matmul_packed(xq, packed_gate_e, scratch.gate_out, 1, D, H);
        matmul_packed(xq, packed_up_e, scratch.up_out, 1, D, H);
    }

    const float scale_g = weight_scale_gate * x_scale;
    const float scale_u = weight_scale_up * x_scale;
    float max_abs = 0.0f;
#if defined(__AVX512F__)
    {
        const __m512 sg = _mm512_set1_ps(scale_g);
        const __m512 su = _mm512_set1_ps(scale_u);
        const int32_t* go = scratch.gate_out;
        const int32_t* uo = scratch.up_out;
        __m512 vmax = _mm512_setzero_ps();
        for (int h = 0; h < H; h += 16) {
            __m512 gate = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(go + h)), sg);
            __m512 up   = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(uo + h)), su);
            __m512 value = _mm512_mul_ps(silu512_ps(gate), up);
            _mm512_storeu_ps(scratch.gated_fp32 + h, value);
            vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value));
        }
        max_abs = _mm512_reduce_max_ps(vmax);
    }
#else
    for (int h = 0; h < H; ++h) {
        const float gate = (float)scratch.gate_out[h] * scale_g;
        const float up = (float)scratch.up_out[h] * scale_u;
        const float value = (gate / (1.0f + expf(-gate))) * up;
        scratch.gated_fp32[h] = value;
        max_abs = std::max(max_abs, fabsf(value));
    }
#endif
    const float hidden_scale = quantize_hidden_row(
        scratch.gated_fp32, scratch.quantized_gated, H, max_abs);

    if (use_s1_blocked) {
        s1_blocked_matmul(
            scratch.quantized_gated, s1_down_e, sum_down_e,
            scratch.down_out, H, D);
    } else if (use_small_down) {
        small_m_matmul_output_major(
            scratch.quantized_gated, w_down_e, sum_down_e,
            scratch.down_out, 1, H, D);
    } else {
        matmul_packed(scratch.quantized_gated, packed_down_e,
                      scratch.down_out, 1, H, D);
    }

    const float dequant = hidden_scale * weight_scale_down;
#if defined(__AVX512F__)
    const __m512 dequant_vec = _mm512_set1_ps(dequant);
    const __m512 weight_vec = _mm512_set1_ps(route_weight);
    for (int d = 0; d < D; d += 16) {
        const __m512i out_i32 = _mm512_loadu_si512(scratch.down_out + d);
        const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
        const __m512 dequantized = _mm512_mul_ps(out_fp32, dequant_vec);
        const __m512 contribution = _mm512_mul_ps(weight_vec, dequantized);
        _mm512_storeu_ps(y_acc + d, contribution);
    }
#else
    for (int d = 0; d < D; ++d)
        y_acc[d] = route_weight * ((float)scratch.down_out[d] * dequant);
#endif
}

struct S1Worker {
    std::atomic<uint32_t> done{0};
    ExpertScratch scratch;
    const MoEWeights* w;
    int D, H, task;
    bool use_small_gate_up, use_small_down;
    float* y_acc;
};

static S1Worker g_s1_wkrs[MAX_TOP_K];
static std::thread g_s1_thrds[MAX_TOP_K];
static int g_s1_pool_sz = 0;
static std::atomic<uint32_t> g_s1_epoch{0};
static const MoEWeights* g_s1_last_w = nullptr;

static void s1_worker_loop(int wid) {
    S1Worker& wk = g_s1_wkrs[wid];
    // R104: Pin each worker to a distinct core adjacent to the main thread
    pin_thread_to_core(wid + 1);
    if (g_amx_runtime_enabled && !wk.scratch.amx_perm)
        wk.scratch.amx_perm = request_amx_permission();
    uint32_t seen = 0;
    for (;;) {
        uint32_t cur;
        while ((cur = g_s1_epoch.load(std::memory_order_acquire)) == seen)
            _mm_pause();
        seen = cur;
        if (seen == 0xFFFFFFFFu) break;
        wk.scratch.ensure(1, (size_t)wk.D, (size_t)wk.H);
        s1_exec_task(*wk.w, wk.D, wk.H, wk.task,
                     wk.use_small_gate_up, wk.use_small_down,
                     wk.scratch, wk.y_acc);
        wk.done.store(seen, std::memory_order_release);
    }
}

static void ensure_s1_pool(int nworkers) {
    for (int i = g_s1_pool_sz; i < nworkers; i++) {
        g_s1_thrds[i] = std::thread(s1_worker_loop, i);
        g_s1_thrds[i].detach();
    }
    g_s1_pool_sz = nworkers;
}

static void compute_single_token_routed_experts_parallel(
    const float* x, const MoEWeights& w, float* y,
    int D, int H, int K, int nthreads) {
    const int8_t* xq = g_quantized_x;
    const float x_scale = g_x_scale[0];
    const size_t expert_elems = (size_t)D * H;
    const int use_small_gate_up = should_use_small_m_kernel(1, D, H);
    const int use_small_down = should_use_small_m_kernel(1, H, D);
    const int num_tasks = K + 1;
    const int nt = (nthreads < num_tasks) ? nthreads : num_tasks;

    // R102: Persistent thread pool path (one worker per routed expert).
    // Eliminates OMP fork/join overhead (~1-2us per call out of ~7us total).
    if (nt == num_tasks && num_tasks >= 2) {
        const int nworkers = num_tasks - 1;
        ensure_s1_pool(nworkers);

        // R130: Worker params are constant across calls (same weights, D, H,
        // task index, kernel selection, y_acc). Skip the setup loop after the
        // first call to avoid dirtying worker cache lines with cross-core writes.
        if (&w != g_s1_last_w) {
            for (int i = 0; i < nworkers; i++) {
                g_s1_wkrs[i].w = &w;
                g_s1_wkrs[i].D = D;
                g_s1_wkrs[i].H = H;
                g_s1_wkrs[i].task = i + 1;
                g_s1_wkrs[i].use_small_gate_up = use_small_gate_up;
                g_s1_wkrs[i].use_small_down = use_small_down;
                g_s1_wkrs[i].y_acc = g_single_yacc[i + 1];
            }
            g_s1_last_w = &w;
        }
        g_s1_epoch.fetch_add(1, std::memory_order_release);

        if (g_amx_runtime_enabled && !tl_scratch.amx_perm)
            tl_scratch.amx_perm = request_amx_permission();
        tl_scratch.ensure(1, (size_t)D, (size_t)H);
        s1_exec_task(w, D, H, 0, use_small_gate_up, use_small_down,
                     tl_scratch, g_single_yacc[0]);

        // R130: Hoist epoch load out of the wait loop.
        const uint32_t target_epoch = g_s1_epoch.load(std::memory_order_relaxed);
        for (int i = 0; i < nworkers; i++) {
            while (g_s1_wkrs[i].done.load(std::memory_order_acquire) != target_epoch)
                _mm_pause();
        }

#if defined(__AVX512F__)
        for (int d = 0; d < D; d += 16) {
            __m512 acc = _mm512_loadu_ps(x + d);
            for (int task = 0; task < num_tasks; ++task)
                acc = _mm512_add_ps(acc, _mm512_loadu_ps(g_single_yacc[task] + d));
            _mm512_storeu_ps(y + d, acc);
        }
#else
        for (int d = 0; d < D; ++d) {
            float acc = x[d];
            for (int task = 0; task < num_tasks; ++task)
                acc += g_single_yacc[task][d];
            y[d] = acc;
        }
#endif
        return;
    }

#pragma omp parallel num_threads(nt)
    {
        if (g_amx_runtime_enabled && !tl_scratch.amx_perm)
            tl_scratch.amx_perm = request_amx_permission();
        tl_scratch.ensure(1, (size_t)D, (size_t)H);

        const int tid = omp_get_thread_num();
        for (int task = tid; task < num_tasks; task += nt) {
            const bool shared = task == 0;
            const int e = shared ? -1 : g_topk_indices[0][task - 1];
            const float route_weight = shared ? 1.0f : g_topk_weights[0][task - 1];
            const int8_t* w_gate_e = shared ? w.sh_gate
                : w.w_gate + (size_t)e * expert_elems;
            const int8_t* w_up_e = shared ? w.sh_up
                : w.w_up + (size_t)e * expert_elems;
            const int8_t* w_down_e = shared ? w.sh_down
                : w.w_down + (size_t)e * expert_elems;
            const int8_t* packed_gate_e = shared ? g_packed_shared_gate
                : g_packed_gate[e];
            const int8_t* packed_up_e = shared ? g_packed_shared_up
                : g_packed_up[e];
            const int8_t* packed_down_e = shared ? g_packed_shared_down
                : g_packed_down[e];
            const int32_t* sum_gate_e = shared ? g_sum_shared_gate : g_sum_gate[e];
            const int32_t* sum_up_e = shared ? g_sum_shared_up : g_sum_up[e];
            const int32_t* sum_down_e = shared ? g_sum_shared_down : g_sum_down[e];
            const float weight_scale_gate = shared ? g_scale_shared_gate : g_scale_gate[e];
            const float weight_scale_up = shared ? g_scale_shared_up : g_scale_up[e];
            const float weight_scale_down = shared ? g_scale_shared_down : g_scale_down[e];
            float* y_acc = g_single_yacc[task];

            if (use_small_gate_up) {
                small_m_gate_up_output_major(
                    xq, w_gate_e, w_up_e,
                    sum_gate_e, sum_up_e,
                    tl_scratch.gate_out, tl_scratch.up_out, 1, D, H);
            } else {
                matmul_packed(xq, packed_gate_e, tl_scratch.gate_out, 1, D, H);
                matmul_packed(xq, packed_up_e, tl_scratch.up_out, 1, D, H);
            }

            const float scale_g = weight_scale_gate * x_scale;
            const float scale_u = weight_scale_up * x_scale;
            float max_abs = 0.0f;
#if defined(__AVX512F__)
            {
                const __m512 sg = _mm512_set1_ps(scale_g);
                const __m512 su = _mm512_set1_ps(scale_u);
                const int32_t* go = tl_scratch.gate_out;
                const int32_t* uo = tl_scratch.up_out;
                __m512 vmax = _mm512_setzero_ps();
                for (int h = 0; h < H; h += 16) {
                    __m512 gate = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(go + h)), sg);
                    __m512 up   = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(uo + h)), su);
                    __m512 value = _mm512_mul_ps(silu512_ps(gate), up);
                    _mm512_storeu_ps(tl_scratch.gated_fp32 + h, value);
                    vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value));
                }
                max_abs = _mm512_reduce_max_ps(vmax);
            }
#else
            for (int h = 0; h < H; ++h) {
                const float gate = (float)tl_scratch.gate_out[h] * scale_g;
                const float up = (float)tl_scratch.up_out[h] * scale_u;
                const float value = (gate / (1.0f + expf(-gate))) * up;
                tl_scratch.gated_fp32[h] = value;
                max_abs = std::max(max_abs, fabsf(value));
            }
#endif
            const float hidden_scale = quantize_hidden_row(
                tl_scratch.gated_fp32, tl_scratch.quantized_gated, H, max_abs);

            if (use_small_down) {
                small_m_matmul_output_major(
                    tl_scratch.quantized_gated, w_down_e, sum_down_e,
                    tl_scratch.down_out, 1, H, D);
            } else {
                matmul_packed(tl_scratch.quantized_gated, packed_down_e,
                              tl_scratch.down_out, 1, H, D);
            }

            const float dequant = hidden_scale * weight_scale_down;
#if defined(__AVX512F__)
            const __m512 dequant_vec = _mm512_set1_ps(dequant);
            const __m512 weight_vec = _mm512_set1_ps(route_weight);
            for (int d = 0; d < D; d += 16) {
                const __m512i out_i32 = _mm512_loadu_si512(tl_scratch.down_out + d);
                const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
                const __m512 dequantized = _mm512_mul_ps(out_fp32, dequant_vec);
                const __m512 contribution = _mm512_mul_ps(weight_vec, dequantized);
                _mm512_storeu_ps(y_acc + d, contribution);
            }
#else
            for (int d = 0; d < D; ++d)
                y_acc[d] = route_weight * ((float)tl_scratch.down_out[d] * dequant);
#endif
        }
    }
#if defined(__AVX512F__)
    for (int d = 0; d < D; d += 16) {
        __m512 acc = _mm512_loadu_ps(x + d);
        for (int task = 0; task < num_tasks; ++task)
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(g_single_yacc[task] + d));
        _mm512_storeu_ps(y + d, acc);
    }
#else
    for (int d = 0; d < D; ++d) {
        float acc = x[d];
        for (int task = 0; task < num_tasks; ++task)
            acc += g_single_yacc[task][d];
        y[d] = acc;
    }
#endif
}
#endif

static void compute_single_token_routed_experts(
    const float* x, const MoEWeights& w, float* y,
    int D, int H, int K, int nthreads) {
#if defined(_OPENMP)
    if (nthreads > 1 && K >= 2) {
        compute_single_token_routed_experts_parallel(
            x, w, y, D, H, K, nthreads);
        return;
    }
#else
    (void)nthreads;
#endif
    const int8_t* xq = g_quantized_x;
    const float x_scale = g_x_scale[0];
    const size_t expert_elems = (size_t)D * H;
    const bool use_small_gate_up = should_use_small_m_kernel(1, D, H);
    const bool use_small_down = should_use_small_m_kernel(1, H, D);

    for (int k = 0; k < K; ++k) {
        const int e = g_topk_indices[0][k];
        const float route_weight = g_topk_weights[0][k];
        const int8_t* w_gate_e = w.w_gate + (size_t)e * expert_elems;
        const int8_t* w_up_e = w.w_up + (size_t)e * expert_elems;
        const int8_t* w_down_e = w.w_down + (size_t)e * expert_elems;

        if (use_small_gate_up) {
            small_m_gate_up_output_major(
                xq, w_gate_e, w_up_e,
                g_sum_gate[e], g_sum_up[e],
                g_gate_out, g_up_out, 1, D, H);
        } else {
            // S1 直接进入分离 AMX/标量打包 kernel，跳过通用分派判断。
            matmul_packed(xq, g_packed_gate[e], g_gate_out, 1, D, H);
            matmul_packed(xq, g_packed_up[e], g_up_out, 1, D, H);
        }

        const float scale_g = g_scale_gate[e] * x_scale;
        const float scale_u = g_scale_up[e] * x_scale;
        float max_abs = 0.0f;
#if defined(__AVX512F__)
        {
            const __m512 sg = _mm512_set1_ps(scale_g);
            const __m512 su = _mm512_set1_ps(scale_u);
            __m512 vmax = _mm512_setzero_ps();
            for (int h = 0; h < H; h += 16) {
                __m512 gate = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(g_gate_out + h)), sg);
                __m512 up   = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(g_up_out + h)), su);
                __m512 value = _mm512_mul_ps(silu512_ps(gate), up);
                _mm512_storeu_ps(g_gated_fp32 + h, value);
                vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value));
            }
            max_abs = _mm512_reduce_max_ps(vmax);
        }
#else
        for (int h = 0; h < H; ++h) {
            const float gate = (float)g_gate_out[h] * scale_g;
            const float up = (float)g_up_out[h] * scale_u;
            const float value = (gate / (1.0f + expf(-gate))) * up;
            g_gated_fp32[h] = value;
            max_abs = std::max(max_abs, fabsf(value));
        }
#endif
        const float hidden_scale = quantize_hidden_row(
            g_gated_fp32, g_quantized_gated, H, max_abs);

        if (use_small_down) {
            small_m_matmul_output_major(
                g_quantized_gated, w_down_e, g_sum_down[e],
                g_down_out, 1, H, D);
        } else {
            matmul_packed(
                g_quantized_gated, g_packed_down[e],
                g_down_out, 1, H, D);
        }

        const float dequant = hidden_scale * g_scale_down[e];
#if defined(__AVX512F__)
        const __m512 dequant_vec = _mm512_set1_ps(dequant);
        const __m512 weight_vec = _mm512_set1_ps(route_weight);
        for (int d = 0; d < D; d += 16) {
            const __m512i out_i32 = _mm512_loadu_si512(g_down_out + d);
            const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
            const __m512 dequantized =
                _mm512_mul_ps(out_fp32, dequant_vec);
            const __m512 contribution =
                _mm512_mul_ps(weight_vec, dequantized);
            const __m512 y_vec = _mm512_loadu_ps(y + d);
            _mm512_storeu_ps(
                y + d, _mm512_add_ps(y_vec, contribution));
        }
#else
        for (int d = 0; d < D; ++d) {
            y[d] += route_weight * ((float)g_down_out[d] * dequant);
        }
#endif
    }
}

// -----------------------------------------------------------------------------
// R5: 按专家并行的路由专家计算
//
// 串行专家循环是 S3/S4 的主要瓶颈。本路径将 for(e) 拆分到多线程：每个线程
// 持有独立 GEMM scratch（A/gate_out/up_out/gated_fp32/quantized_gated/down_out）
// 与独立输出累加缓冲 y_acc[th]，避免竞争全局 g_amx_A 等；专家处理完毕后在
// 并行区内对所有 y_acc 切片求和并加回 y（y 已含共享专家输出与残差 x）。
//
// AMX Tile 配置 g_tile_cfg/g_last_tile_rows 已是 thread_local；但
// arch_prctl(ARCH_REQ_XCOMP_PERM) 是每线程状态，工作线程首次进入时须单独
// 申请 XTILEDATA 权限，否则执行 TDPBSSD 会触发 SIGILL。
// -----------------------------------------------------------------------------
#if defined(_OPENMP)

static float* g_yacc_pool = nullptr;
static size_t g_yacc_cap = 0;
// R204: S3 lazy accumulation leaves stale rows in each reusable y_acc slice.
// Track rows actually written by each worker so the reduction can skip them
// instead of first clearing every untouched row.
alignas(64) static uint8_t g_yacc_token_touched[8][MAX_NUM_TOKENS];
static float* acquire_yacc_pool(size_t need_elems) {
    if (need_elems > g_yacc_cap) {
        free(g_yacc_pool);
        g_yacc_pool = (float*)ExpertScratch::xalloc(need_elems * sizeof(float));
        g_yacc_cap = need_elems;
    }
    return g_yacc_pool;
}

static void compute_routed_experts_parallel(const MoEWeights& w, float* y,
                                            int num_tokens, int D, int H,
                                            int E, int nthreads) {
    int max_count = 1;
    for (int e = 0; e < E; ++e)
        if (g_expert_token_count[e] > max_count)
            max_count = g_expert_token_count[e];

    const size_t ystride = (size_t)num_tokens * D;
    float* yacc_pool = acquire_yacc_pool((size_t)nthreads * ystride);
    const size_t expert_elems = (size_t)D * H;

    // R32: LPT (longest-processing-time-first) expert ordering. Sort expert
    // indices by token count descending so schedule(dynamic) grabs the heaviest
    // experts first. With few experts per thread (S3: 16 experts / 8 threads =
    // 2 each) this pairs heavy+light per thread and cuts load-imbalance barrier
    // spin (VTune: libgomp ~14% of S3 CPU). Float-accumulation order changes by
    // <1e-7 (already non-deterministic under dynamic schedule), far below the
    // 2e-3 RMSE threshold, so correctness is preserved.
    int expert_order[MAX_NUM_EXPERTS];
    for (int e = 0; e < E; ++e) expert_order[e] = e;
    std::sort(expert_order, expert_order + E, [](int a, int b) {
        return g_expert_token_count[a] > g_expert_token_count[b];
    });
    // R203: preserve LPT order but avoid libgomp dynamic-workshare overhead.
    std::atomic<int> next_expert{0};

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();

        if (g_amx_runtime_enabled && !tl_scratch.amx_perm)
            tl_scratch.amx_perm = request_amx_permission();
        tl_scratch.ensure((size_t)max_count, (size_t)D, (size_t)H);

        float* y_acc = yacc_pool + (size_t)tid * ystride;
        // R87/R88: Lazy y_acc init only for small ystride (S3, <=L2).
        // For large ystride (S4, >L2): full memset is faster - lazy init's branch
        // overhead and zero-untouched pass add cost without L2 benefit.
        const bool use_lazy_init = (ystride < 131072);
        if (!use_lazy_init)
            std::memset(y_acc, 0, ystride * sizeof(float));
        uint8_t* token_touched = g_yacc_token_touched[tid];
        if (use_lazy_init)
            std::memset(token_touched, 0, (size_t)num_tokens);

        for (;;) {
            const int ii = next_expert.fetch_add(1, std::memory_order_relaxed);
            if (ii >= E) break;
            const int e = expert_order[ii];
            const int count = g_expert_token_count[e];
            if (count == 0) continue;

            int8_t* A = tl_scratch.A;
            for (int i = 0; i < count; ++i) {
                const int t = g_expert_token_list[e][i];
                std::memcpy(A + (size_t)i * D,
                            g_quantized_x + (size_t)t * D,
                            (size_t)D * sizeof(int8_t));
            }
            const int8_t* w_gate_e = w.w_gate + (size_t)e * expert_elems;
            const int8_t* w_up_e   = w.w_up   + (size_t)e * expert_elems;
            const int8_t* w_down_e = w.w_down + (size_t)e * expert_elems;

            matmul_gate_up(A, g_packed_gate[e], g_packed_up[e],
                           w_gate_e, w_up_e,
                           g_sum_gate[e], g_sum_up[e],
                           tl_scratch.gate_out, tl_scratch.up_out,
                           count, D, H);

            for (int i = 0; i < count; ++i) {
                const int t = g_expert_token_list[e][i];
                const float scale_g = g_scale_gate[e] * g_x_scale[t];
                const float scale_u = g_scale_up[e]   * g_x_scale[t];
                float* h_row  = tl_scratch.gated_fp32      + (size_t)i * H;
                int8_t* hq_row = tl_scratch.quantized_gated + (size_t)i * H;
                float max_abs = 0.0f;
#if defined(__AVX512F__)
                {
                    const __m512 sg = _mm512_set1_ps(scale_g);
                    const __m512 su = _mm512_set1_ps(scale_u);
                    const int32_t* go = tl_scratch.gate_out + (size_t)i * H;
                    const int32_t* uo = tl_scratch.up_out   + (size_t)i * H;
                    __m512 vmax = _mm512_setzero_ps();
                    for (int h = 0; h < H; h += 16) {
                        __m512 gate = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(go + h)), sg);
                        __m512 up   = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(uo + h)), su);
                        __m512 value = _mm512_mul_ps(silu512_ps(gate), up);
                        _mm512_storeu_ps(h_row + h, value);
                        vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value));
                    }
                    max_abs = _mm512_reduce_max_ps(vmax);
                }
#else
                for (int h = 0; h < H; ++h) {
                    const float gate = (float)tl_scratch.gate_out[(size_t)i * H + h] * scale_g;
                    const float up   = (float)tl_scratch.up_out[(size_t)i * H + h] * scale_u;
                    const float value = (gate / (1.0f + expf(-gate))) * up;
                    h_row[h] = value;
                    max_abs = std::max(max_abs, fabsf(value));
                }
#endif
                tl_scratch.gated_scale[i] =
                    quantize_hidden_row(h_row, hq_row, H, max_abs);
            }

            matmul_small_m_or_packed(
                tl_scratch.quantized_gated, g_packed_down[e], w_down_e,
                g_sum_down[e],
                tl_scratch.down_out, count, H, D);

            for (int i = 0; i < count; ++i) {
                const int t = g_expert_token_list[e][i];
                const float route_weight = g_expert_token_weight[e][i];
                const float dequant = tl_scratch.gated_scale[i] * g_scale_down[e];
                float* y_t = y_acc + (size_t)t * D;
                const int32_t* out_t = tl_scratch.down_out + (size_t)i * D;
#if defined(__AVX512F__)
                const __m512 dequant_vec = _mm512_set1_ps(dequant);
                const __m512 weight_vec = _mm512_set1_ps(route_weight);
                if (!use_lazy_init || token_touched[t]) {
                    for (int d = 0; d < D; d += 16) {
                        const __m512i out_i32 = _mm512_loadu_si512(out_t + d);
                        const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
                        const __m512 dequantized = _mm512_mul_ps(out_fp32, dequant_vec);
                        const __m512 contribution = _mm512_mul_ps(weight_vec, dequantized);
                        const __m512 y_vec = _mm512_loadu_ps(y_t + d);
                        _mm512_storeu_ps(y_t + d, _mm512_add_ps(y_vec, contribution));
                    }
                } else {
                    for (int d = 0; d < D; d += 16) {
                        const __m512i out_i32 = _mm512_loadu_si512(out_t + d);
                        const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
                        const __m512 dequantized = _mm512_mul_ps(out_fp32, dequant_vec);
                        const __m512 contribution = _mm512_mul_ps(weight_vec, dequantized);
                        _mm512_storeu_ps(y_t + d, contribution);
                    }
                    token_touched[t] = 1;
                }
#else
                if (!use_lazy_init || token_touched[t]) {
                    for (int d = 0; d < D; ++d)
                        y_t[d] += route_weight * ((float)out_t[d] * dequant);
                } else {
                    for (int d = 0; d < D; ++d)
                        y_t[d] = route_weight * ((float)out_t[d] * dequant);
                    token_touched[t] = 1;
                }
#endif
            }
        }

        // R204: the lazy S3 path leaves untouched rows stale. Their
        // per-worker touched bits are published at this barrier and the
        // reduction below skips them, avoiding a cache-polluting zero pass.
#pragma omp barrier

// R18b: threshold-based y_acc reduction.
        // Large ystride (S4: 524288 elems=2MB/slice > L2): per-thread sequential pass
        //   gives contiguous reads from yacc_pool (DRAM-BW friendly) -> +3.7% on S4.
        // Small ystride (S3: 32768 elems=128KB/slice <= L2): strided reduction keeps
        //   everything cache-resident and avoids nthreads-fold barrier overhead.
        if (ystride >= 131072) {
            for (int th = 0; th < nthreads; ++th) {
                const float* y_acc_th = yacc_pool + (size_t)th * ystride;
        #pragma omp for schedule(static)
                for (size_t idx = 0; idx < ystride; ++idx)
                    y[idx] += y_acc_th[idx];
            }
        } else {
            // S3: work in whole token rows so stale, untouched y_acc rows
            // need neither a memset nor a load during the reduction.
        #pragma omp for schedule(static) nowait
            for (int t = 0; t < num_tokens; ++t) {
                float* y_t = y + (size_t)t * D;
            #if defined(__AVX512F__)
                for (int d = 0; d < D; d += 16) {
                    __m512 sum = _mm512_setzero_ps();
                    for (int th = 0; th < nthreads; ++th) {
                        if (g_yacc_token_touched[th][t]) {
                            const float* src = yacc_pool + (size_t)th * ystride +
                                               (size_t)t * D + d;
                            sum = _mm512_add_ps(sum, _mm512_loadu_ps(src));
                        }
                    }
                    _mm512_storeu_ps(y_t + d,
                        _mm512_add_ps(_mm512_loadu_ps(y_t + d), sum));
                }
            #else
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int th = 0; th < nthreads; ++th) {
                        if (g_yacc_token_touched[th][t])
                            sum += yacc_pool[(size_t)th * ystride + (size_t)t * D + d];
                    }
                    y_t[d] += sum;
                }
            #endif
            }
        }
    }
}
#else
static inline void compute_routed_experts_parallel(const MoEWeights&, float*,
                                                   int, int, int, int, int) {}
#endif

// R12c: INT8 AMX router with vectorized FP16 refinement for large E.
// Uses AMX TDPBSSD to compute approximate INT32 logits, then vectorized
// sigmoid (exp512_ps, 16 at a time) for candidate selection, and FP16
// refinement of top-16 candidates for precision.
static void compute_routing_int8_amx(const float* x, const MoEWeights& w,
                                      int num_tokens, int D, int E, int K) {
    constexpr int TILE_M = 16;
    constexpr int TILE_K = 64;
    constexpr int TILE_N = 16;

    // Step 1: AMX INT8 GEMM: logits_int32[N, E] = quantized_x[N, D] * router_int8[D, E]
#if defined(__AMX_INT8__) && defined(__AMX_TILE__)
    static thread_local bool router_amx_perm = false;
#pragma omp parallel if(num_tokens >= OMP_TOKEN_THRESHOLD)
    {
        if (!router_amx_perm) router_amx_perm = request_amx_permission();
#pragma omp for schedule(static)
        for (int t0 = 0; t0 < num_tokens; t0 += TILE_M) {
            const int rows_m = std::min(TILE_M, num_tokens - t0);
            configure_amx_tiles(rows_m);
            const int8_t* A = g_quantized_x + (size_t)t0 * D;
            int32_t* C = g_router_logits + (size_t)t0 * E;
            // R175 candidate: router E=512 is also a wide AMX GEMM.
            // Reuse each A tile for two adjacent 16-expert output tiles.
            for (int n0 = 0; n0 < E; n0 += 2 * TILE_N) {
                _tile_zero(0);
                _tile_zero(1);
                for (int k0 = 0; k0 < D; k0 += TILE_K) {
                    _tile_loadd(2, A + k0, D);
                    const int8_t* b = g_packed_router +
                        (size_t)(k0 / 4) * (E * 4) + n0 * 4;
                    _tile_loadd(3, b, E * 4);
                    _tile_loadd(4, b + TILE_N * 4, E * 4);
                    _tile_dpbssd(0, 2, 3);
                    _tile_dpbssd(1, 2, 4);
                }
                _tile_stored(0, C + n0, E * 4);
                _tile_stored(1, C + n0 + TILE_N, E * 4);
            }
        }

        // Step 2 stays in this team: avoid a second OpenMP fork/join and
        // keep the router-logit buffer warm after AMX production.
        // R179 candidate: fused AMX Router and candidate-selection teams.
        constexpr int N_CAND = 4;
#pragma omp for schedule(static)
        for (int t = 0; t < num_tokens; ++t) {
        const float x_scale = g_x_scale[t];
        const float* x_t = x + (size_t)t * D;
        const int32_t* logits_i32 = g_router_logits + (size_t)t * E;

        int cand_idx[N_CAND];
        float cand_affinity[N_CAND];
        float cand_score[N_CAND];
        // R20: track top-N_CAND via a running minimum instead of a per-expert
        // sorted insert. Each expert needs only 1 compare (recompute the min of
        // the N_CAND only when a new candidate enters) -> ~7x fewer comparisons
        // and no shifts. VTune showed Step-2 candidate insertion was 22.8% of S4
        // CPU time (branchy sorted insert over 512 experts/token). The candidate
        // SET is identical to before (just unordered); Step-3 refinement is
        // order-independent, so correctness is preserved.
        float min_score = -1e30f;
        int min_pos = 0;
        for (int i = 0; i < N_CAND; ++i) {
            cand_idx[i] = -1;
            cand_affinity[i] = 0.0f;
            cand_score[i] = -1e30f;
        }

#if defined(__AVX512F__)
        const __m512 xs_vec = _mm512_set1_ps(x_scale);
        for (int e = 0; e < E; e += 16) {
            __m512 logits_v = _mm512_cvtepi32_ps(
                _mm512_loadu_si512(logits_i32 + e));
            logits_v = _mm512_mul_ps(logits_v, xs_vec);
            logits_v = _mm512_mul_ps(logits_v, _mm512_loadu_ps(g_scale_router + e));
            __m512 neg_v = _mm512_xor_ps(logits_v, _mm512_set1_ps(-0.0f));
            __m512 exp_v = exp512_ps(neg_v);
            __m512 one_v = _mm512_set1_ps(1.0f);
            __m512 aff_v = _mm512_div_ps(one_v, _mm512_add_ps(one_v, exp_v));
            __m512 score_v = _mm512_add_ps(aff_v, _mm512_loadu_ps(w.bias + e));
            float aff_arr[16], score_arr[16];
            _mm512_storeu_ps(aff_arr, aff_v);
            _mm512_storeu_ps(score_arr, score_v);
            for (int j = 0; j < 16; ++j) {
                const float s = score_arr[j];
                if (s > min_score) {
                    const int p = min_pos;
                    cand_idx[p] = e + j;
                    cand_affinity[p] = aff_arr[j];
                    cand_score[p] = s;
                    min_pos = 0;
                    min_score = cand_score[0];
                    for (int m = 1; m < N_CAND; ++m)
                        if (cand_score[m] < min_score) {
                            min_score = cand_score[m];
                            min_pos = m;
                        }
                }
            }
        }
#else
        for (int e = 0; e < E; ++e) {
            const float logit = (float)logits_i32[e] * x_scale * g_scale_router[e];
            const float affinity = 1.0f / (1.0f + expf(-logit));
            insert_routing_candidate(e, affinity, affinity + w.bias[e], N_CAND,
                                    cand_idx, cand_affinity, cand_score);
        }
#endif

        // Step 3: Refine top-N_CAND candidates with exact FP16 dot products
        int best_idx[MAX_TOP_K] = {-1, -1, -1, -1};
        float best_affinity[MAX_TOP_K] = {};
        float best_score[MAX_TOP_K] = {};

#if defined(__AVX512F__)
        // R174 candidate: N_CAND=4 lets FP16 refinement reuse each x ZMM
        // across four candidate rows and exposes four independent FMA chains.
        int c = 0;
        for (; c + 3 < N_CAND && cand_idx[c + 3] >= 0; c += 4) {
            const int e0 = cand_idx[c + 0];
            const int e1 = cand_idx[c + 1];
            const int e2 = cand_idx[c + 2];
            const int e3 = cand_idx[c + 3];
            const __m256i* r0 = reinterpret_cast<const __m256i*>(
                g_router_f16 + (size_t)e0 * D);
            const __m256i* r1 = reinterpret_cast<const __m256i*>(
                g_router_f16 + (size_t)e1 * D);
            const __m256i* r2 = reinterpret_cast<const __m256i*>(
                g_router_f16 + (size_t)e2 * D);
            const __m256i* r3 = reinterpret_cast<const __m256i*>(
                g_router_f16 + (size_t)e3 * D);
            __m512 a0 = _mm512_setzero_ps();
            __m512 a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps();
            __m512 a3 = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
                const __m512 xv = _mm512_loadu_ps(x_t + d);
                a0 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(
                    _mm256_loadu_si256(r0 + d / 16)), a0);
                a1 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(
                    _mm256_loadu_si256(r1 + d / 16)), a1);
                a2 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(
                    _mm256_loadu_si256(r2 + d / 16)), a2);
                a3 = _mm512_fmadd_ps(xv, _mm512_cvtph_ps(
                    _mm256_loadu_si256(r3 + d / 16)), a3);
            }
            const int experts[4] = {e0, e1, e2, e3};
            const float logits[4] = {_mm512_reduce_add_ps(a0),
                _mm512_reduce_add_ps(a1), _mm512_reduce_add_ps(a2),
                _mm512_reduce_add_ps(a3)};
            // R176 candidate: evaluate the four refinement sigmoids as one
            // AVX-512 vector instead of four scalar libm expf calls.
            const __m512 lv = _mm512_castps128_ps512(_mm_loadu_ps(logits));
            const __m512 one = _mm512_set1_ps(1.0f);
            const __m512 neg = _mm512_xor_ps(lv, _mm512_set1_ps(-0.0f));
            const __m512 affinity_v = _mm512_div_ps(one,
                _mm512_add_ps(one, exp512_ps(neg)));
            alignas(64) float affinities[16];
            _mm512_store_ps(affinities, affinity_v);
            for (int j = 0; j < 4; ++j) {
                const int e = experts[j];
                const float affinity = affinities[j];
                insert_routing_candidate(e, affinity, affinity + w.bias[e], K,
                                        best_idx, best_affinity, best_score);
            }
        }
        for (; c < N_CAND; ++c) {
            if (cand_idx[c] < 0) break;
            const int e = cand_idx[c];
            const __m256i* r_e = reinterpret_cast<const __m256i*>(
                g_router_f16 + (size_t)e * D);
            __m512 acc = _mm512_setzero_ps();
            for (int d = 0; d < D; d += 16) {
                const __m512 xv = _mm512_loadu_ps(x_t + d);
                acc = _mm512_fmadd_ps(xv,
                    _mm512_cvtph_ps(_mm256_loadu_si256(r_e + d / 16)), acc);
            }
            const float logit = _mm512_reduce_add_ps(acc);
            const float affinity = 1.0f / (1.0f + expf(-logit));
            insert_routing_candidate(e, affinity, affinity + w.bias[e], K,
                                    best_idx, best_affinity, best_score);
        }
#else
        for (int c = 0; c < N_CAND; ++c) {
            if (cand_idx[c] < 0) break;
            const int e = cand_idx[c];
            const float* r_e = w.w_router + (size_t)e * D;
            float logit = 0.0f;
            for (int d = 0; d < D; ++d) logit += r_e[d] * x_t[d];
            const float affinity = 1.0f / (1.0f + expf(-logit));
            insert_routing_candidate(e, affinity, affinity + w.bias[e], K,
                                    best_idx, best_affinity, best_score);
        }
#endif

        // Normalize routing weights
        float affinity_sum = 0.0f;
        for (int k = 0; k < K; ++k) affinity_sum += best_affinity[k];
        for (int k = 0; k < K; ++k) {
            g_topk_indices[t][k] = best_idx[k];
            g_topk_weights[t][k] = best_affinity[k] / affinity_sum;
        }
        }
    }
#else
    compute_routing(x, w, num_tokens, D, E, K);
    return;
#endif
}

#if defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("align-functions=64")
#endif
__attribute__((noinline, aligned(64)))
void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int D = w.d_model;
    const int H = w.d_ff;
    const int E = w.num_experts;
    const int K = w.top_k;

    // R132: Fast path for S1 (N=1, D=256, H=128, K>=2).
    // Skips omp_set_num_threads, s2_split check, and wrapper overhead.
    // R134: S1 fast path with INT8 VNNI routing.
    if (num_tokens == 1 && D == S1_PACK_D && H == S1_PACK_H && K >= 2) {
        // R139: One-pass quantization - buffer x in ZMM registers during the
        // max-abs scan so the conversion pass reads from registers (or L1
        // spill) instead of re-loading x from L2. Saves ~224 cyc on S1.
#if defined(__AVX512F__)
        {
            const float* x_t = x;
            int8_t* q_t = g_quantized_x;
            __m512 xb[16];
            __m512 vmax = _mm512_setzero_ps();
            for (int d = 0; d < 256; d += 16) {
                xb[d >> 4] = _mm512_loadu_ps(x_t + d);
                vmax = _mm512_max_ps(vmax, _mm512_abs_ps(xb[d >> 4]));
            }
            const float max_abs = _mm512_reduce_max_ps(vmax);
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            g_x_scale[0] = scale;
            const __m512 invs = _mm512_set1_ps(1.0f / scale);
            const __m512 lo = _mm512_set1_ps(-127.0f);
            const __m512 hi = _mm512_set1_ps(127.0f);
            for (int d = 0; d < 256; d += 16) {
                __m512 q = _mm512_mul_ps(xb[d >> 4], invs);
                q = _mm512_roundscale_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                q = _mm512_min_ps(hi, _mm512_max_ps(lo, q));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(q_t + d),
                    _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(q)));
            }
        }
#else
        quantize_input(x, 1, D);
#endif
        compute_routing_int8_vnni(0, x, w, D, E, K);
        compute_single_token_routed_experts_parallel(x, w, y, D, H, K, K + 1);
        return;
    }

#if defined(_OPENMP)
    // 根据批大小动态选择线程数，使各场景自动达到最优：
    //   N >= 512：使用全部可用核心（S4：16线程）
    //   N >= 64 ：限制为 8 线程（S3 实测最优）
    //   N < 64  ：单线程（S1、S2）
    // get_num_procs_cached() 返回作业分配的 CPU 数（如 -c 16 → 16），
    // 不受之前 omp_set_num_threads() 调用影响。
    int opt_threads = 1;
    {
        const int num_procs = get_num_procs_cached();
        if (num_tokens >= 512) {
            opt_threads = (num_procs < 8) ? num_procs : 8;
        } else if (num_tokens >= OMP_TOKEN_THRESHOLD) {
            opt_threads = (num_procs < 8) ? num_procs : 8;
        } else {
            // N < 64: pre-size the OMP pool for the single-token path.
            // S2 split (D>=1024,H>=512) uses exactly 15 threads; a 16th
            // idle thread causes measurable cache contention, so size to 15.
            // S1 uses K+1 threads but a larger pool (16) keeps threads warm
            // between calls, reducing fork/join wake-up latency.
            if (D >= 1024 && H >= 512) {
                opt_threads = 15;
            } else {
                opt_threads = (num_procs < 16) ? num_procs : 16;
            }
        }
        omp_set_num_threads(opt_threads);
    }
#else
    const int opt_threads = 1;
#endif

#if defined(_OPENMP)
    // R30: E<64 路径将 quantize_input 与 compute_routing 合并进同一 parallel 区域，
    // 消除一次 fork/join（VTune 显示 S3 libgomp 开销约 44%）。
    if (num_tokens >= OMP_TOKEN_THRESHOLD && !(E >= 64 && g_amx_runtime_enabled)) {
#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int t = 0; t < num_tokens; ++t) {
                quantize_input_token(t, x, D);
                compute_routing_token(t, x, w, D, E, K);
            }
        }
    } else {
        quantize_input(x, num_tokens, D);
        if (E >= 64 && g_amx_runtime_enabled) {
            compute_routing_int8_amx(x, w, num_tokens, D, E, K);
        } else {
            compute_routing(x, w, num_tokens, D, E, K);
        }
    }
#else
    quantize_input(x, num_tokens, D);
    if (E >= 64 && g_amx_runtime_enabled) {
        compute_routing_int8_amx(x, w, num_tokens, D, E, K);
    } else {
        compute_routing(x, w, num_tokens, D, E, K);
    }
#endif

#if defined(_OPENMP)
    if (num_tokens == 1) {
#if defined(__AVX512VNNI__)
        if (compute_single_token_s2_spmd(x, w, y, D, H, K)) return;
        if (compute_single_token_s2_split(x, w, y, D, H, K)) return;
#endif
        int k_threads = K + 1;
        const int np = get_num_procs_cached();
        if (k_threads > np) k_threads = np;
        if (k_threads < 1) k_threads = 1;
        compute_single_token_routed_experts(
            x, w, y, D, H, K, k_threads);
        return;
    }
#endif

    compute_shared_expert(x, w, y, num_tokens, D, H);

    // N=1 不需要按专家聚拢：直接按 Top-K 顺序计算，跳过计数器、列表、
    // 16/512 专家全扫描以及输入 Gather。
    if (num_tokens == 1) {
        compute_single_token_routed_experts(x, w, y, D, H, K, 1);
        return;
    }

    dispatch_tokens_to_experts(num_tokens, E, K);

    // 多 token 且多线程时按专家并行计算路由专家；否则走串行回退路径。
    if (opt_threads > 1) {
        compute_routed_experts_parallel(w, y, num_tokens, D, H, E, opt_threads);
        return;
    }

    for (int e = 0; e < E; ++e) {
        const int count = g_expert_token_count[e];
        if (count == 0) continue;

        for (int i = 0; i < count; ++i) {
            const int t = g_expert_token_list[e][i];
            std::memcpy(g_amx_A + (size_t)i * D,
                        g_quantized_x + (size_t)t * D,
                        (size_t)D * sizeof(int8_t));
        }

        const size_t expert_elems = (size_t)D * H;
        const int8_t* w_gate_e = w.w_gate + (size_t)e * expert_elems;
        const int8_t* w_up_e = w.w_up + (size_t)e * expert_elems;
        const int8_t* w_down_e = w.w_down + (size_t)e * expert_elems;

        matmul_gate_up(
            g_amx_A,
            g_packed_gate[e], g_packed_up[e],
            w_gate_e, w_up_e,
            g_sum_gate[e], g_sum_up[e],
            g_gate_out, g_up_out,
            count, D, H);

        for (int i = 0; i < count; ++i) {
            const int t = g_expert_token_list[e][i];
            const float scale_g = g_scale_gate[e] * g_x_scale[t];
            const float scale_u = g_scale_up[e] * g_x_scale[t];
            float* h_row = g_gated_fp32 + (size_t)i * H;
            int8_t* hq_row = g_quantized_gated + (size_t)i * H;
            float max_abs = 0.0f;

#if defined(__AVX512F__)
            {
                const __m512 sg = _mm512_set1_ps(scale_g);
                const __m512 su = _mm512_set1_ps(scale_u);
                const int32_t* go = g_gate_out + (size_t)i * H;
                const int32_t* uo = g_up_out   + (size_t)i * H;
                __m512 vmax = _mm512_setzero_ps();
                for (int h = 0; h < H; h += 16) {
                    __m512 gate = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(go + h)), sg);
                    __m512 up   = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(uo + h)), su);
                    __m512 value = _mm512_mul_ps(silu512_ps(gate), up);
                    _mm512_storeu_ps(h_row + h, value);
                    vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value));
                }
                max_abs = _mm512_reduce_max_ps(vmax);
            }
#else
            for (int h = 0; h < H; ++h) {
                const float gate = (float)g_gate_out[(size_t)i * H + h] * scale_g;
                const float up = (float)g_up_out[(size_t)i * H + h] * scale_u;
                const float value = (gate / (1.0f + expf(-gate))) * up;
                h_row[h] = value;
                max_abs = std::max(max_abs, fabsf(value));
            }
#endif
            g_gated_scale[i] =
                quantize_hidden_row(h_row, hq_row, H, max_abs);
        }

        matmul_small_m_or_packed(
            g_quantized_gated, g_packed_down[e], w_down_e,
            g_sum_down[e],
            g_down_out, count, H, D);

        for (int i = 0; i < count; ++i) {
            const int t = g_expert_token_list[e][i];
            const float route_weight = g_expert_token_weight[e][i];

            const float dequant = g_gated_scale[i] * g_scale_down[e];
            float* y_t = y + (size_t)t * D;
            const int32_t* out_t = g_down_out + (size_t)i * D;
#if defined(__AVX512F__)
            const __m512 dequant_vec = _mm512_set1_ps(dequant);
            const __m512 weight_vec = _mm512_set1_ps(route_weight);
            for (int d = 0; d < D; d += 16) {
                const __m512i out_i32 = _mm512_loadu_si512(out_t + d);
                const __m512 out_fp32 = _mm512_cvtepi32_ps(out_i32);
                const __m512 dequantized =
                    _mm512_mul_ps(out_fp32, dequant_vec);
                const __m512 contribution =
                    _mm512_mul_ps(weight_vec, dequantized);
                const __m512 y_vec = _mm512_loadu_ps(y_t + d);
                _mm512_storeu_ps(y_t + d,
                                 _mm512_add_ps(y_vec, contribution));
            }
#else
            for (int d = 0; d < D; ++d) {
                y_t[d] += route_weight * ((float)out_t[d] * dequant);
            }
#endif
        }
    }
}
#if defined(__GNUC__)
#pragma GCC pop_options
#endif
