// RISC-V Bonus: RVV routing plus SpaceMiT IME expert matrix multiplies.

#include "moe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>

namespace {

constexpr int IME_M = 4;
constexpr int IME_N = 4;
constexpr int IME_K = 8;
constexpr int IME_TILE_BYTES = IME_M * IME_K;
constexpr int MAX_RVV_THREADS = 4;

struct PackedMatrix {
    int8_t* data = nullptr;
    int rows = 0;
    int cols = 0;
    size_t matrix_bytes = 0;
};

static PackedMatrix g_gate;
static PackedMatrix g_up;
static PackedMatrix g_down;
static PackedMatrix g_shared_gate;
static PackedMatrix g_shared_up;
static PackedMatrix g_shared_down;

alignas(64) static int8_t g_xq[MAX_NUM_TOKENS][MAX_D_MODEL];
alignas(64) static float g_x_scale[MAX_NUM_TOKENS];
static int g_topk_index[MAX_NUM_TOKENS][MAX_TOP_K];
static float g_topk_weight[MAX_NUM_TOKENS][MAX_TOP_K];
static int g_expert_count[MAX_NUM_EXPERTS];
static int g_expert_tokens[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];
static float g_expert_weights[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];
static int g_expert_slots[MAX_NUM_EXPERTS][MAX_NUM_TOKENS];
alignas(64) static float
    g_routed_output[MAX_NUM_TOKENS][MAX_TOP_K][MAX_D_MODEL];

alignas(64) static float
    g_thread_output[MAX_RVV_THREADS][MAX_NUM_TOKENS][MAX_D_MODEL];

struct ExpertScratch {
    alignas(64) uint8_t input_pack[IME_M * MAX_D_MODEL];
    alignas(64) float hidden[IME_M][MAX_D_FF];
    alignas(64) int8_t hidden_q[IME_M][MAX_D_FF];
    alignas(64) uint8_t hidden_pack[IME_M * MAX_D_FF];
    alignas(64) int32_t gate_tile[IME_M * IME_N];
    alignas(64) int32_t up_tile[IME_M * IME_N];
    alignas(64) int32_t down_tile[IME_M * IME_N];
};

static thread_local ExpertScratch g_scratch;

static inline int rvv_thread_count() {
#if defined(_OPENMP)
    return std::min(MAX_RVV_THREADS, std::max(1, omp_get_num_procs()));
#else
    return 1;
#endif
}

static void pack_matrix(PackedMatrix& packed, const int8_t* source, int count,
                        int rows, int cols) {
    packed.rows = rows;
    packed.cols = cols;
    packed.matrix_bytes = (size_t)rows * cols;
    packed.data = new int8_t[(size_t)count * packed.matrix_bytes];

    const int col_tiles = cols / IME_K;
#pragma omp parallel for schedule(static) if (count > 1)
    for (int matrix = 0; matrix < count; ++matrix) {
        const int8_t* src = source + (size_t)matrix * rows * cols;
        int8_t* dst = packed.data + (size_t)matrix * rows * cols;
        for (int row_block = 0; row_block < rows / IME_N; ++row_block) {
            for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
                int8_t* tile = dst +
                    ((size_t)row_block * col_tiles + col_tile) * IME_TILE_BYTES;
                for (int row = 0; row < IME_N; ++row) {
                    std::memcpy(tile + row * IME_K,
                                src + (size_t)(row_block * IME_N + row) * cols +
                                    col_tile * IME_K,
                                IME_K);
                }
            }
        }
    }
}

static inline float rvv_dot_f32(const float* lhs, const float* rhs, int length) {
    float total = 0.0f;
    int offset = 0;
    while (offset < length) {
        size_t vl = __riscv_vsetvl_e32m8((size_t)(length - offset));
        vfloat32m8_t va = __riscv_vle32_v_f32m8(lhs + offset, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(rhs + offset, vl);
        vfloat32m8_t product = __riscv_vfmul_vv_f32m8(va, vb, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t sum =
            __riscv_vfredusum_vs_f32m8_f32m1(product, zero, vl);
        total += __riscv_vfmv_f_s_f32m1_f32(sum);
        offset += (int)vl;
    }
    return total;
}

static inline float rvv_amax_f32(const float* values, int length) {
    float maximum = 0.0f;
    int offset = 0;
    while (offset < length) {
        size_t vl = __riscv_vsetvl_e32m8((size_t)(length - offset));
        vfloat32m8_t value = __riscv_vle32_v_f32m8(values + offset, vl);
        value = __riscv_vfabs_v_f32m8(value, vl);
        vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(maximum, 1);
        vfloat32m1_t reduced =
            __riscv_vfredmax_vs_f32m8_f32m1(value, seed, vl);
        maximum = __riscv_vfmv_f_s_f32m1_f32(reduced);
        offset += (int)vl;
    }
    return maximum;
}

static inline void quantize_token(const float* token, int8_t* quantized,
                                  float& scale, int length) {
    float maximum = rvv_amax_f32(token, length);
    scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    float inverse = 1.0f / scale;
    for (int index = 0; index < length; ++index) {
        quantized[index] = (int8_t)lrintf(token[index] * inverse);
    }
}

static inline float fast_exp_f32(float value) {
    value = std::max(-80.0f, std::min(80.0f, value));
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2_HI = 0.693359375f;
    constexpr float LN2_LO = -2.12194440e-4f;
    int exponent = (int)lrintf(value * LOG2E);
    float reduced = (value - exponent * LN2_HI) - exponent * LN2_LO;
    float reduced2 = reduced * reduced;
    float polynomial =
        1.0f + reduced + reduced2 *
        (0.5f + reduced * (0.1666666716f + reduced *
        (0.0416666679f + reduced * (0.0083333338f +
        reduced * 0.0013888889f))));
    uint32_t bits = (uint32_t)(exponent + 127) << 23;
    float power_of_two;
    std::memcpy(&power_of_two, &bits, sizeof(power_of_two));
    return polynomial * power_of_two;
}

static inline void compute_token_routing(int token_index, const float* x,
                                         const MoEWeights& w) {
    const float* token = x + (size_t)token_index * w.d_model;
    float affinity[MAX_NUM_EXPERTS];
    int selected[MAX_TOP_K];
    float selected_score[MAX_TOP_K];
    for (int slot = 0; slot < w.top_k; ++slot) {
        selected[slot] = -1;
        selected_score[slot] = -INFINITY;
    }

    for (int expert = 0; expert < w.num_experts; ++expert) {
        float logit = rvv_dot_f32(
            w.w_router + (size_t)expert * w.d_model, token, w.d_model);
        float score = 1.0f / (1.0f + expf(-logit));
        affinity[expert] = score;
        float biased = score + w.bias[expert];
        for (int slot = 0; slot < w.top_k; ++slot) {
            if (biased > selected_score[slot]) {
                for (int move = w.top_k - 1; move > slot; --move) {
                    selected_score[move] = selected_score[move - 1];
                    selected[move] = selected[move - 1];
                }
                selected_score[slot] = biased;
                selected[slot] = expert;
                break;
            }
        }
    }

    float gate_sum = 0.0f;
    for (int slot = 0; slot < w.top_k; ++slot) {
        g_topk_index[token_index][slot] = selected[slot];
        gate_sum += affinity[selected[slot]];
    }
    float inverse_sum = 1.0f / gate_sum;
    for (int slot = 0; slot < w.top_k; ++slot) {
        g_topk_weight[token_index][slot] =
            affinity[selected[slot]] * inverse_sum;
    }
}

static inline void pack_four_rows(const int8_t* rows[IME_M], int cols,
                                  uint8_t* packed) {
    const int col_tiles = cols / IME_K;
    for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
        uint8_t* tile = packed + (size_t)col_tile * IME_TILE_BYTES;
        for (int row = 0; row < IME_M; ++row) {
            uint64_t value = 0;
            if (rows[row] != nullptr) {
                std::memcpy(&value, rows[row] + col_tile * IME_K, IME_K);
            }
            std::memcpy(tile + row * IME_K, &value, IME_K);
        }
    }
}

static inline void ime_gate_up_4x4(const uint8_t* input,
                                   const int8_t* gate_weight,
                                   const int8_t* up_weight, int col_tiles,
                                   int32_t* gate_output,
                                   int32_t* up_output) {
    const uint8_t* input_ptr = input;
    const int8_t* gate_ptr = gate_weight;
    const int8_t* up_ptr = up_weight;
    int tiles = col_tiles;
    asm volatile(
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vmv.v.i v28, 0\n\t"
        "vmv.v.i v30, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "1:\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[gate])\n\t"
        "vle8.v v2, (%[up])\n\t"
        ".word 0xe2103e2b\n\t"
        ".word 0xe2203f2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[gate], %[gate], 32\n\t"
        "addi %[up], %[up], 32\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[gate])\n\t"
        "vle8.v v2, (%[up])\n\t"
        ".word 0xe2103e2b\n\t"
        ".word 0xe2203f2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[gate], %[gate], 32\n\t"
        "addi %[up], %[up], 32\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[gate])\n\t"
        "vle8.v v2, (%[up])\n\t"
        ".word 0xe2103e2b\n\t"
        ".word 0xe2203f2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[gate], %[gate], 32\n\t"
        "addi %[up], %[up], 32\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[gate])\n\t"
        "vle8.v v2, (%[up])\n\t"
        ".word 0xe2103e2b\n\t"
        ".word 0xe2203f2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[gate], %[gate], 32\n\t"
        "addi %[up], %[up], 32\n\t"
        "addi %[tiles], %[tiles], -4\n\t"
        "bnez %[tiles], 1b\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vse32.v v28, (%[gate_output])\n\t"
        "vse32.v v30, (%[up_output])\n\t"
        : [input] "+r"(input_ptr), [gate] "+r"(gate_ptr),
          [up] "+r"(up_ptr), [tiles] "+r"(tiles)
        : [gate_output] "r"(gate_output), [up_output] "r"(up_output)
        : "t0", "memory");
}

static inline void ime_down_4x4(const uint8_t* input,
                                const int8_t* weight, int col_tiles,
                                int32_t* output) {
    const uint8_t* input_ptr = input;
    const int8_t* weight_ptr = weight;
    int tiles = col_tiles;
    asm volatile(
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vmv.v.i v28, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "1:\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[weight])\n\t"
        ".word 0xe2103e2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[weight], %[weight], 32\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[weight])\n\t"
        ".word 0xe2103e2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[weight], %[weight], 32\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[weight])\n\t"
        ".word 0xe2103e2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[weight], %[weight], 32\n\t"
        "vle8.v v0, (%[input])\n\t"
        "vle8.v v1, (%[weight])\n\t"
        ".word 0xe2103e2b\n\t"
        "addi %[input], %[input], 32\n\t"
        "addi %[weight], %[weight], 32\n\t"
        "addi %[tiles], %[tiles], -4\n\t"
        "bnez %[tiles], 1b\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vse32.v v28, (%[output])\n\t"
        : [input] "+r"(input_ptr), [weight] "+r"(weight_ptr),
          [tiles] "+r"(tiles)
        : [output] "r"(output)
        : "t0", "memory");
}

static inline void run_expert_block(
    const PackedMatrix& gate_matrix, const PackedMatrix& up_matrix,
    const PackedMatrix& down_matrix, int matrix_index, float gate_scale,
    float up_scale, float down_scale, const int token_indices[IME_M],
    const float combine_weight[IME_M], const int output_slots[IME_M],
    float* destination, int num_tokens, int d_model, int d_ff) {
    ExpertScratch& scratch = g_scratch;
    const int8_t* input_rows[IME_M];
    for (int row = 0; row < IME_M; ++row) {
        input_rows[row] = token_indices[row] >= 0
            ? g_xq[token_indices[row]]
            : nullptr;
    }
    pack_four_rows(input_rows, d_model, scratch.input_pack);

    const int8_t* gate_base =
        gate_matrix.data + (size_t)matrix_index * gate_matrix.matrix_bytes;
    const int8_t* up_base =
        up_matrix.data + (size_t)matrix_index * up_matrix.matrix_bytes;
    const int8_t* down_base =
        down_matrix.data + (size_t)matrix_index * down_matrix.matrix_bytes;
    const int input_tiles = d_model / IME_K;
    const int hidden_tiles = d_ff / IME_K;
    const size_t gate_block_stride = (size_t)input_tiles * IME_TILE_BYTES;
    const size_t down_block_stride = (size_t)hidden_tiles * IME_TILE_BYTES;
    float hidden_max[IME_M] = {};

    for (int output_block = 0; output_block < d_ff / IME_N;
         ++output_block) {
        ime_gate_up_4x4(scratch.input_pack,
                        gate_base + (size_t)output_block * gate_block_stride,
                        up_base + (size_t)output_block * gate_block_stride,
                        input_tiles, scratch.gate_tile, scratch.up_tile);
        for (int row = 0; row < IME_M; ++row) {
            if (token_indices[row] < 0) continue;
            float input_scale = g_x_scale[token_indices[row]];
            for (int column = 0; column < IME_N; ++column) {
                int hidden_index = output_block * IME_N + column;
                int tile_index = row * IME_N + column;
                int32_t gate_acc = scratch.gate_tile[tile_index];
                int32_t up_acc = scratch.up_tile[tile_index];
                float gate_value =
                    (float)gate_acc * (input_scale * gate_scale);
                float up_value =
                    (float)up_acc * (input_scale * up_scale);
                float hidden_value = gate_value /
                    (1.0f + fast_exp_f32(-gate_value)) * up_value;
                scratch.hidden[row][hidden_index] = hidden_value;
                hidden_max[row] =
                    std::max(hidden_max[row], fabsf(hidden_value));
            }
        }
    }

    float hidden_scale[IME_M];
    const int8_t* hidden_rows[IME_M];
    for (int row = 0; row < IME_M; ++row) {
        if (token_indices[row] < 0) {
            hidden_scale[row] = 1.0f;
            hidden_rows[row] = nullptr;
            continue;
        }
        hidden_scale[row] =
            hidden_max[row] > 0.0f ? hidden_max[row] / 127.0f : 1.0f;
        float inverse = 1.0f / hidden_scale[row];
        for (int hidden = 0; hidden < d_ff; ++hidden) {
            scratch.hidden_q[row][hidden] =
                (int8_t)lrintf(scratch.hidden[row][hidden] * inverse);
        }
        hidden_rows[row] = scratch.hidden_q[row];
    }
    pack_four_rows(hidden_rows, d_ff, scratch.hidden_pack);

    for (int output_block = 0; output_block < d_model / IME_N;
         ++output_block) {
        ime_down_4x4(scratch.hidden_pack,
                     down_base + (size_t)output_block * down_block_stride,
                     hidden_tiles, scratch.down_tile);
        for (int row = 0; row < IME_M; ++row) {
            int token_index = token_indices[row];
            if (token_index < 0) continue;
            float output_scale = hidden_scale[row] * down_scale *
                combine_weight[row];
            float* output = output_slots != nullptr
                ? g_routed_output[token_index][output_slots[row]]
                : destination + (size_t)token_index * d_model;
            for (int column = 0; column < IME_N; ++column) {
                int model_index = output_block * IME_N + column;
                int32_t accumulator =
                    scratch.down_tile[row * IME_N + column];
                float value = (float)accumulator * output_scale;
                if (output_slots != nullptr) {
                    output[model_index] = value;
                } else {
                    output[model_index] += value;
                }
            }
        }
    }
    (void)num_tokens;
}

static void compute_shared_expert(const MoEWeights& w, float* y,
                                  int num_tokens) {
    const int block_count = (num_tokens + IME_M - 1) / IME_M;
    const int threads = rvv_thread_count();
#pragma omp parallel for schedule(static) if (num_tokens >= 32) num_threads(threads)
    for (int block = 0; block < block_count; ++block) {
        int tokens[IME_M];
        float weights[IME_M];
        for (int row = 0; row < IME_M; ++row) {
            int token = block * IME_M + row;
            tokens[row] = token < num_tokens ? token : -1;
            weights[row] = 1.0f;
        }
        run_expert_block(g_shared_gate, g_shared_up, g_shared_down, 0,
                         w.sh_s_gate, w.sh_s_up, w.sh_s_down, tokens, weights,
                         nullptr, y, num_tokens, w.d_model, w.d_ff);
    }
}

static void dispatch_tokens(const MoEWeights& w, int num_tokens) {
    std::memset(g_expert_count, 0,
                (size_t)w.num_experts * sizeof(g_expert_count[0]));
    for (int token = 0; token < num_tokens; ++token) {
        for (int slot = 0; slot < w.top_k; ++slot) {
            int expert = g_topk_index[token][slot];
            int position = g_expert_count[expert]++;
            g_expert_tokens[expert][position] = token;
            g_expert_weights[expert][position] = g_topk_weight[token][slot];
            g_expert_slots[expert][position] = slot;
        }
    }
}

static void compute_routed_experts(const MoEWeights& w, float* y,
                                   int num_tokens) {
    if (num_tokens == 1 && w.top_k > 1) {
        const int threads = rvv_thread_count();
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (int expert = 0; expert < w.num_experts; ++expert) {
            if (g_expert_count[expert] == 0) continue;
            int tokens[IME_M] = {g_expert_tokens[expert][0], -1, -1, -1};
            float weights[IME_M] = {g_expert_weights[expert][0], 0.0f, 0.0f,
                                         0.0f};
            int slots[IME_M] = {g_expert_slots[expert][0], 0, 0, 0};
            run_expert_block(g_gate, g_up, g_down, expert,
                             w.s_gate[expert], w.s_up[expert],
                             w.s_down[expert], tokens, weights, slots, nullptr,
                             num_tokens, w.d_model, w.d_ff);
        }
        for (int model = 0; model < w.d_model; ++model) {
            float sum = 0.0f;
            for (int slot = 0; slot < w.top_k; ++slot) {
                sum += g_routed_output[0][slot][model];
            }
            y[model] += sum;
        }
        return;
    }

    const int threads = num_tokens >= 32 ? rvv_thread_count() : 1;
    const size_t output_elements = (size_t)num_tokens * w.d_model;
    if (threads == 1) {
        for (int expert = 0; expert < w.num_experts; ++expert) {
            for (int begin = 0; begin < g_expert_count[expert];
                 begin += IME_M) {
                int tokens[IME_M];
                float weights[IME_M];
                for (int row = 0; row < IME_M; ++row) {
                    int position = begin + row;
                    tokens[row] = position < g_expert_count[expert]
                        ? g_expert_tokens[expert][position]
                        : -1;
                    weights[row] = position < g_expert_count[expert]
                        ? g_expert_weights[expert][position]
                        : 0.0f;
                }
                run_expert_block(g_gate, g_up, g_down, expert,
                                 w.s_gate[expert], w.s_up[expert],
                                 w.s_down[expert], tokens, weights, nullptr, y,
                                 num_tokens, w.d_model, w.d_ff);
            }
        }
        return;
    }

    for (int thread = 0; thread < threads; ++thread) {
        std::memset(&g_thread_output[thread][0][0], 0,
                    output_elements * sizeof(float));
    }
#pragma omp parallel num_threads(threads)
    {
        int thread_index = omp_get_thread_num();
        float* thread_output = &g_thread_output[thread_index][0][0];
#pragma omp for schedule(dynamic, 1)
        for (int expert = 0; expert < w.num_experts; ++expert) {
            for (int begin = 0; begin < g_expert_count[expert];
                 begin += IME_M) {
                int tokens[IME_M];
                float weights[IME_M];
                for (int row = 0; row < IME_M; ++row) {
                    int position = begin + row;
                    tokens[row] = position < g_expert_count[expert]
                        ? g_expert_tokens[expert][position]
                        : -1;
                    weights[row] = position < g_expert_count[expert]
                        ? g_expert_weights[expert][position]
                        : 0.0f;
                }
                run_expert_block(g_gate, g_up, g_down, expert,
                                 w.s_gate[expert], w.s_up[expert],
                                 w.s_down[expert], tokens, weights,
                                 nullptr, thread_output, num_tokens, w.d_model,
                                 w.d_ff);
            }
        }
    }

#pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t index = 0; index < output_elements; ++index) {
        float sum = 0.0f;
        for (int thread = 0; thread < threads; ++thread) {
            sum += (&g_thread_output[thread][0][0])[index];
        }
        y[index] += sum;
    }
}

}  // namespace

void preprocess(MoEWeights& w) {
#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(rvv_thread_count());
#endif
    pack_matrix(g_gate, w.w_gate, w.num_experts, w.d_ff, w.d_model);
    pack_matrix(g_up, w.w_up, w.num_experts, w.d_ff, w.d_model);
    pack_matrix(g_down, w.w_down, w.num_experts, w.d_model, w.d_ff);
    pack_matrix(g_shared_gate, w.sh_gate, 1, w.d_ff, w.d_model);
    pack_matrix(g_shared_up, w.sh_up, 1, w.d_ff, w.d_model);
    pack_matrix(g_shared_down, w.sh_down, 1, w.d_model, w.d_ff);
}

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    const int threads = rvv_thread_count();
#pragma omp parallel for schedule(static) if (num_tokens >= 32) num_threads(threads)
    for (int token = 0; token < num_tokens; ++token) {
        const float* input = x + (size_t)token * w.d_model;
        compute_token_routing(token, x, w);
        quantize_token(input, g_xq[token], g_x_scale[token], w.d_model);
        std::memcpy(y + (size_t)token * w.d_model, input,
                    (size_t)w.d_model * sizeof(float));
    }

    compute_shared_expert(w, y, num_tokens);
    dispatch_tokens(w, num_tokens);
    compute_routed_experts(w, y, num_tokens);
}

#else

void preprocess(MoEWeights& w) { (void)w; }

void moe_forward_optimized(const float* x, const MoEWeights& w, float* y,
                           int num_tokens) {
    moe_forward_ref(x, w, y, num_tokens);
}

#endif
