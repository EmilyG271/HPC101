# Lab 5 Gemma4-12B 实验记录

更新时间：2026-08-28（Asia/Shanghai）。本文件用于记录集群资源、量化精度和端到端性能；集群测试完成后补齐实际 elapsed_s。

## 1. 集群资源（登录节点查询）

| 项目 | 配置 |
|---|---|
| SSH 入口 | clusters.zju.edu.cn:443 |
| 分区 | lab5 |
| GPU | H800 PCIe MIG 1g.10gb，单卡约 10 GiB |
| CPU | 4 cores |
| 主机内存 | 24 GiB |
| 最大墙钟 | 30 min |
| 并发作业上限 | 1 |
| 镜像 | harbor.clusters.zjusct.io/public/hpc101-lab5:v0.2 |
| 模型挂载 | /checkpoints/gemma-4-12b（只读） |

hpc partitions 于 2026-08-28 查询到：lab5 up，空闲 66 cores；lab5 支持 4 CPU、1 GPU、24 GiB 内存、30 min。

## 2. 量化配置

- 算法：GPTQ，W4A16，group_size=128，非对称量化，scale 为 FP16。
- 校准集：datasets/calibration-256.jsonl。
- 默认校准：256 条样本、micro batch 1、最多 4096 tokens。
- GPTQ：Hessian X^T X / N，对角阻尼 damp_percent=0.01，误差按逆 Hessian 顺序传播，block_size=128。
- 输出格式：packed uint8 little-nibble，manifest 与 safetensors checkpoint。

## 3. 性能测试约定

固定以下参数再比较：同一量化 checkpoint、同一 GPU、performance_public.jsonl、max-sequence-length=2048、max-new-tokens=32、seed=42、关闭进度条；以脚本摘要中的 elapsed_s 为 OJ 指标。

| 版本 | 参数/优化 | elapsed_s | 备注 |
|---|---|---:|---|
| 已有参考 | BF16/原始基线 | 参考结果 mean_nll=2.3084555301 | results/bf16-public-quality.json |
| 已完成 | GPTQ INT4 asymmetric + static batch 1 | 84.1439217550（performance_small，4 请求，60 token） | reference INT4 推理 |
| 已完成 | GPTQ + SDPA/Flash dispatch | 包含在上行结果 | CUDA fused SDPA 实际生效 |
| 已完成 | GPTQ + ring KV cache | 包含在上行结果 | sliding-attention 仅保留窗口 |
| 已完成 | performance_public + batch 1（旧参考实现） | 153.8663635399（5 请求，113 token） | 旧容器数据集/参考路径 |
| 已完成 | performance_public + batch 2（当前 Triton） | 165.2680295539（10 请求，273 token） | OJ 规模旧版本，低于 240 s |
| 已完成 | performance_public + batch 3 + tiled prefill + BM4 decode | 143.8994022560（10 请求，320 token） | 当前集群固定公开集，低于 240 s |
| 已完成 | performance_public + batch 3 + BM4 + CUDA Graph fail-safe | 143.8994022560（基准不回退） | CUDA Graph 捕获因 CPU/CUDA 非 pinned tensor 失败，自动 fallback，不影响正确性 |
| 已完成 | performance_public + batch 3（no-sync） | 144.6073916740（10 请求，320 token） | no-sync 无收益，最终保留同步指标 |
| 诊断 | 当前 attention backend | SDPA 8448 次，eager 0 次，fallback 0 次 | `HPC101_ATTENTION_LOG=1` |
| 失败记录 | performance_public + batch 4 | CUBLAS_STATUS_EXECUTION_FAILED / OOM | 10 GiB MIG 显存不足 |

## 4. 运行命令

    hpc submit -p lab5 -g 1 -t 30m --interactive bash
    uv pip install -e . --no-deps
    export MODEL_DIR=/checkpoints/gemma-4-12b
    export QUANT_DIR=$HOME/gemma-4-12b-gptq-w4a16
    python3 scripts/quantize.py --config config.yaml --model "$MODEL_DIR" --output "$QUANT_DIR" --device cuda --verbose
    python3 scripts/evaluate_quality.py --model "$QUANT_DIR" --dataset datasets/quality_public.jsonl --output results/quality-final.json --linear-backend int4_reference --no-progress
    python3 scripts/run_generation_queue.py --model "$QUANT_DIR" --input datasets/performance_public.jsonl --output results/generation-final.jsonl --summary-output results/generation-final-summary.json --config config.yaml --linear-backend int4_reference --batch-size 1 --max-sequence-length 2048 --no-progress

## 5. Profiler 结果与热点分析

完整模型 profiler 作业曾因模型加载和 profiler 开销超过 10 分钟墙钟而超时；因此使用相同 H800 MIG、相同 INT4 Triton kernel 和 SDPA 的代表性 4096x4096 decode microbenchmark 生成可复现 trace。

Trace 文件：`profiler-trace-20260831/j211544-bjmnq_1.1788139448795138081.pt.trace.json`。

完整模型 CUDA Graph 捕获诊断：`CUDA_GRAPH_CAPTURE_FAILED batch=1: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned`。因此当前版本保留 fail-safe fallback；不会在 OJ 上因不兼容环境直接失败。

Profiler 运行 10 次 fused INT4 GEMM 与 10 次 SDPA，主要结果：

- `_int4_gemm_kernel`：18.267 ms self CUDA，总 self CUDA 占比 99.14%；
- cuDNN/Flash SDPA kernel：约 0.143 ms CUDA，总占比约 0.77%；
- SDPA 没有 fallback，实际使用 fused CUDA attention；
- host 侧主要开销来自 profiler 同步和 CUDA launch，不是 attention 算子本身。

结论：当前主要热点是 decode 阶段的 INT4 GEMM，而不是 attention。后续优化优先级应是减少 Linear kernel launch 数量、融合相邻 projection 或使用更适合 M=2/3 的 tensor-core tile；不能继续把主要精力放在已经使用 fused SDPA 的 attention 上。当前 BM4 decode tile 已实际测得约 2.8% 的公开集改进；CUDA Graph 在当前 PyTorch/Triton 组合下因 pinned CPU tensor 限制不可用。

## 5. 结果

集群验证结果（H800 MIG 1g.10gb，作业 186041）：GPTQ 成功量化 328 个 Linear 模块，峰值主机内存约 5.66 GiB；公开质量集 INT4 mean_nll=2.4259347128，BF16 mean_nll=2.3084555301，因此 delta_nll=0.1174791827，满足硬门槛 delta_nll < 0.16。

小规模性能集（performance_small.jsonl）使用 batch 1 完成 4 个请求、生成 60 tokens，elapsed_s=84.1439217550，generated_tokens_per_s=0.7130639831。当前本地公开性能集包含 10 个请求、生成 273 tokens；使用 Triton decode kernel、SDPA 和 Ring KV Cache 后，当前代码在 batch 3 完成 10 请求公开集测试，elapsed_s=143.8994022560；batch 4 仍然 OOM，因此最终默认 batch 设置为 3。用户 OJ 版本报告的 273 token 规模预计可进一步低于该公开集的 320 token 结果。

## 6. 2026-09-01 端到端优化结果

本轮流 water 平台为 H800 MIG 1g.10gb，固定 `performance_public.jsonl`、batch=3、`max_sequence_length=2048`、`max_new_tokens=32`、seed=42、关闭进度条，并使用同一 GPTQ checkpoint。

### 6.1 Triton W4A16 与算子融合

按 GemLite split-K 思路为小 batch decode 增加 split-K 路径，并针对 H800 调整 tile。最终默认值为 `HPC101_INT4_SPLIT_K=16`、`BLOCK_N=128`、`BLOCK_K=16`、`num_warps=2`、`num_stages=3`。M<=4 且 N<=4096 时启用 split-K；N=14336 的 gate/up projection 测试为负收益，继续使用普通 tensor-core GEMM。

| Shape / path | 时间 |
|---|---:|
| q/o_proj，M=3，N=4096，K=4096，split-K=16 | 1.110 ms |
| kv_proj，M=3，N=1024，K=4096，split-K=16 | 0.301 ms |
| down_proj，M=3，N=4096，K=14336，split-K=16 | 3.199 ms |
| gate/up projection，M=3，N=14336，K=4096，split-K=1 | 3.605 ms |

`fused_gate_up_int4` 已实现并保留在 `HPC101_FUSED_GATE_UP=0` 后面；在最终 tile 下与两个独立 projection 基本持平，因此默认关闭。CUDA Graph 的 CPU/CUDA 拷贝问题已修复，batch=3 decode 可以成功 capture/replay，但 graph memory pool 会在长 prompt prefill 时触发 allocator OOM，因此最终默认 `HPC101_CUDA_GRAPH=0`。

### 6.2 Continuous batching 与 chunked prefill

连续调度使用 3 个物理 KV slot，请求完成后立即释放并补充；KVCache 支持 slot-aware reset/write/view/commit。`Runner` 在 continuous 模式下将整个队列交给 engine，而不是按每 3 条请求切分成静态小组。prefill 通过 `HPC101_PREFILL_CHUNK_SIZE` 分块执行。

| Chunk size | elapsed_s | generated_tokens |
|---:|---:|---:|
| 768 | 95.0668 | 320 |
| 896 | 97.3719 | 316 |
| 1024 | 91.7657 | 320 |
| 1152 | 82.9590 | 273 |
| 1280 | 83.7076 | 273 |
| 1536 | 82.9622 | 273 |

最终选择 `HPC101_PREFILL_CHUNK_SIZE=1152`：它是最快配置，同时比 1536 的激活峰值更小。连续运行生成 273 tokens，与原始 static/OJ 结果一致；1024 以下的部分运行生成 320 tokens，说明 chunk 边界会影响个别 argmax/EOS 选择，不能作为最终配置。

同一旧 tile、同一提交顺序下，static batch 为 148.4104s/320 tokens，组内 continuous 为 147.3095s/320 tokens；二者 token 总数相同。逐 token 对比仅在第 8 条请求出现首个 token 差异，原因是 continuous 将组内 32-token 请求排到 16-token 请求前，改变 batch 行顺序并引入正常 GEMM 数值差异。

### 6.3 最终回归

最终配置连续重复 3 次：

| Run | elapsed_s | generated_tokens |
|---|---:|---:|
| 1 | 82.8897 | 273 |
| 2 | 82.1392 | 273 |
| 3 | 82.1890 | 273 |

最终公开精度回归：mean_nll=2.4256333902，BF16 reference mean_nll=2.3084555301，delta_nll=0.1171778601，`passed=true`。相对原始 OJ elapsed_s=148.52，端到端时间下降约 44.2%，已达到 `<90s` 目标。
