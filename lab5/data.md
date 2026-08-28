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

- 算法：GPTQ，W4A16，group_size=128，对称量化，scale 为 FP16。
- 校准集：datasets/calibration-256.jsonl。
- 默认校准：256 条样本、micro batch 1、最多 4096 tokens。
- GPTQ：Hessian X^T X / N，对角阻尼 damp_percent=0.01，误差按逆 Hessian 顺序传播，block_size=128。
- 输出格式：packed uint8 little-nibble，manifest 与 safetensors checkpoint。

## 3. 性能测试约定

固定以下参数再比较：同一量化 checkpoint、同一 GPU、performance_public.jsonl、max-sequence-length=2048、max-new-tokens=32、seed=42、关闭进度条；以脚本摘要中的 elapsed_s 为 OJ 指标。

| 版本 | 参数/优化 | elapsed_s | 备注 |
|---|---|---:|---|
| 待测 | BF16/原始基线 | — | 需在集群补测 |
| 待测 | INT4 reference + static batch | — | 需在集群补测 |
| 待测 | INT4 + SDPA/Flash dispatch | — | CUDA 使用 scaled_dot_product_attention |
| 待测 | INT4 + ring KV cache | — | sliding-attention 仅保留窗口 |
| 待测 | 最终配置 | — | 需在集群补测 |

## 4. 运行命令

    hpc submit -p lab5 -g 1 -t 30m --interactive bash
    uv pip install -e . --no-deps
    export MODEL_DIR=/checkpoints/gemma-4-12b
    export QUANT_DIR=$HOME/gemma-4-12b-gptq-w4a16
    python3 scripts/quantize.py --config config.yaml --model "$MODEL_DIR" --output "$QUANT_DIR" --device cuda --verbose
    python3 scripts/evaluate_quality.py --model "$QUANT_DIR" --input datasets/quality_public.jsonl --linear-backend int4_reference
    python3 scripts/run_generation_queue.py --model "$QUANT_DIR" --input datasets/performance_public.jsonl --output results/generation-final.jsonl --summary-output results/generation-final-summary.json --config config.yaml --linear-backend int4_reference --batch-size 4 --max-sequence-length 2048 --no-progress

## 5. 结果

本地 Windows 工作区没有 CUDA/PyTorch 运行时，且本次集群作业的 home 挂载未能读取上传到登录节点的工作区文件，因此尚未伪造 delta_nll 或 elapsed_s。代码已完成静态编译检查；实际数字应由上述命令生成后填入第 3 节。
