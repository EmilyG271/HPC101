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
| 已完成 | GPTQ + SDPA/Flash dispatch | 包含在上行结果 | CUDA 使用 scaled_dot_product_attention |
| 已完成 | GPTQ + ring KV cache | 包含在上行结果 | sliding-attention 仅保留窗口 |
| 已完成 | performance_public + batch 1 | 153.8663635399（5 请求，113 token） | 公开性能集，成功完成 |
| 已完成 | performance_public + batch 2 | 133.7539581680（5 请求，113 token） | 当前最佳，较 batch 1 快 13.1% |
| 失败记录 | performance_public + batch 4 | OOM | 10 GiB MIG 在 MLP 输出拼接阶段申请约 176 MiB 时失败 |

## 4. 运行命令

    hpc submit -p lab5 -g 1 -t 30m --interactive bash
    uv pip install -e . --no-deps
    export MODEL_DIR=/checkpoints/gemma-4-12b
    export QUANT_DIR=$HOME/gemma-4-12b-gptq-w4a16
    python3 scripts/quantize.py --config config.yaml --model "$MODEL_DIR" --output "$QUANT_DIR" --device cuda --verbose
    python3 scripts/evaluate_quality.py --model "$QUANT_DIR" --dataset datasets/quality_public.jsonl --output results/quality-final.json --linear-backend int4_reference --no-progress
    python3 scripts/run_generation_queue.py --model "$QUANT_DIR" --input datasets/performance_public.jsonl --output results/generation-final.jsonl --summary-output results/generation-final-summary.json --config config.yaml --linear-backend int4_reference --batch-size 1 --max-sequence-length 2048 --no-progress

## 5. 结果

集群验证结果（H800 MIG 1g.10gb，作业 186041）：GPTQ 成功量化 328 个 Linear 模块，峰值主机内存约 5.66 GiB；公开质量集 INT4 mean_nll=2.4259347128，BF16 mean_nll=2.3084555301，因此 delta_nll=0.1174791827，满足硬门槛 delta_nll < 0.16。

小规模性能集（performance_small.jsonl）使用 batch 1 完成 4 个请求、生成 60 tokens，elapsed_s=84.1439217550，generated_tokens_per_s=0.7130639831。公开性能集使用 batch 1 完成 5 个请求、生成 113 tokens，elapsed_s=153.8663635399；batch 2 的 elapsed_s=133.7539581680，是当前最佳结果。batch 4 在 10 GiB MIG 上仍然 OOM，因此最终默认 batch 保守设置为 1；OJ 性能命令建议显式使用 batch 2。
