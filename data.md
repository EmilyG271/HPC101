# Lab4 AMSS-NCKU 性能优化试验记录

> **用途**：汇总 ARM64 集群上的 AMSS-NCKU 性能优化试验，并区分“已验证结果”“明确失败方向”“无效/未完成试验”和“可继续验证的假设”。
>
> **记录时间范围**：2026-08-05 至 2026-08-06（均为历史记录）。  
> **结论口径**：只有在配置可比、运行完成且数值校验通过时，才可作为性能结论；编译器诊断、未完成运行、或环境状态不受控的结果不得用于加速比声明。

---

## 1. 一页结论

### 1.1 当前可确认的结论

| 结论 | 证据等级 | 说明 |
|---|---|---|
| 基线编译选项为通用 ARMv8-A，默认 NEON 路径表现优于本次测试的 SVE 路径。 | 已验证 | `-march=native` 和 `-march=armv8-a+sve` 均未带来收益。 |
| `-march=native` 不适用于本次 GCC 14 / Kunpeng 环境。 | 已验证为负向 | 它启用 LSE 等特性；记录显示 MPI 共享内存通信相关开销上升，整体变慢。 |
| SVE-only 不适用于本次代码与编译器组合。 | 已验证为负向 | 第 1 步 CPU 时间比参考值慢约 29%。 |
| OpenMP（MPI=30、OMP=2）不适用于当前并行划分。 | 已验证为负向 | 约慢 47%，fork/join 开销大于收益。 |
| `-funroll-loops` 不适用。 | 已验证为负向 | 约慢 27%，推测为指令缓存压力。 |
| LTO 可正确运行，但没有足够稳定的性能证据证明其带来收益。 | 数值正确；性能未确认 | 两次 t=5 记录为 1805s / 1843s，环境波动较大，按“无已确认收益”处理。 |
| 2MB Open MPI vader 段大小的首次收益不可复现。 | 未确认，不保留 | 第一次 A/B 快 14.46%，复测明显变慢，不能作为优化结果。 |
| t=40 标准 Python-driver 调度运行在 30 分钟限额内未完成。 | 已验证为不完整 | 完成至 t=35，数值正确，但不能当作 t=40 性能结果。 |

### 1.2 当前推荐策略

1. **保留基线编译策略**：`-O3 -fno-strict-aliasing -cpp`，不增加 SVE/LSE 架构选项，不使用不安全浮点选项。
2. **继续使用数值校验**：每次有效试验都必须检查 `max_rel`、RMS、约束量和（可行时）输出哈希。
3. **实验前清理/检查 `/dev/shm`**，但不要把它当成已经验证的性能优化；它是环境卫生措施。
4. **避免重复已知负向方向**：见第 5 节。
5. 若继续做源码优化，优先验证 `kodis` 的无效边界扫描消除；循环拆分本身尚无干净、可复现的端到端性能结果。

### 1.3 MPI=6 验证结果（需补充）

| 配置 | 每步 CPU 时间 | 相对 MPI=30 | 正确性 | 状态 |
|---|---|---|---|---|
| MPI=30（基线） | 51.4s | 1.00x | PASS | 基线 |
| MPI=6 | 45.4s | 0.86x（加速 14.3%） | PASS | **VALID（唯一有效方向）** |

**结论**：在当前代码和编译器下，MPI=6 是已验证的、可复现的正向优化。应将 MPI=6 作为后续所有优化的基础配置，而非 MPI=30。
---

## 2. 试验范围、环境与判定规则

### 2.1 工作负载与参考配置

| 项目 | 参考值 |
|---|---|
| 程序/案例 | AMSS-NCKU，`GW250118` |
| 并行配置 | MPI=30，OMP=1（除 OpenMP 对比试验） |
| 初始数据 | TwoPuncture cache 已复用 |
| 短测试 | CPU `t_final=5`，5 个 timestep |
| 正式参考 | CPU `t_final=40`，40 个 timestep |
| 基线优化 | `-O3`；后续受控 A/B 还固定 `-fno-strict-aliasing -cpp` |
| 正确性标准 | 轨迹匹配、RMS、约束检查；必要时比较数值输出 SHA256 |

### 2.2 记录中的平台信息

| 项目 | 记录值 |
|---|---|
| 架构 | ARM64 |
| CPU 标识 | Kunpeng 920B / HiSilicon（implementer `0x48`）；另一条环境记录标为 TaiShan-v120 |
| 编译器 | GCC/GFortran 14.2.0（Debian） |
| MPI | Open MPI 5.0.7 / PRRTE |
| CPU 特性 | 包含 NEON、SVE、LSE、dotprod、i8mm、bf16 等；实际启用特性由编译选项决定 |
| 常用 cpuset | `0-63`；调度任务中为 `64-123` |
| 容器 `/dev/shm` | 64 MB |
| 重要限制 | 曾发现 `cpu.max=400000 100000`，等效约 4 CPU 配额，可能引入显著时序波动 |
| 性能分析限制 | 容器 PMU 不可用（`perf stat/record` 失败，NMI watchdog locked） |

### 2.3 结果状态定义

| 状态 | 含义 | 是否可用于性能结论 |
|---|---|---|
| **VALID** | 运行完成、配置明确、数值校验通过 | 可以；仍须注意重复性 |
| **NEGATIVE** | 运行充分表明比参考慢或正确性失败 | 可以用于排除该方向 |
| **INVALID / INCOMPLETE** | 被杀、未完成、环境受污染、或未做完整校验 | 不可以 |
| **DIAGNOSTIC** | 仅编译器报告或代码审计 | 不可以，需后续运行验证 |
| **BLOCKED** | 调度器、资源或工具链阻塞 | 不代表算法/优化效果 |

---

## 3. 基线与已验证的有效运行

### 3.1 基线记录

| 试验 | 配置 | Wall 时间 | 单步信息 | 正确性 | 判定 |
|---|---|---:|---|---|---|
| `baseline_t40`（2026-08-05 02:48） | MPI=30，`t=40`，`-O3` | 14719s | 40 步；约 368s/步 wall；记录 CPU/步约 45.3s | `max_rel=0`，Ham=0.277，PASS | **VALID** 基线 |
| `mpi30_ltoreal_t5`（2026-08-05 18:16） | MPI=30，`t=5`，`-O3 -flto` | 1843s | 5 步；368.6s/步 | `max_rel=0`，PASS | **VALID** 数值结果；性能作为 t=5 参考 |
| `mpi30_lto_t5` | MPI=30，`t=5`，`-O3 -flto` | 1805s | CPU/步：44.40、40.70、44.99、45.62、45.73s | `max_rel=0`，RMS=0，PASS | **VALID** 数值结果；与 1843s 差异不足以确认 LTO 收益 |

> **注意**：原始记录把 t=40 `-O3` 与 t=5 `-O3 -flto` 都称作“baseline”上下文。二者的运行长度和部分构建设置不同，不能直接用来计算架构选项的严格加速比。

### 3.2 调度器中的标准 t=40 运行

| Job ID | 配置 | 调度结果 | 已完成进度 | 正确性 | 判定 |
|---:|---|---|---|---|---|
| 44599 | MPI=30，`t=40`，GCC/GFortran 14.2.0，`-O3`，经 `python3 AMSS_NCKU_Program.py` 启动 | 1800s 硬限制后 `Timeout` | 完成 timestep 35；BH 输出至 t=34，约束至 t=35 | 已完成部分 35/35 匹配，RMS=0，约束 PASS | **INCOMPLETE** |

- 已记录的 35 步 CPU 时间：平均 **43.2166s**，最小 **42.5749s**，最大 **44.6308s**。
- 线性估计完整 40 步约 **34.3 分钟**，超过该分区 30 分钟硬限制；这只是估算，不是测量结果。
- 部分输出哈希：
  - `bssn_BH.dat`：`1e50fbfc5096227df289558cce73c6006afb8a2b351fc440ea3be6ca05d18a8e`
  - `bssn_constraint.dat`：`bf376bdfd16c4f39f98893e9835d3c8e93da74351db061c6e326812902521870`

---

## 4. 已测试方向汇总

### 4.1 编译与体系结构选项

| 方向 | 关键配置/观察 | 结果 | 状态 | 可提取结论 |
|---|---|---|---|---|
| LTO | `-O3 -flto`，t=5 记录 1805s 与 1843s | 数值 PASS；差异很小且环境不稳定 | **VALID（数值）** | 可用，但没有已确认的性能收益 |
| 循环展开 | `-funroll-loops` | 约慢 27% | **NEGATIVE** | 不再尝试 |
| `-march=native` | 启用 SVE、LSE 等；运行 35+ 分钟后终止 | 计算时间不优，整体比约 30.7 分钟参考更慢超过 13% | **NEGATIVE** | 不再尝试；记录归因于 LSE 对 MPI 共享内存通信的不利影响 |
| SVE-only | `-march=armv8-a+sve`，不含 LSE | 第 1 步 CPU 65.36s vs 50.64s（约慢 29%） | **NEGATIVE** | 本代码/GCC 14/该平台下默认 NEON 更快 |
| PGO | `-fprofile-generate` | Phase 1 在第 5 步长时间停滞，58 分钟仅完成 4 步，未产出可用 `gcda` | **INCOMPLETE** | 计算只占 wall 时间约 12%，预期收益过低，停止 |
| `-O2`、`-fno-semantic-interposition`、`-fno-plt` | 曾排队或准备 | 没有记录到可比较的最终结果 | **NOT RUN / 无结论** | 不纳入结论 |

### 4.2 并行、绑定与 MPI 调优

| 方向 | 配置/现象 | 结果 | 状态 | 可提取结论 |
|---|---|---|---|---|
| MPI rank 数 | MPI=8 | RMS=5.67%，超过 0.1% 阈值 | **NEGATIVE（正确性）** | 不可用于本作业的正确性要求 |
| CPU/core/NUMA 绑定 | `--bind-to core --map-by core` 等 | 16 分钟仍为 0 步；多种绑定策略均退化 | **NEGATIVE** | 当前 Pod 拓扑下不要使用 |
| OpenMP | MPI=30、OMP=2、`-fopenmp` | 约慢 47%，未完整验证 | **NEGATIVE** | fork/join 开销大于收益；不再尝试同一并行划分 |
| vader 段大小 | `OMPI_MCA_btl_vader_segment_size=2097152` | 首次 1876s，比配对 2193s 快 14.46%；复测没有复现 | **INVALID（不可复现）** | 不保留为优化；只保留 `/dev/shm` 检查习惯 |
| `single_copy_threshold=0` | mpitune 试验 | `SIGBUS` 崩溃 | **NEGATIVE** | 不再尝试 |

### 4.3 源码级循环拆分

| 候选 | 目的 | 已发生情况 | 当前判定 |
|---|---|---|---|
| `fderivs` / `fdderivs` 循环拆分 | 将无分支内部区域与保留原逻辑的边界区域分开，改善向量化机会 | 首版 `fderivs` 误将内层循环嵌套进外层同名循环，导致 Fortran 编译失败；后续已修正编译 | **尚未完成干净的端到端性能验证** |
| `kodis` 循环拆分 | 同上，针对 Kreiss–Oliger 耗散 | 编译器诊断显示内部循环可用 16-byte NEON 向量化 | **DIAGNOSTIC**，不能声称加速 |
| `mpi30_ls_neon_t5` | 默认 NEON + 循环拆分 | 约 10 分钟未完成第 1 步；CPU/进程 51–82s 接近参考，但 wall 被 MPI 等待主导 | **INVALID（`/dev/shm` 污染）**；不能据此判断循环拆分性能 |

---

## 5. 已知失败方向：不要重复尝试

| 方向 | 失败原因或证据 | 重试条件 |
|---|---|---|
| MPI=8 | 浮点操作顺序变化，RMS=5.67% | 除非任务的正确性准则/并行算法发生变化 |
| OpenMP，MPI=30 + OMP=2 | 约慢 47%，fork/join 成本过高 | 除非改为不同的 MPI/线程混合并行设计 |
| 绑定（NUMA/socket/core） | 当前 Pod 拓扑下持续退化 | 除非资源分配与亲和性拓扑改变 |
| `-funroll-loops` | 约慢 27%，疑似 i-cache 压力 | 除非改变热点代码且重新基准测试 |
| `-march=native` | 总体变慢；LSE 相关 MPI 通信开销问题 | 除非更换编译器/MPI/硬件，并重新进行受控 A/B |
| `-march=armv8-a+sve` | 第 1 步约慢 29% | 除非更换 SVE 编译器后端或明显改变计算核 |
| `single_copy_threshold=0` | `SIGBUS` | 不建议重试 |
| PGO | 成本高、未产出可用 profile，理论 wall 收益有限 | 仅当先显著降低 MPI/分析占比时再评估 |
| BiSheng | 集群不可用 | 工具链可用后才可评估 |
| `perf` | 容器 PMU 不可访问 | 管理员开放 PMU 后才可使用 |

---

## 6. 环境异常与实验卫生

### 6.1 `/dev/shm` 容量与残留段

- 容器 `/dev/shm` 仅 **64MB**。
- 曾发现 **290 个**残留的 `sm_segment.*` 文件，合计显示已用约 **51MB（80%）**。
- 在该状态下，Open MPI 共享内存段可能无法建立并退回较慢通信路径；因此受影响的试验不能用于性能结论。
- 后续受控 A/B 要求：启动前确认没有 ABE、`prterun`、`mpiexec` 进程，`/dev/shm` 为 0% 使用且无残留 `sm_segment.*` 文件。
- 清理残留段是**实验卫生**，不是已经验证的加速手段。

### 6.2 可重复性风险

- `cpu.max=400000 100000` 对应有效约 4 CPU 配额，即使 cpuset 显示 64 个逻辑 CPU，也可能因节流导致时序强烈波动。
- t=5 运行之间存在明显波动：例如干净默认段运行的前两步可达 74.60s、62.48s；调优复测第 1 步为 76.6865s，而第一次调优为 47.093s。
- 因此，单次 t=5 成绩只能作为筛选信号；保留优化前必须至少在干净、受控的条件下复测。

---

## 7. 编译器诊断与后续可验证候选

> 下表仅代表向量化诊断或代码审计，不代表端到端性能收益。

### 7.1 诊断摘要

| 文件/位置 | 诊断观察 | 含义 |
|---|---|---|
| `diff_new.f90`：`fderivs`、`fdderivs` | 内层运行时 `if/elseif` 分支导致 “unsupported control flow” | 可尝试保持数学次序不变地拆分 interior/boundary 区域 |
| `kodiss.f90` | 原始内层条件阻碍向量化；拆分后内部循环可生成 16-byte NEON 向量 | 需要在干净环境中做正确性与性能 A/B |
| `bssn_rhs.f90`、`fadmquantites_bssn.f90`、`rungekutta4_rout.f90` | 已有部分 SVE/向量化报告 | 不构成架构选项有效性的证据 |
| C++ TwoPunctures | 大量 missed report，多为字符串操作 | 不应仅凭报告投入优化 |

### 7.2 低风险的下一候选

1. **先审计并删除 `kodis` 拆分后的纯无效边界扫描**：原始条件恰好等于 interior-box 判定，文档记录认为后续“boundary”扫描不会更新任何点。
2. 对该单一改动执行：干净 `/dev/shm` → 编译 → t=5 数值校验 → 至少一次复测 → 只有稳定后才考虑更长运行。
3. 不要将多项改变（LTO、循环拆分、MPI 参数、架构选项）合并后直接比较；一次只改变一个变量。

---

## 8. 受控 `/dev/shm` A/B 记录

固定条件：ARM64 TaiShan-v120、GCC/GFortran 14.2.0、Open MPI 5.0.7、MPI=30、OMP=1、无绑定、默认 NEON、`-O3 -fno-strict-aliasing`、不使用不安全浮点选项；A/B 的源码哈希和编译/链接命令相同，只改变 `btl_vader_segment_size`。

| 时间（UTC） | 试验 | 唯一变量 | Wall(s) | 正确性 | 判定 |
|---|---|---|---:|---|---|
| 2026-08-06 03:10 | `mpi30_shmclean_baseline_t5` | 默认 16MB vader 段 | 2193 | `max_rel=0`、RMS=0、约束 PASS；数值 SHA256 `8a67e05b...d8b27` | **VALID**，但前两步冷启动/异常慢 |
| 2026-08-06 03:56 | `mpi30_vader2m_clean_t5` | 2MB vader 段 | 1876 | 同一数值 SHA256，`max_rel=0`、RMS=0、约束 PASS | 单次快 14.46%，**待复现** |
| 2026-08-06 04:37 | `mpi30_shmclean_baseline2_t5` | 默认 16MB vader 段复测 | >1900 后停止 | 31 分钟无一步完成；无 OOM | **INVALID**，环境/通信异常 |
| 2026-08-06 05:10 | `mpi30_vader2m_clean2_t5` | 2MB vader 段复测 | 第 1 步后停止 | 第 1 步 76.6865s vs 首次 47.093s | **INVALID**，收益不可复现 |

**决定**：不保留 `btl_vader_segment_size=2097152` 为优化，也不允许基于此提交 t=40 或 Git 改动。

---

## 9. 调度器/资源问题记录

### 9.1 调度器约束

- DevPod 没有原生 `sinfo`、`scontrol`、`sbatch`、`squeue`；使用 `/usr/local/bin/hpc` 的 Slurm 风格封装。
- `lab4` 分区限制：最多 60 CPU、100Gi 内存、30 分钟 walltime、每用户一个活动作业。
- 原计划的 64 ranks 会被压到 60；这与应用内 MPI=30 的标准配置不是同一件事。
- 不允许在 login/DevPod 节点直接执行 `mpirun ./ABE`。

### 9.2 短测试作业

| Job ID | 资源/节点 | 结果 | 原因 | 判定 |
|---:|---|---|---|---|
| 44557 | `lab4`，60 CPU | 构建前失败 | `hpc` wrapper 拒绝空的 `ntasks-per-node` 自检 | **BLOCKED** |
| 44558 | `zjusct-920b-1`，cpuset `64-123`，配额 60 CPU | CMake 前失败 | 归档源码为 CRLF，`compile.sh` 报 `pipefail\r` | **BLOCKED** |
| 44567 | `zjusct-920b-1`，60 CPU，100Gi | 构建成功，未启动运行 | 镜像缺少 `/usr/bin/time`；ABE 与 TwoPunctureABE 均以 `-O3` 构建成功 | **BLOCKED** |
| 44573 | `zjusct-920b-1`，60 CPU，100Gi | MPI 启动失败 | PRRTE 对 `mpirun -n 60` 报 slots 不足 | **BLOCKED** |

这些作业没有产生可用数值演化或性能数据；因此没有保留源码改动，也没有进行 Git 提交。

---

## 10. 原始时间线索引

| 会话 | 日期 | 核心事件 | 最终状态 |
|---|---|---|---|
| Session 8 | 2026-08-05 20:06+ | PGO、`-march=native`、SVE-only、循环拆分准备 | PGO/架构选项被排除；循环拆分待验证 |
| Session 9 | 2026-08-06 05:20+ | 修复 `fderivs` 循环拆分的 Fortran 嵌套循环错误；发现 `/dev/shm` 问题 | 拆分可构建，但运行受环境污染 |
| Session 10 | 2026-08-06 | 清洁 `/dev/shm` 的 MPI vader 段 A/B；默认 NEON 向量诊断 | MPI 调优不可复现；诊断不构成性能结论 |
| Session 11 | 2026-08-06 | `hpc` 调度器短测试 | 被脚本、CRLF、计时工具和 MPI slots 问题阻塞 |
| Session 12 | 2026-08-06 | 标准 Python-driver 的 t=40 调度运行 | 数值正确至 t=35，但被 30 分钟上限截断 |

---

## 11. 仍可提取的原始关键数字

- t=40 `-O3` 基线：14719s / 40 steps，约 368s wall/step。
- t=5 `-O3 -flto` 参考：1843s / 5 steps，368.6s wall/step。
- PGO Phase 1 前三步 CPU：62.07s、61.59s、63.41s；相对对应基线约 +22.6%、+51.1%、+25.1%。
- `-march=native` 前三步 CPU：53.88s、52.58s、50.69s；对应记录基线为 50.64s、40.76s、50.68s。
- SVE-only 第 1 步 CPU：65.36s；对应参考 50.64s。
- `mpi30_ls_neon_t5` 被污染时各进程 CPU：51–82s；不能据此比较 wall 性能。

---

## 12. 运行与归档说明

- 集群账户、SSH 命令和旧工作目录属于历史环境信息，不是当前项目运行必需项；如需复现实验，应改为环境变量或本地私有配置，不应写入通用项目脚本。
- 没有完成、没有正确性结论、或环境未受控的结果，应保留为故障排查记录，而不是“优化成功”证据。
- 本文档不声明任何已提交的源码优化；截至记录结束，未因上述候选保留源码改动或创建 Git commit。



---

## 13. Implementation record: MPI=6 profiling and first OpenMP experiment (2026-08-07)

### 13.1 Implemented experiment infrastructure

- `AMSS_NCKU_Input.py` accepts experiment-only environment overrides: `AMSS_MPI_PROCESSES`, `AMSS_OMP_THREADS`, and `AMSS_CPU_FINAL_EVOLUTION_TIME`. Defaults remain the official baseline values.
- `AMSS_ENABLE_PROFILE=ON` emits rank-aggregated ABE timing for initialization, evolution, recursive steps, level steps, `RestrictProlong`, analysis, `Psi4`, constraints, MPI transfer, and MPI sync. `AMSS_PROFILE=1` also emits Python-driver timings for TwoPuncture, input assembly, ABE, and the Program Cost boundary.
- `scripts/run_cpu_experiment.sh` uses isolated build/output directories; records cpuset, CPU quota, NUMA, and `/dev/shm`; supports MPI/OMP/tfinal/profile switches; and invokes `--allow-partial` only for short t<40 correctness screening. Formal t=40 still uses strict checker behavior.
- `scripts/submit_cpu_experiment.sh` submits CPU work to `lab4` (default 30 CPU, 100Gi, 30 min).
- The initial Fortran OpenMP regions are now protected by `AMSS_ENABLE_OMP_KERNELS=ON`, so a normal OpenMP-runtime build cannot accidentally enable this unproven implementation.

### 13.2 MPI=6 x OMP=1 no-cache t=5 baseline (Job 52192)

| Item | Result |
|---|---|
| Node/resources | `zjusct-920b-3`; `lab4`; 30 CPU; cpuset `64-93`; 100Gi |
| Build | GCC/GFortran 14.2.0, `-O3`, OpenMP runtime and profiling enabled |
| Configuration | MPI=6, OMP=1, t=5, no TwoPuncture cache |
| TwoPuncture | 289.958586s |
| ABE evolution | 223.783449s |
| Program Cost | **513.777732s** |
| Runner wall | 523s (includes plotting/finalization after Program Cost) |
| Mean of five reported step CPU times | **43.55356s/step** (44.4823, 43.8304, 42.8695, 43.3604, 43.2252) |
| Correctness | Partial checker PASS: trajectory 5/5, RMS=0, constraints PASS |

The ABE/Python run completed normally. Scheduler Job 52192 was marked failed only because the original remote `check.sh` lacked its executable bit. The result was subsequently validated with the new short-run partial checker; the runner now invokes `bash ./check.sh` for formal validation.

Profile note: `recursive_step`, `level_step`, and communication values are cumulative function times and are nested; they must not be summed as a wall-time breakdown. Usable wall boundaries are ABE evolution=223.783449s and Program Cost=513.777732s. The five-call analysis total was 100.302476s; MPI transfer total was 16.413476s and MPI sync total was 8.937992s. This motivates analysis and communication work before compiler-flag micro-tuning.

### 13.3 MPI=6 x OMP=5 first Fortran OpenMP experiment (Job 52237)

The initial directive set covered 11 outer-k loops in `diff_new.f90` and one loop each in `kodiss.f90` and `lopsidediff.f90`, using `collapse(2)` and `schedule(static)`. The body writes independent grid points.

| Item | MPI=6 x OMP=1 | MPI=6 x OMP=5 | Assessment |
|---|---:|---:|---|
| Node | `zjusct-920b-3` | `zjusct-920b-1` | Different nodes: not a strict A/B pair |
| TwoPuncture | 289.958586s | 291.261585s | Essentially unchanged; it has no OpenMP work yet |
| ABE evolution | 223.783449s | 234.099877s | **4.61% slower** |
| Program Cost | 513.777732s | 525.366504s | **2.26% slower** |
| Mean step CPU time | 43.55356s | 50.84284s | **16.74% slower** |
| Correctness | PASS | trajectory 5/5, RMS=0, constraints PASS | Numerically correct |

Conclusion: the first OpenMP directive set is numerically correct but has no performance benefit at MPI=6 x OMP=5. It is retained only behind the explicit `AMSS_ENABLE_OMP_KERNELS=ON` experiment gate. A follow-up OpenMP attempt must be profile-driven and audit scheduling, AMR block granularity, false sharing, and parallel-region scope before adding more directives.

### 13.4 Direct consequence for CPU t=40 <=340s

- No-cache t=5 already spends about 290s in TwoPuncture alone. The current CPU end-to-end implementation therefore has no margin for a 340s target.
- The MPI=6 x OMP=1 t=5 ABE time (223.78s) also makes a no-cache t=40 submission infeasible under the 30-minute scheduler limit; no wasteful no-cache t=40 job was submitted.
- Next priorities: (1) parallelize/reduce allocation in TwoPuncture, (2) remove repeated analysis work, (3) retest MPI=4/5/6/7 after each source change, and (4) design a new OpenMP strategy from profile data.


### 13.5 Paused iteration: TwoPuncture profile build configuration failure (2026-08-07 08:52 UTC)

- Intended change: add an internal TwoPuncture timing profile so the next CPU experiment could identify the dominant solver routine before further OpenMP edits.
- Build command: isolated `build-twop-profile`, GCC/GFortran 14.2.0, `AMSS_ENABLE_OPENMP=ON`, `AMSS_ENABLE_PROFILE=ON`, `AMSS_ENABLE_TWOP_OMP=OFF`.
- Result: **CMake configure failed; no executable was produced and no numerical run was started.**
- Error: `target_compile_definitions` attempted to apply `AMSS_ENABLE_TWOP_PROFILE=1` to `TwoPunctureABE` before `add_executable(TwoPunctureABE ...)` had created that target.
- Required repair before resuming: move the TwoPuncture profile compile definition below the `add_executable(TwoPunctureABE ...)` declaration (or defer it until that target exists), then reconfigure an isolated build directory.
- Per the autonomous-optimization safety rule, work is paused immediately after this compile failure. No performance conclusion is drawn from this iteration.


### 13.6 Paused iteration: reusable Thomas workspace compile failure (2026-08-07 09:15 UTC)

- Intended change: eliminate repeated `new[]`/`delete[]` in `ThomasAlgorithm`, which is reached from the measured 193.788s cumulative `relax` hot path. The change replaced the four local work-array allocations with reusable `vector<double>` storage and did not alter arithmetic.
- Build command: isolated `build-twop-workspace`, GCC/GFortran 14.2.0, OpenMP enabled, profiling enabled, experimental ABE/TwoPuncture OpenMP regions disabled.
- Result: **compile failed; no executable was produced and no numerical run was started.**
- Error: `src/TwoPunctures.C:2382: error: 'vector' does not name a type`.
- Cause: this legacy source conditionally includes `<vector>` only under the legacy `newc` preprocessor branch, but that header was not visible in the effective TwoPuncture compilation unit.
- Required repair before resuming: include `<vector>` for the active compilation path (or replace the workspace with an allocation method already available in that path), then rebuild an isolated directory before testing.
- Per the autonomous safety rule, work is paused immediately after this compile failure. No performance conclusion is drawn from this iteration.


### 13.7 Resumed profile result and rejected Thomas workspace experiment (2026-08-07)

- The CMake target-order fix was applied and a profile-only MPI=6 x OMP=1 t=5 run (Job 52421) completed with strict short-run validation: trajectory 5/5, RMS=0, constraints PASS.
- TwoPuncture profile from Job 52421: solve=289.548075s; Newton=286.325455s (6 calls); BiCGStab=281.095183s (6 calls); relax=193.788075s (34,800 calls); Derivatives_AB3=84.717288s (204 calls); J_times_dv=78.226172s (180 calls). Categories are nested but identify relaxation as the dominant tunable internal component.
- The reusable `ThomasAlgorithm` vector-workspace experiment was rebuilt after adding the missing header and run as Job 52471 on the same `zjusct-920b-1` class of node, MPI=6 x OMP=1, t=5, no cache. It passed the partial checker (RMS=0, constraints PASS) but regressed: TwoPuncture=299.565362s, ABE=225.260281s, Program Cost=524.830006s.
- Decision: reject the Thomas workspace implementation and revert it. The likely allocation reduction was too small relative to scheduler/node variation and did not produce an end-to-end benefit.

### 13.8 Infrastructure iteration: collective t=5 submission argument failure (2026-08-07 09:38 UTC)

- Intended candidate: pack the real/imaginary waveform modes into one buffer and replace the two waveform `MPI_Allreduce` calls with a single collective; likewise replace the seven scalar mass/angular-momentum reductions with one 7-double collective.  The local floating-point operations and output cadence are unchanged.
- Job 52502 was submitted as MPI=6 x OMP=1, t=5, no cache, using the prebuilt isolated `build-analysis-collective` directory.
- Result: **runner argument failure before compilation or numerical execution** (`unknown argument: --openmp`). The remote submission command acquired a trailing carriage return on its final `--openmp` argument; no executable was run and no performance/correctness conclusion is drawn.
- Corrective action: resubmit the identical preconfigured build without the redundant final `--openmp` runner option, using CRLF-safe command transport.

### 13.9 Rejected iteration: packed analysis collectives (2026-08-07 09:46–09:55 UTC)

- Change tested in isolated `build-analysis-collective`: in the primary CPU `surf_Wave` path, pack the real/imaginary mode arrays and replace two `MPI_Allreduce` calls with one `2*NN` collective; in the primary CPU `surf_MassPAng` path, replace seven scalar reductions with one 7-double collective.  No local arithmetic, physical/grid parameter, output cadence, or cache behavior was changed.
- Job 52508: MPI=6 x OMP=1, t=5, GCC 14.2.0, `-O3 -fno-strict-aliasing -cpp`, profiling enabled, no TwoPuncture cache, node `zjusct-920b-1`, cpuset `64-93`.
- Correctness: **PASS** — trajectory 5/5, RMS=0, constraints PASS.
- Timing: TwoPuncture=290.577872s; ABE=224.935336s; Program Cost=515.517162s.  Against the same-node profile baseline Job 52421 (TwoPuncture=290.404s, ABE=225.576s, Program Cost=515.984s), the end-to-end difference is only -0.467s (-0.09%), well below the 5% retention threshold and within normal run variation.
- Decision: **reject and revert** the collective packing change in `src/surface_integral.C`; it is numerically sound but not a material end-to-end optimization at MPI=6.  The original source was restored locally and copied back to the remote workspace.  The residual ABE analysis profile still motivates a more structural analysis-path optimization only after addressing the much larger TwoPuncture bottleneck.

### 13.10 Effective iteration: OpenMP red/black line relaxation in TwoPuncture (2026-08-07 09:57–10:04 UTC)

- Evidence-driven change: the TwoPuncture internal profile identified `relax` as 193.788s cumulative within BiCGStab.  Its four red/black line-solve phases are separated by their required color barriers, while the lines *inside* each phase operate on disjoint `dv` rows/columns and read the opposite color.  Added guarded `!$omp`-equivalent C++ `#pragma omp parallel for schedule(static)` worksharing to each of those eight phase loops in `TwoPunctures::relax`; the existing barriers at the end of each worksharing loop preserve the original phase ordering.  This is enabled only by `AMSS_ENABLE_TWOP_OMP=ON` and does not change arithmetic, physics/grid parameters, output cadence, or cache behavior.
- Job 52533: isolated `build-twop-relax-omp`, GCC/GFortran 14.2.0, `-O3 -fno-strict-aliasing -cpp`, OpenMP/profile enabled, MPI=6 x OMP=5, t=5, no cache; node `zjusct-920b-1`, cpuset `64-93`.
- Correctness: **PASS** — trajectory 5/5, RMS=0, constraints PASS.
- Timing: TwoPuncture=156.143124s; ABE=229.196005s; Program Cost=385.343702s.  Versus same-node Job 52421 (515.984s Program Cost), this is -130.640s (-25.32%); versus the best validated baseline Job 52192 (513.778s), it is -128.434s (-24.999%).  TwoPuncture itself improves from 290.404s to 156.143s (-46.23%).
- Decision: **retain** this source optimization as the new best validated t=5 candidate.  Its change is isolated to the red/black line-relaxation loops.  Required next step: run the same binary and OMP=5 setting at MPI=4,5,7 so the full MPI=4/5/6/7 scan is complete before selecting the formal configuration.

### 13.11 Correctness-stop iteration: MPI=4 rank retest of retained TwoPuncture OpenMP candidate (2026-08-07 10:08–10:15 UTC)

- Job 52554 used the retained `build-twop-relax-omp` binary, MPI=4 x OMP=5, t=5, no cache, GCC/GFortran 14.2.0 with `-O3 -fno-strict-aliasing -cpp`; node `zjusct-920b-1`, cpuset `64-93`.
- Timing before validation: TwoPuncture=160.026107s; ABE=246.978984s; Program Cost=407.009566s.  This is slower than the MPI=6 x OMP=5 candidate (385.343702s).
- Correctness: **FAIL** — trajectory coverage 5/5 but RMS=0.111231861 (11.123186%), above the 0.001 threshold; constraints remained PASS.  The runner exited 1 and no formal result is claimed.
- Decision: reject MPI=4 for this source candidate.  Per the autonomous-optimization safety rule, stop the rank scan and pause immediately on this correctness failure; do not submit the queued MPI=5/MPI=7 retests or any additional source/performance candidates until the user explicitly resumes.  The validated MPI=6 x OMP=5 result from Job 52533 remains retained; the MPI=4 failure does not invalidate that separately verified configuration.

### 13.12 MPI=5 retest of retained TwoPuncture OpenMP candidate (2026-08-07 10:16–10:23 UTC)

- Job 52570: retained `build-twop-relax-omp`, MPI=5 x OMP=5, t=5, no cache, GCC/GFortran 14.2.0 with `-O3 -fno-strict-aliasing -cpp`; node `zjusct-920b-1`, cpuset `64-93`.
- Correctness: **PASS** — trajectory 5/5, RMS=0, constraints PASS.
- Timing: TwoPuncture=159.134285s; ABE=243.839048s; Program Cost=402.977983s.
- Decision: retain MPI=5 only as a correct comparison point, not as the selected configuration.  It is 17.634281s (4.58%) slower than MPI=6 x OMP=5 Job 52533 (385.343702s) on the same node class.  Continue with the required MPI=7 retest; MPI=4 remains rejected on correctness.

### 13.13 Correctness-stop iteration: MPI=7 retest and completed rank-scan evidence (2026-08-07 10:24–10:31 UTC)

- Job 52597: retained `build-twop-relax-omp`, MPI=7 x OMP=5, t=5, no cache, GCC/GFortran 14.2.0 with `-O3 -fno-strict-aliasing -cpp`; node `zjusct-920b-1`, cpuset `64-93`.
- Timing before validation: TwoPuncture=158.524565s; ABE=252.826345s; Program Cost=411.355891s, slower than MPI=6 x OMP=5 (385.343702s).
- Correctness: **FAIL** — trajectory coverage 5/5 but RMS=0.111312551 (11.131255%), above the 0.001 threshold; constraints PASS.  The runner exited 1, so no formal result is claimed.
- Completed source-change rank scan summary at OMP=5: MPI=4 FAIL (RMS=0.111231861, 407.009566s); MPI=5 PASS (RMS=0, 402.977983s); MPI=6 PASS (RMS=0, 385.343702s); MPI=7 FAIL (RMS=0.111312551, 411.355891s).  Therefore **MPI=6 x OMP=5 is both the fastest and the only selected correct configuration in the requested 4/5/6/7 neighborhood**; MPI=5 is correct but 4.58% slower.
- Decision: reject MPI=7.  Per the autonomous-optimization safety rule, pause immediately after this correctness failure.  The retained code and validated MPI=6 x OMP=5 candidate are unchanged; no further source modification or t=40 submission has been made.

## 14. ABE analysis-path profiling and OpenMP interpolation

### 14.1 Detailed profile baseline (Job 52650, 2026-08-07 10:39–10:46 UTC)

- A profiling-only build was created from the retained MPI=6 x OMP=5 TwoPuncture candidate.  It adds nested, rank-aggregated timers around CPU waveform/ADM surface integrals, their `Patch::Interp_Points` calls, and their remaining local integration/collective work.  Profiling is compiled only with `AMSS_ENABLE_PROFILE=ON`.
- Job 52650, MPI=6 x OMP=5, t=5, no cache, passed strict short-run validation (trajectory 5/5, RMS=0, constraints PASS).  Program Cost=386.272881s; variation from Job 52533 is attributable to normal run variation/profiling.
- Measured ABE bottleneck: Analysis=99.892874 rank-seconds over 5 calls.  `surf_MassPAng`=88.182339s over 40 calls, of which `Interp_Points`=88.151917s; waveform interpolation is 10.386191s and waveform integration only 1.289416s.  ADM local integration is only 0.011266s.  Thus the next optimization targets CPU interpolation worksharing, not MPI reduction packing or ADM arithmetic.

### 14.2 Effective iteration: OpenMP point-level CPU interpolation (Job 52686, 2026-08-07 10:50–10:56 UTC)

- Change: added isolated `AMSS_ENABLE_INTERP_OMP` build/runner option and used actual `#pragma omp parallel for schedule(static)` worksharing over independent surface points in the CPU `Patch::Interp_Points` path.  The shared interpolation scratch bounds and variable-list cursor were made per-iteration (`llb_local`, `uub_local`, `varl_local`); each iteration writes only its own `shellf[j,*]` and `weight[j]`.  MPI collectives remain outside the parallel region.  No physics, grid, output cadence, or cache behavior was changed.
- Job 52686: isolated `build-interp-omp`, GCC/GFortran 14.2.0, `-O3 -fno-strict-aliasing -cpp`, MPI=6 x OMP=5, t=5, profile/TwoPuncture OMP/interpolation OMP enabled, no cache; node `zjusct-920b-1`, cpuset `64-93`.
- Correctness: **PASS** — trajectory 5/5, RMS=0, constraints PASS.
- Timing: TwoPuncture=160.043755s; ABE=200.454368s; Program Cost=360.502770s.  Compared with the detailed-profile baseline Job 52650, ABE improves by 25.309s (11.21%) and Program Cost by 25.770s (6.67%).  Compared with retained Job 52533, Program Cost improves by 24.841s (6.45%).
- Profile confirmation: Analysis drops from 99.892874s to 74.814501s; ADM interpolation drops from 88.151917s to 65.845462s (25.30%); waveform interpolation drops from 10.386191s to 7.638514s.  The limited gain indicates point ownership provides fewer independent points per MPI rank than there are threads, so a task-decomposition of independent (point, variable) interpolation calls is the next evidence-based candidate.
- Decision: retain point-level interpolation OpenMP as the current best validated t=5 configuration while testing the more fine-grained task decomposition in a separate build.

### 14.3 Compile-only failure and repair: task-level interpolation candidate (2026-08-07 11:00 UTC)

- Intended change: replace point-level interpolation worksharing with finer independent `(surface point, field)` tasks after a serial point-to-owner-block map, addressing the profile-confirmed lack of points per MPI rank.
- Job 52715 configured the isolated `build-interp-task-omp` build successfully but **failed during ABE compilation; no executable or numerical run was produced.**
- Error: `src/MPatch.C:336: error: unterminated #else`.
- Cause: the new `AMSS_ENABLE_INTERP_OMP` conditional closed its own `#ifdef` but omitted the following `#endif` that closes the enclosing legacy `USE_GPU` CPU/GPU conditional.
- Repair: restored the missing outer `#endif`, rechecked the preprocessor structure locally, and copied the repaired source to the remote workspace.  No performance or correctness conclusion is drawn from Job 52715; the retained point-level interpolation implementation remains the active fallback.

### 14.4 Rejected iteration: task-level interpolation worksharing (Job 52747, 2026-08-07 11:08–11:14 UTC)

- After repairing the preprocessor-only compile issue recorded in 14.3, Job 52747 tested serial owner-block mapping plus OpenMP worksharing over independent `(point, field)` interpolation tasks, MPI=6 x OMP=5, t=5, no cache.
- Correctness: **PASS** — trajectory 5/5, RMS=0, constraints PASS.
- Timing: TwoPuncture=158.024360s; ABE=201.312755s; Program Cost=359.338259s.  This is statistically indistinguishable from the retained point-level implementation (Job 52686: 360.502770s); its ABE analysis time is 75.098932s vs. 74.814501s, and ADM interpolation is 66.107218s vs. 65.845462s.
- Decision: **reject and revert** the task decomposition.  The apparent 1.165s end-to-end difference is below the 5% retention threshold and the detailed profile is slightly worse.  Restored the simpler, separately validated point-level OpenMP interpolation implementation.

### 14.5 Isolated compiler candidate prepared: GCC `-Ofast` (2026-08-07 11:16 UTC)

- Added forwarding of an explicit `AMSS_OPT` environment value through `compile.sh` into the isolated CMake build, while preserving the default `-O3` build when it is unset.  This makes compiler-flag experiments reproducible without contaminating other build directories.
- Next test is GCC 14.2.0 `-Ofast` with MPI=6 x OMP=5, TwoPuncture OpenMP and point-level interpolation OpenMP retained.  It is strictly an isolated candidate: it must pass the full t=5 trajectory RMS and constraint checks before any result is retained.

### 14.6 Effective compiler iteration: GCC 14.2.0 `-Ofast` (Job 52771, 2026-08-07 11:16–11:21 UTC)

- Job 52771 used a fully isolated `build-ofast` directory with GCC/GFortran 14.2.0, `AMSS_OPT=-Ofast`, retained `-fno-strict-aliasing -cpp`, MPI=6 x OMP=5, TwoPuncture relaxation OpenMP and point-level interpolation OpenMP, t=5, no cache.  The build log and environment metadata both explicitly record `AMSS_OPT=-Ofast`.
- Correctness: **PASS** — trajectory 5/5, RMS=0, constraints PASS.  This confirms the fast-math compiler candidate satisfies the short-run numerical gates for this configuration.
- Timing: TwoPuncture=110.816497s; ABE=160.404823s; Program Cost=271.226030s.  Relative to the retained `-O3` point-interpolation candidate Job 52686 (360.502770s), this is -89.277s (-24.77%) end-to-end; TwoPuncture improves 30.75% and ABE 19.98%.
- Decision: **retain as the new best validated t=5 candidate.**  The next priority is a no-profile, isolated formal CPU t=40 run with the same MPI=6 x OMP=5 and `-Ofast` settings, followed by the strict formal checker.  MPI=4 and MPI=7 were already invalid in this requested rank neighborhood under the same physics/grid and OpenMP decomposition; MPI=5 was correct but slower under the prior equivalent build.  The formal MPI=6 candidate is therefore submitted first to maximize the remaining authorized window.

### 14.7 Formal t=40 result: scheduler-limit success but target/checker failure (Job 52797, 2026-08-07 11:24–11:48 UTC)

- Formal candidate: isolated no-profile `build-ofast-t40-formal`; GCC/GFortran 14.2.0 `-Ofast`, `-fno-strict-aliasing -cpp`, MPI=6 x OMP=5, real TwoPuncture relaxation OpenMP and point-level interpolation OpenMP, no cache, physical/grid/output parameters unchanged.
- The full CPU evolution reached physical time t=40 and completed normally within the 30-minute scheduler allocation.  `This Program Cost = 1411.280833s` and runner wall time was 1420s.  This is a material improvement over the prior >30-minute baseline, but it is **not** the required ≤340s target (1071.281s / 315.1% over).
- Formal checker result: **FAIL**.  Constraints passed (40 time groups, 9 levels; all maxima <=2), but the current strict checker compares against a 100-time-group golden trajectory and rejected the t=40 output as incomplete (`matched 40/100; no unused target time within 1e-08 of reference time 40`).  Thus this is not a valid formal PASS under the current checker implementation, even though the requested t=40 evolution itself completed.
- Decision: do not claim a validated formal result.  Per the autonomous safety rule, pause immediately after this correctness/checker failure.  No subsequent MPI, compiler, or source candidate is submitted.  The best short-run validated configuration remains Job 52771: MPI=6 x OMP=5, `-Ofast`, TwoPuncture/interpolation OpenMP, t=5 Program Cost=271.226030s, RMS=0, constraints PASS.
- Required resolution before further formal acceptance: make the checker’s expected trajectory coverage explicitly match the requested formal t=40 scope (while still requiring all t=0..40 reference samples, RMS, and constraints), or provide the intended t=40 golden reference.  This is a validation-scope issue; it must not be bypassed with an unconstrained partial checker.
