# OpenMM MD provider

`mmpbsa` 0.7 将 OpenMM 作为可选的、仅实现 `md` capability 的 provider。GROMACS
仍是默认 MD 引擎；新的 OpenMM 计算必须同时选择 `@2` engine-neutral protocol 和
`md_engine_profile: openmm_8_5@2`。`@1` protocol 保持冻结，只能由 GROMACS 执行。
旧的 `openmm_8_5@1` 使用 GROMACS 输入表示；其既有结果可只读审计，不能与新的
Amber 输入 profile 交叉复用或恢复。

```bash
python -m pip install -e ".[openmm,test]"
```

OpenMM 依赖在 worker 子进程内懒加载；`openmm` extra 同时安装 trajectory v3 用的
MDTraj 和 ParmEd。安装基础包而不安装 OpenMM 时，provider discovery 和全部
GROMACS `@1` 路径仍可用；只有真正选择 OpenMM MD 时才要求
OpenMM `>=8.5.1,<8.6`。

## Configuration

Workflow 只声明科学协议，不声明 GPU 或 executable：

```yaml
protocol:
  preset: production_3x15ns@2
  md_engine_profile: openmm_8_5@2
  models: [gb, pb]
  completion_policy: all
```

CUDA runtime 示例：

```yaml
schema_version: mmpbsa.runtime.v1
providers:
  admission: backends
  md: openmm
  trajectory: backends
  trajectory_qc: backends
  scoring: backends
  report.assets: backends
  report: report
scheduler:
  gpu_ids: ["0", "1"]
  max_workers: 2
  threads_per_job: 4
  cpu_slots: 8
backends:
  openmm:
    common:
      probe_cwd: .
    md:
      python_executable: /path/to/python
      platform:
        name: CUDA
        precision: mixed
        deterministic_forces: true
  backends:
    trajectory:
      cpptraj:
        executable: cpptraj
        version_arguments: [-V]
        major_policy: {accepted: [6]}
    # admission, trajectory_qc, scoring and report.assets remain unchanged
  report: {}
environment: {}
```

`DeviceIndex` 只能来自 scheduler lease；runtime 不能另外填写 `device_index`。请求的
CUDA platform、precision 或 leased device 不可用时直接失败，不回退到 CPU、Reference
或另一张卡。CPU smoke 使用 `platform: {name: CPU, deterministic_forces: true}` 且
`scheduler.gpu_ids: []`；Reference 只用于极小的诊断体系。

CPU smoke 的 `scheduler.cpu_slots` 是本框架的并发准入额度，不是操作系统线程上限。
本机 OpenMM 8.5.1 的 CPU+PME 诊断显示，Context `Threads=4` 之外仍会创建 PME/FFT
线程池。可在 `backends.openmm.md.environment` 显式设置
`OPENMM_CPU_THREADS: "4"` 与 `OPENBLAS_NUM_THREADS: "1"`；本机极小 PME 诊断的
进程线程数由 325 降至 138，但独立 FFT 池仍存在，因此不能把此配置宣传为严格的
4 核隔离。真实体系的 CUDA smoke 应等待可用 GPU，不抢占已忙的设备。
未安装 `netCDF4` 时，MDTraj 会提示退回 scipy NetCDF 实现，可能拖慢较长轨迹的
读写；当前 20 ps smoke 的功能门禁不依赖该可选性能包。

## Scientific and artifact contracts

两个 MD provider 从同一份 `mmpbsa.md-phase-plan.v2` 读取 ensemble、步数、温压、
PME/cutoff、constraint 和输出频率。OpenMM profile 固定使用 LangevinMiddle
thermostat、MonteCarloBarostat、2 fs timestep、PME 1.0 nm、Ewald tolerance `1e-5`、
H-bond constraints、rigid water、dispersion correction 和每 100 steps 的 COM removal。
公共 EM contract 只规定无约束体系、50,000 次上限及最终最大力不超过
`1000 kJ mol⁻¹ nm⁻¹`；GROMACS profile 使用 steepest descent 和 `-DFLEXIBLE` 水，
OpenMM profile 使用 L-BFGS（optimizer tolerance `10`）。约束从 NVT 开始启用。
这样最大力门禁检查的是无约束 EM 体系，而不是刚性水约束方向上的内部 raw force。
它与 GROMACS 的 V-rescale/C-rescale 实现的是相同目标 ensemble，不宣称逐步轨迹或
逐项能量完全相同。

MD stage 始终只收到 role-neutral `PreparedMdView`，看不到 receptor、ligand、binding
或实体 metadata。GROMACS 从已封存的 `md_topology` (`topol.top`) 与
`md_coordinates` (`system.gro`) 建体系；OpenMM Amber profile 从同一 PreparedCase
已封存的 `solvated_topology` (`solvated.prmtop`) 与 `solvated_coordinates`
(`solvated.inpcrd`) 建体系。输入表示、所选 artifact 摘要与 adapter 版本属于 MD
身份，不按文件后缀猜测，也不把旧 checkpoint 带入新表示。Amber 坐标、盒子、
原子顺序和键图必须与 GROMACS 表示通过预检；能解析文件只证明流程兼容，尚不
证明两种引擎的逐项能量、力或采样统计等价。OpenMM 对 AMBER 输入的官方示例见
[OpenMM user guide](https://docs.openmm.org/latest/userguide/application/02_running_sims.html)。
完成后两个 provider 都发布同名角色：

- `full_trajectory`：完整显式体系 XTC，production `t=0` 后按 protocol interval；
- `coordinates`：最终全体系坐标；
- `state`：引擎状态；
- `checkpoint`：同引擎恢复检查点；
- `energy_log`：可审计能量/温压时间序列。

后续 trajectory v3 才结合 `source_md_atom_indices`、MD 的 `md_topology`（ParmEd
化学核验）与 `md_coordinates`（MDTraj 读取 `.gro` 原子排列）、dry complex topology
和 primary binding：它流式投影完整 dry production，保留 frame 1 作为 post-NPT production-start
源帧；先按化学键图重建完整分子，再由 cpptraj 做 ligand-anchor/receptor-fixed autoimage。
处理后的首帧单独封存为 `prepared_reference_trajectory`，analysis window/stride 帧
成为 scoring trajectory；二者不做 RMS fit，避免 NetCDF 丢失旋转后的盒方向。
receptor-interface fit 只作用于展示 XTC。逐帧键完整性检查必须通过才能评分。
这一规则也适用于 GROMACS `@2`，不改变 MD 科学参数或引擎完成契约。ParmEd
重算 `mmpbsa.atom_order.v1` 并逐原子验证 MD↔scoring residue crosswalk；源 XTC 还须
严格满足 `t=0`、protocol interval、simulation-step grid 和 stored-frame count。该阶段
不依赖 TPR、GROMACS index group 或文件名猜测引擎。

`@2` QC 只在预设分析窗口的帧上计算四个 gate，但以这条完整 dry production
经完整分子重建和相同成像处理的 `t=0` 作为结构与 native-contact 参考。
QC 自行进行 RMS 对齐，接触距离使用完整复合物的 `noimage` 非周期距离。
它能判断分析窗口相对初始 pose 的
偏移，不能捕捉窗口前出现又恢复的短暂漂移；原始完整轨迹仍作为审计产物保存。
QC payload 的 `reference_definition` 明确记录 `prepared_reference_trajectory`
及其 `selected_trajectory` frame 1 来源、`post_npt_production_start` 语义；
旧 `@1` 的参考定义不回溯修改。

`preparation.v2` 当前对多链 peptide 的 QC backbone 自动识别只接受唯一一条具有
完整 `N/CA/C/O` 主链的 chain。同一 chain 混有 glycan 或采用非标准骨架时会明确
拒绝，而不会猜测 peptide 边界；将来要支持这些输入，需在 preparation/binding
metadata 中显式给出 ligand 实体及用于 QC 的 backbone 选择。

## Resume boundary

EM/NVT/NPT 与每个 production chunk 都有独立封存 receipt。恢复时只复用 request、
size 和 SHA-256 全部匹配的完整单元；第一个不完整单元及其后内容重新计算。OpenMM
binary checkpoint 只在 exact OpenMM version、System、platform、precision 和
deterministic properties 一致时加载。阶段转换使用 XML State；不会拿 NVT binary
checkpoint 跨到含 barostat 的 NPT System。最终合并会拒绝缺帧、重复 step、错误时间轴
或被修改的 chunk。

PreparedCase 文件在每个 attempt 执行前复制到该 attempt 的固定输入快照并再次核对
size/SHA-256，恢复时也核对同一快照。GROMACS topology 的递归 `#include` 必须
完整封存于 PreparedCase 根目录并进入科学哈希；缺失、未声明或越界的 include
在执行前拒绝。OpenMM worker 的实际版本必须与预留 attempt 时探测的版本一致。

## Development status

OpenMM 为显式启用的实验 provider，GROMACS 是默认引擎。当前源码的真实 CUDA、
多 replica、参数表示的能量/力差分和体系外推尚需独立科学验收。单条短流程只能
检查输入和脚本兼容，不能替代生产统计或证明分数的科学有效性。
OpenMM 8.5.1 不能解析 GROMACS topology 中 `[ pairs ]` function 2；含该表示的
体系应使用经过校验的 Amber 输入表示。

## Acceptance gates for production support

主线可包含默认关闭的实验 provider，这不等于 Amber/CUDA 的科学生产验收。
从实验支持提升为生产支持前至少要求：基础安装不导入 OpenMM、CPU worker/pipeline smoke、CUDA 4090
无 fallback smoke、精确首帧/frame count/time axis、同平台 interruption/resume、
trajectory v3 到 GB/PB 的最小端到端，以及 small-molecule、modified peptide 和
protein–peptide 三类 PreparedCase。科学验收比较 ensemble/QC 和最终工作流
完整性；不把随机轨迹或最终 MM/PB(GB)SA 数值逐点相等当作正确性标准。

CUDA smoke 是可选的本机门禁，不放入无 GPU 的 required CI：

```bash
MMPBSA_OPENMM_CUDA_SMOKE=1 MMPBSA_OPENMM_CUDA_DEVICE=0 \
  python -m pytest -q tests/backends/test_openmm_backend.py -k cuda_worker
```
