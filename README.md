# mmpbsa 0.7

`mmpbsa` 是以显式实体、可复现协议和独立 replica 为核心的 MM/PB(GB)SA 工作流
框架。GROMACS 是默认 MD 引擎；OpenMM 是需要显式启用的[实验 provider](docs/openmm.md)，
当前源码的真实 CUDA 和多 replica 科学验收仍待完成。

MD 只消费完整物理体系，不识别 receptor/ligand；封存的 `PreparedCase v2` 明确
声明实体、原子映射和唯一 primary binding，供轨迹、QC、GB/PB 和报告使用。小分子
ligand 的主 RMSD 使用重原子，多肽使用 backbone，修饰仍属于同一个实体。结构选择、
辅因子和参数化等化学决策由项目侧完成，框架只验证并封存最终产物。

每个 replica 依次执行 MD、轨迹处理和四项 QC gate，合格后运行所选评分模型。
GB 和 PB 可单选或并行执行，默认同时选择并要求全部完成；它们使用同一个 frame
manifest。QC BLOCK 是科学排除，不伪装成计算崩溃。独立 finalize 后生成分析和报告。

## 安装

```bash
python -m pip install -e '.[parmed,test]'
# 可选：OpenMM 及 engine-neutral 轨迹处理
python -m pip install -e '.[openmm,test]'
```

环境与原生依赖见 [environment.md](docs/environment.md)。GROMACS、AmberTools 等
外部工具需由 runtime 配置显式提供。

## 基本流程

验证并封存项目侧准备好的产物：

```bash
mmpbsa prepared-case validate --draft case.draft.yaml
mmpbsa prepared-case inspect --draft case.draft.yaml
mmpbsa prepared-case seal --draft case.draft.yaml --output prepared/case_a
mmpbsa prepared-case validate --manifest prepared/case_a/prepared_case.json
```

使用 `mmpbsa.workflow.v3` 显式选择带版本的 preset、case 和独立 replica；下例仅
展示配置形状，运行前需提供实际封存输入和 runtime：

```yaml
schema_version: mmpbsa.workflow.v3
project_id: example
protocol:
  preset: production_3x5ns@1
  models: [gb, pb]
  completion_policy: all
cases:
  - case_id: ligand_a
    prepared_case: prepared/ligand_a/prepared_case.json
replicas:
  - {replica_id: rep01, replica_index: 1, seed: 41001}
  - {replica_id: rep02, replica_index: 2, seed: 41002}
  - {replica_id: rep03, replica_index: 3, seed: 41003}
```

```bash
mmpbsa --workflow workflow.yaml --runtime runtime.yaml config check
mmpbsa --workflow workflow.yaml config resolve
mmpbsa --workflow workflow.yaml --runtime runtime.yaml --workspace _workspace batch run
mmpbsa --workflow workflow.yaml --runtime runtime.yaml --workspace _workspace finalize case --case ligand_a
mmpbsa --workflow workflow.yaml --runtime runtime.yaml --workspace _workspace finalize project
mmpbsa --workflow workflow.yaml --runtime runtime.yaml --workspace _workspace report
mmpbsa --workflow workflow.yaml --workspace _workspace status --summary
```

新项目也可用 `mmpbsa project init` 从已封存输入和 runtime 模板生成 workflow、
smoke 和 campaign；初始化、预检和状态查询不启动计算。生产阶段要求对应 smoke
gate 通过。资源协调、恢复与导出方式见 [scheduling.md](docs/scheduling.md)。

## 科学参数与验证

支持 3 replica 的 5、15、50 和 100 ns production presets。窗口、评分帧、MD/GB/PB
参数、QC 阈值与引用以 [scientific-defaults.md](docs/scientific-defaults.md) 为准。
只有 replica 完整且 QC 和模型评分有效的 case 才有跨 case 比较资格；比较按模型和
内部介电常数分组，缺失值不会补造。

[文档索引](docs/README.md) 包含架构、环境、调度和可选引擎说明。公开输入与验证
步骤见 [小分子验证](validations/tyk2_3x5ns/README.md)及
[蛋白–多肽验证](validations/sp2016_3x5ns/README.md)。仓库不包含本地计算输出。

0.7 执行 workflow v3，不将旧 workflow/workspace 转换为可复用的当前 attempt；旧
记录仅支持只读 inventory。完整契约见 [architecture.md](docs/architecture.md)。
