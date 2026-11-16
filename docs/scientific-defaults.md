# Scientific defaults for production presets

本文定义 `production_3x5ns`、`production_3x15ns`、
`production_3x50ns` 和 `production_3x100ns` 的 `@1/@2` 默认值。`@1` 是冻结的
GROMACS contract；`@2` 是 GROMACS/OpenMM 共用的 engine-neutral phase contract。
二者的 production 时长和抽帧计划相同。这些 0.7 默认值是可审计、
版本化的工程基线，不是对所有靶点都已收敛的声明。改变科学参数应使用 typed
override 并产生新的 resolved science hash；不要手写 raw MDP/MDIN 绕过 preset。

## 1. Replica、窗口与 frame 计划

所有 production preset 都运行三个 seed 不同的独立 replica，XTC 从 0 ps 起每
20 ps 保存一帧。分析窗口两端都包含，故每个 replica、每个 model 的 frame 数为：

```text
frames = (analysis_end - analysis_start) / scoring_interval + 1
```

| Preset | Replicas | Production/replica | XTC interval | Analysis window | Scoring interval | XTC start:end:stride | Frames/replica/model | Frames/3 replicas/model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `production_3x5ns@1/@2` | 3 | 5 ns | 20 ps | 3–5 ns | 20 ps | 151:251:1 | 101 | 303 |
| `production_3x15ns@1/@2` | 3 | 15 ns | 20 ps | 5–15 ns | 20 ps | 251:751:1 | 501 | 1,503 |
| `production_3x50ns@1/@2` | 3 | 50 ns | 20 ps | 10–50 ns | 40 ps | 501:2501:2 | 1,001 | 3,003 |
| `production_3x100ns@1/@2` | 3 | 100 ns | 20 ps | 20–100 ns | 40 ps | 1001:5001:2 | 2,001 | 6,003 |

50/100 ns 只把 scoring interval 放宽到 40 ps，仍保留整个后 80% 时间窗口。这是在
长时间覆盖与 PB 成本之间的温和缩放：40 ps 等于 25 snapshots/ns，也是 20 ps XTC
网格上仍不低于约 20 snapshots/ns 的最疏间隔（下一个 60 ps 只有约 16.7/ns）。这
不意味着相邻 frame 已统计独立。GB 与 PB
共享同一个 `mmpbsa.frame-manifest.v1` 和 digest；选择两个 model 时，两者必须使用
完全相同的 start/end/stride，而不是各自抽帧。

窗口起点是固定 protocol burn-in，不是自动平衡检测。`3 ns`、`5 ns`、`10 ns`、
`20 ns` 之前的数据被排除，只说明本 preset 的计算约定；它不能证明之后的轨迹已
达到平衡。报告必须同时给出逐 replica 趋势、block/convergence diagnostics 和
between-replica spread。如果体系在窗口内仍漂移，应延长/重做或声明不可收敛，
不能移动窗口直到结果“好看”。

选择三个 replica 是有意的：Adler 与 Beroza 发现单条轨迹内部看似收敛，独立轨迹
之间仍可能给出不同 MM/PBSA 能量，而延长单条轨迹未必消除这种差异。因此 replica 是
uncertainty 的一级单位，frame 只是相关时间序列中的观测，不能把 6,003 个 frame
当成 6,003 个独立重复实验。

## 2. MD baseline

默认动力学采用 explicit solvent；体系准备的基线是 protein `ff14SB`、water
`TIP3P`、small molecule `GAFF2 + AM1-BCC`、10 Å solvent padding 和 0.15 M salt。
多肽按 peptide/protein force field 处理；非标准残基、糖基、cap、PTM、staple、
金属和共价化学必须在 PreparedCase provenance 中显式记录，不能靠 residue name
自动猜测。

下表是 `@2` phase plan 及两个 engine profile 的明确 lowering。冻结的 GROMACS
`@1` 仍保留原有 steepest-descent EM（`emtol=1000`、`emstep=0.001 nm`），不由此表
追溯改写。

| Phase/setting | Common scientific target | GROMACS profile | OpenMM profile |
|---|---|---|---|
| Energy minimization | no constraints; accept only final maximum force ≤ `1000 kJ mol⁻¹ nm⁻¹`; 50,000-step/iteration ceiling | steepest descent, `emtol=1000`, `define=-DFLEXIBLE` | L-BFGS, optimizer tolerance `10` |
| NVT | 100 ps at 300 K; replica/phase-derived seed | V-rescale, `tau_t=1 ps` | LangevinMiddle, friction `1 ps⁻¹` |
| NPT | 200 ps at 300 K and 1 bar | C-rescale, `tau_p=5 ps`, compressibility `4.5×10⁻⁵ bar⁻¹` | MonteCarloBarostat every 25 steps |
| Production | 2 fs; H-bond constraints; PME; 1.0 nm cutoff; dispersion correction | LINCS; Verlet list every 20 steps | constraint tolerance `1e-5`; rigid water; COM removal every 100 steps |
| Trajectory | full-system XTC includes production `t=0`, then every 20 ps | GROMACS XTC | OpenMM XTCReporter |

GROMACS `@2` 的 EM 在 `mdrun` 前额外检查 `grompp -pp` 生成的预处理 topology；只要
仍存在非空 `[constraints]` 或 `[settles]` section 就拒绝执行，不能把未响应
`-DFLEXIBLE` 的外部 topology 当作无约束体系。

这些值延续 v1 的已验证基线目标，但不是对复杂修饰多肽的充分平衡保证。若 preparation
证据表明需要更长 restrained equilibration，应建新 preset/version，而不是让某个
case 的隐式脚本改变公共 preset。

workflow 可覆盖的科学字段被刻意限制为 temperature、pressure、salt、internal/
external dielectric、interface/contact cutoffs 和四个 QC threshold。冻结的 `@1`
native engine override 使用 allowlist：GROMACS 只开放 EM 的 `emtol/emstep`，以及
NVT/NPT/production 的 thermostat/barostat、compressibility 和 LINCS 调节；`@2`
拒绝 GROMACS native override，由 engine profile 完整 lower。同为两版使用的 Amber
只开放 GB `surften` 与 PB `fillratio/maxitn/dprob`。production duration、frame window、
stride、`igb/ipb/inp`、model topology 和 selection 不能通过 raw text 改写；要改变
这些 contract，应发布新的 preset version。

## 3. Explicit-solvent trajectory、dry GB/PB

显式水和离子参与 MD。结构水可在 preparation 判断后保留于 MD 体系，例如稳定
binding-site 几何的晶体水；MD 本身不把它称作 receptor 或 ligand。

baseline scoring 是 dry single-trajectory MM/PB(GB)SA：PreparedCase 的 scoring
投影从同一 MD frame 中抽出 primary receptor + ligand，并使用匹配的 dry complex、
receptor 和 ligand topologies。bulk water、ions 与 excluded water 不进入 baseline
GB/PB。显式 wet/Nwat 方案可能对某些口袋有价值，但需要不同、明确版本化的方法学，
不在四个 production preset 中。

默认同时启用：

| Model | Main defaults |
|---|---|
| GB | `igb=5`, `mbondi2`, salt `0.15 M`, external dielectric `78.5`, `surften=0.0072` |
| PB | `ipb=2`, `inp=2`, `mbondi2/radiopt=0`, salt `0.15 M` (`150 mM` in native PB input), external dielectric `80`, probe `1.4 Å`, `fillratio=4` |
| Entropy | off by default |

GB/PB 可在 `protocol.models` 中独立选择。默认 `[gb, pb]` 和
`completion_policy: all`，表示报告要求两者都有明确 completed/skipped/failed
状态；这不表示把二者混成一个平均 score。熵默认关闭，因为 normal-mode entropy
成本高、噪声大，并且在同系列相对比较中常不值得引入额外不确定性；需要时应作为
独立、版本化的分析协议。

## 4. Primary binding 与 ligand-aware QC

MD 不认识 ligand；后处理必须认识。每个 PreparedCase 必须声明唯一 `primary`
binding，并给出 receptor/ligand/excluded entity IDs 和跨 MD/scoring 原子序稳定的
命名选择。不能用“最后一个 residue”“非蛋白 residue”或临时 index group 猜 ligand。

ligand 的主 QC selection 由实体类型决定：

- `small_molecule`：所有 ligand 重原子 RMSD；
- `peptide`：peptide backbone RMSD，修饰仍留在同一 ligand entity，但不让柔性侧链
  或糖基支配 backbone stability 指标。

## 5. 四个 QC gates

QC 只做少而关键的 decision gates；温度、压力、能量和所有原始曲线仍可进入完整
report/provenance，但不重复变成几十个同义 PASS 项。
`@2` QC 沿用表中的 analysis window 帧数，却统一以 post-NPT production 的
`t=0` 构象经相同 autoimage/receptor-fit 处理后的单帧轨迹作为 RMSD 和
native-contact 参考；receptor interface backbone 的对齐
与 ligand primary selection 保持一致。以 5 ns preset 为例，QC 计算 3–5 ns 的
101 帧相对 `t=0` 的偏移；0–3 ns 内出现又恢复的短暂偏移不进入这四个 gate，
应从完整生产轨迹诊断。冻结的 `@1` QC 定义不追溯更改。

| Gate | Selection/reducer | Default decision |
|---|---|---|
| Interface backbone stability | 初始 5 Å interface 的 receptor backbone，aligned RMSD P95 | `> 3 Å` → `BLOCK` |
| Ligand primary stability | small molecule heavy atoms 或 peptide backbone，RMSD P95 | `> 5 Å` → `WARN` |
| Native-contact retention | 初始 receptor–ligand heavy-atom contacts，4.5 Å cutoff；窗口内最小保留数 | `< 1` → `BLOCK` |
| Binding separation | receptor–ligand minimum heavy-atom distance；窗口内最大值 | `> 5 Å` → `BLOCK` |

ligand RMSD 只告警：大而仍结合的 ligand 可以发生合理构象变化。初始原生接触归零
表示初始接触模式丢失，不等于所有受体接触归零，也不能单独证明永久解离；距离 gate
提供结合分离证据，仍应结合轨迹判断。BLOCK 报告补充原生接触归零占比、最长连续
零接触样本数及有明确 nonnative 表时的其他/总接触信息，均为解释而非新增 gate。
任何 `BLOCK` 都作为科学 outcome 保留，所选 GB/PB
task 显式 skipped；报告不得把它改写成执行错误，也不得省略该 replica。

## 6. Automatic internal dielectric

GB/PB 的默认 `internal_dielectric: auto` 从明确 primary binding 的界面化学分类，
只允许 `2/3/4`：

- `2`：中性、以非极性环境为主；
- `3`：中性但界面有显著 polar residues/groups；
- `4`：charged ligand 或界面存在 charged residues/groups。

小分子先检查显式 formal charge，再检查 receptor interface；peptide 同时检查
receptor 与 peptide interface。分类输入必须来自 PreparedCase entity、残基/原子
选择和 provenance，而不是 MD stage 猜测 ligand。遇到未知/不能可靠分类的修饰、
atom type 或 charge metadata 时，发出 `WARN` 并保守 fallback 到 `4`；不静默选择，
也不阻断整个 case。显式 override 只接受 `2`、`3` 或 `4`。
每个评分 replica 都记录实际 ε；固定值也必须记录最终数值，不能只保留配置模式。
同一 case/model 的 replica 若采用不同 ε，该 model 不进入比较。跨 case 的
排名与相关性按 model 和实际 ε 分组，不把 ε=2/3/4 的分数混成一个总名次。

## 7. 报告与统计解释

`whole_molecules_v1` 的逐帧键完整性检查属于轨迹数据有效性门禁，不新增科学
QC 指标。评分坐标先按拓扑重建完整分子再成像，物理 NetCDF 不做 RMS 对齐；
`@2` QC 的接触距离使用 `noimage`，RMS 对齐只作用于诊断或展示。旧处理版本的
评分即使解析成功，也不能替代经过当前几何门禁的数据。

报告应完整呈现每个 model、case 和 replica 的状态、均值/区间、between-replica
spread、逐帧趋势、收敛诊断、四个 QC gate、分解、实验相关性、结构资产、参数与
provenance，但不重复同一事实。每个表应注明统计单位；frame-level standard error
不能替代 replica-level uncertainty。相邻 20/40 ps snapshots 很可能自相关，frame
总数只用于确定计算负载和时间覆盖，不是有效样本数。若报告 effective sample size，
必须说明估计方法并保留原始 frame 数。
partial 或 scientifically blocked 的 case 可显示已取得的描述性分数，但不进入
正式名次或相关性。比较资格按 model 判定，要求全部声明 replica 均已完成，
各自 QC 为 PASS/WARN，且该 model 在每条 replica 上都有有效分数；缺少实验
reference 时仍可在对应 ε 组内排名，但不参与相关性。

逐帧曲线为 binding score（受体、ligand、complex 的差），不是整个 MD 系统的总能量。
累计均值逐渐变平本身不能证明平衡或收敛：累积平均天然会压低后期短时波动，必须
结合跨 replica 一致性、窗口漂移与采样覆盖。只有 canonical frame selection 与实际
receipt 的 selection/digest 相符、实际帧轴为完整 1..N 时才显示绝对 `Time (ns)`；
缺少可验证来源的 legacy 数据显示 `Analysis frame`，不从 preset 名称猜时间。

`Ligand RMSD occupancy basin` 是一维构象占据诊断。trajectory v3 的完整 dry
projection 明确保留 post-NPT production `t = 0 ps`，每个 replica 分别以该构象为
参考；它直接复用 `prepared_reference_trajectory`，其中已按化学键图重建完整分子，
并做与评分轨迹一致的 ligand-anchor/receptor-fixed PBC autoimage。
诊断时再用 `receptor_interface_backbone` 对齐，最后用
`ligand_qc` 计算 ligand RMSD（小分子重原子、多肽 backbone）。为使多条曲线能在同一
RMSD 横轴上直接比较，所有可用 replica 共用全局最小/最大值和同一组 bin 边界；每个
replica 的占据计数和零点仍分别计算。绘制 `−RT ln(P/Pmax)`，其中温度直接来自 resolved
protocol，最高占据 bin 为零点。默认 bin 数为 `clamp(4√Nmax, 6, 40)`，`Nmax` 是可用
replica 中最大的 frame 数。这不是 GB/PB binding energy，也不是严格的
thermodynamic free-energy surface；不同 replica 的零点也不能当作绝对自由能比较。
未采样的零占据 bin 不赋值，曲线在该处断开，不跨 gap 连线。
该 RMSD 只在 `report.assets` 重建时计算；`@2` QC 与它共用 post-NPT `t=0`
参考，但 QC 仍仅检查预设 analysis window，不新增 workflow stage，也不重跑 MD
或 MMPBSA。报告必须验证 cpptraj frame
从 1 开始且连续，不按行号静默修补缺失 frame。

公共离线交接导出 `cases.csv`（每 case/model 一行）、`replicas.csv`（每 replica/model
一行）、中文字段字典与验证摘要，CSV 使用 UTF-8 BOM。权威分析中的 selected score
已完成单位转换，必须与 native score/单位分列；不要把 kJ/mol 再乘 4.184。QC BLOCK
replica 不计分，空值不填零；n<2 时 replica SD/SEM 不可用。部分 BLOCK case 可保留
其余 replica 的描述性分数但不排名；排名仅复制原项目可比组，不能跨队列拼接。
可选 `mmpbsa.scientific-context.v1` 摘要只包含已封存的实体、选择、修饰/建模假设，
以及可验证的 reference 和 frame receipts；旧来源缺失明确为 unknown，不能补造。

## 8. 依据与限制

- Adler & Beroza, *Improved ligand binding energies derived from molecular
  dynamics: replicate sampling enhances the search of conformational space*,
  JCIM 2013, [DOI 10.1021/ci400285z](https://doi.org/10.1021/ci400285z)：支持使用
  多条独立 trajectory 并显式报告 trajectory 间差异。
- [Amber MM-PBSA tutorial](https://ambermd.org/tutorials/advanced/tutorial3/index.php)：
  说明从 equilibrated MD ensemble 取 snapshots 的 single-trajectory MM/PB(GB)SA
  工作流，也明确教程参数只是示例而非普适最优值。
- Genheden & Ryde, *The MM/PBSA and MM/GBSA methods to estimate ligand-binding
  affinities*, [DOI 10.1517/17460441.2015.1032936](https://doi.org/10.1517/17460441.2015.1032936)：
  综述 endpoint 方法的近似、采样、介电常数和 entropy 限制。
- Homeyer & Gohlke, *Free Energy Calculations by the Molecular Mechanics
  Poisson–Boltzmann Surface Area Method*,
  [DOI 10.1002/minf.201100135](https://doi.org/10.1002/minf.201100135)：
  总结 MM/PBSA 实践中的轨迹、模型与误差来源。
- Wright et al., *Computing clinically relevant binding free energies of HIV-1
  protease inhibitors*,
  [DOI 10.1021/ct4007037](https://doi.org/10.1021/ct4007037)
  ([open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3966525/))：
  在该体系中比较 sampling rate，观察到超过约 20 snapshots/ns 的收益很小。这里
  仅把它作为 50/100 ns 使用 40 ps stride 的成本依据之一，不外推成通用收敛定律。

文献不能替任何新体系证明平衡或收敛。四个 preset 的窗口、frame 数和 QC 阈值是
0.7 的显式默认 contract；项目应根据 replicate consistency、时间序列和已知化学
风险决定是否升级 preset 或建立新的版本化 protocol。
