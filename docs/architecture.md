# mmpbsa 0.7 architecture

## 模块与 provider 边界

`mmpbsa` 是单一 distribution，顶层使用 PEP 420 namespace。依赖方向为
`backends → core → utils` 和 `report → core → utils`；backends 与 report 不互相
导入，core 不静态导入具体实现。utils 只提供 JSON/hash、文件事务、路径、时间、
锁及 subprocess 等技术能力；core 持有 workflow、workspace、runner、analysis、CLI
和 contracts；backends 适配 GROMACS、OpenMM、AmberTools、cpptraj、ParmEd 和
展示资源；report 只消费 analysis 与已声明资源。

Provider 通过 `mmpbsa.providers.v2` entry point 发现，声明 provider ID、API 版本
`2`、distribution 版本、capabilities 与 `get_service()`。Service 只交换 core 定义的
`BackendCompatibility`、`StageInput`、`StageRequest`、`StageResult` 和 `StageService`。
workflow v3、PreparedCase v2 与 provider ABI v2 分别版本化。OpenMM 是同一
distribution 中单独注册的可选 MD provider，按需加载 OpenMM/MDTraj；没有安装
它们不会影响 GROMACS `@1`。backend 根据显式 MD 输入表示消费 GROMACS
`top+gro` 或 OpenMM `prmtop+inpcrd` 快照，不按文件后缀猜测。

Preparation helper 可作为普通 callable，但不是隐式 stage；体系特异建模属于项目
或 validation。旧 preparation v1/v2 reader 和 dispatcher 保留兼容，使用
`PreparationIntent = PreparationSpec | PreparationSpecV2`，只修缺陷，不接纳新的
assembly、cofactor、metal 或 reconstruction 特例。

## 1. 产品边界

0.7 是统一的、replica-addressable MM/PB(GB)SA workflow framework。它不是某个
靶点的 runner，也不会在运行时猜测蛋白链、ligand、共价连接、修饰或 protonation。
体系准备在 workflow 外完成，并产出经过完整化学决策的只读
`mmpbsa.prepared_case.v2` manifest。admission 重新校验所有外部 artifact 的路径、
尺寸和 SHA-256。每个 stage 执行前将实际消费的输入复制到 attempt 内的固定
快照并再次核对摘要；恢复也验证并复用该快照。

复杂建模属于 validation/project，而不是 `src/mmpbsa`：逐例程序决定 source model、
biological assembly、altloc、缺失结构、端基、辅因子、PTM、显式键、参数化和溶剂化。
框架提供的 `mmpbsa.prepared_case_draft.v1` 只是机械 sealing port，输入必须已经包含
最终 MD/scoring topology、coordinates、scoring→MD 映射、entities/selections、唯一
primary binding、index 和建模 receipt。它不打开原始 PDB/mmCIF，不生成或修复化学。
sealing admission 只验证可由最终产物证明的事实：MD→dry scoring 的 atom type、
charge/mass、LJ 与内部 bonded terms，Amber radii，entity 的 residue/name/atom
containment，heavy/backbone selection，以及两套 topology 都不存在 receptor–ligand
跨边界共价键。哪一部分应被声明为 receptor、peptide ligand 或修饰实体，仍必须由
validation 明确决定；框架不会反向猜测生物学角色。

配置层是 `mmpbsa.workflow.v3`。workflow 选择 package-owned、带版本号的 preset，
只允许 typed overrides 和受限的 engine overrides。冻结的 `@1` MD contract 只能由
GROMACS 执行；engine-neutral `@2` 另选 `md_engine_profile`，当前支持
`gromacs_canonical@1` 和 `openmm_8_5@2`。GROMACS MDP、OpenMM request 和 Amber
control 均由 canonical resolved protocol 生成；raw native input 不是 workflow
source of truth。可用 `config resolve` 审计完整 science hash 输入，用
`config explain` 查看窗口、stride 和 frame 数。

## 2. MD 与实体语义分离

MD system 是 role-neutral 的物理体系。MD stage 只消费 `PreparedMdView` 中所选
输入表示的拓扑、坐标、原子数和 system signature；它不接收 index、entity 或 binding，
也不因同一组原子后来被解释为 receptor、ligand 或 excluded entity 而改变动力学。
GROMACS profile 选用 `topol.top + system.gro`，OpenMM Amber profile 选用
`solvated.prmtop + solvated.inpcrd`。旧 OpenMM GROMACS 输入结果与新表示具有
不同科学身份，不能交叉复用。

`PreparedCase v2` 在不污染 MD 的前提下为后处理提供确定语义：

- `md`：完整显式溶剂体系及其 MD 原子序；
- `scoring`：dry-scoring 原子序、radii model、topology/coordinates，以及从 scoring
  原子到 MD 原子的逐原子投影；
- `entities`：稳定 `entity_id`、类型、MD/scoring 原子索引、残基身份和命名选择；
- `primary_binding`：一个且仅一个 thermodynamic binding，显式列出 receptor、
  ligand、excluded entities 及 complex/receptor/ligand topology roles。

一个 case 当前只能有一个 `binding_id: primary`。primary ligand 的实体类型只能是
`small_molecule` 或 `peptide`。修饰多肽的 cap、PTM、glycan、cyclization 或 staple
通过 feature 保留在同一个 peptide entity 内，不能因为 residue name 特殊而从
ligand selection 丢失。receptor–ligand 的跨实体共价键不属于当前 single-trajectory
endpoint contract，admission 会拒绝；ligand 内部修饰连接允许存在。

同一实体的命名选择同时携带 MD 与 scoring 投影。由此 trajectory/QC 可以在显式
溶剂 MD 轨迹上工作，而 GB/PB 使用严格 dry 的 receptor/ligand/complex topology，
不依赖易漂移的 residue name 或自动 index group。

`@2` MD provider 的公共完成边界是 `full_trajectory`、`coordinates`、`state`、
`checkpoint` 和 `energy_log`。GROMACS 与 OpenMM 的 phase-local 文件仍是各引擎私有
审计证据。trajectory v3 才同时接收 role-neutral 的完整轨迹、MD 的 `md_topology`
与 `md_coordinates`、dry complex topology、显式 scoring→MD 原子投影和 binding
selections；它保留
post-NPT production `t=0` 的 dry source reference；先根据经核验的化学键图流式重建
完整分子，再用 cpptraj 做 ligand-anchor/receptor-fixed autoimage。
`prepared_reference_trajectory` 与评分用 `prepared_trajectory` 保留未经 RMS 对齐的
物理坐标和匹配盒信息；仅展示用 `centered_trajectory` 做 interface fit。
原始投影 `selected_trajectory` 与重建后的 `whole_trajectory` 分别留作审计。
处理修订为 `whole_molecules_v1`，评分必须具有逐帧键完整性审计；旋转盒无法写入
NetCDF 的警告不能视为成功。因此 MD provider 无需、也不允许识别 ligand。冻结的 GROMACS `@1` 继续发布
既有 native `tpr/log/edr/gro/xtc/cpt` roles，不迁移为新边界。

## 3. 固定 stage graph

```text
case admission
   ├─ replica 1: MD → trajectory → trajectory_qc → scoring/<model_id>
   ├─ replica 2: MD → trajectory → trajectory_qc → scoring/<model_id>
   └─ replica 3: MD → trajectory → trajectory_qc → scoring/<model_id>
          ↓
explicit finalize_case (complete or reasoned partial inventory)
          ↓
project analysis → optional display assets → typed report
```

原子调度单位是 `(case_id, replica_id)`；scoring task identity 再加入 `model_id`。
GB 和 PB 因而拥有独立 science hash、attempt history、失败/重试状态与资源调度，
可以选一个或同时选择。所有所选 model 绑定同一个 frame-manifest digest。
`completion_policy=all` 要求全部所选 model 完成，`any` 接受至少一个；production
preset 默认 `[gb, pb]` 与 `all`。

trajectory QC 先于 scoring。它只保留四个决策 gate：interface backbone RMSD、
ligand primary RMSD、native-contact retention 和 receptor–ligand minimum distance。
`@2` QC 的分析帧仍由 preset 窗口决定，结构和 contact 的参考帧统一为
post-NPT production `t=0`；它不检查窗口前出现又恢复的短暂事件。
`BLOCK` 是科学结论：该 replica 的所选 scoring task 被跳过，而不是伪装成 backend
崩溃；finalization、partial analysis 和 report 仍可继续并展示原因。

CPU/GPU lease 是 stage-scoped；默认 process-local，旧 runtime 行为不变。
新项目可显式选择同 Linux 主机/用户的 cooperative host registry，在参与的
tmux/CLI controller 间原子仲裁 physical-core/GPU-UUID 租约。它不拥有 Slurm、
cgroup 或显存隔离，不排除其他用户或手工程序，不抢占已有计算。policy 必须显式
初始化；等待不占部分 CPU/GPU，存活子进程或未知证据的孤儿租约不按 heartbeat
超时归还。只读状态、preflight、inspect/dry-run reconcile 均无写入。

可选 `scheduler.execution_mode=pipeline` 将执行单位细化为就绪 stage，仍由同一
process-local pool 仲裁；MD、轨迹/QC 和 GB/PB 因而可重叠推进。可选
`cpu_affinity=physical` 分配具体且不重叠的物理核，CPU/GPU ID 仅进入执行记录，
不进入 science 或 backend compatibility identity。保留核保护未结束的 MD 与
后处理阶段，队列等待不占 worker。旧 runtime 不含这些字段时保留原有行为。
配置、线程限制和资源边界见 [scheduling.md](scheduling.md)。

## 4. Science、runtime 与 reuse

`workflow.yaml` 只包含可哈希科学输入：project、protocol preset/overrides、cases、
replica seeds、analysis 和 report 声明，且禁止 shell 环境插值。`runtime.yaml` 只含
provider、可执行程序、环境、本机 GPU、worker/thread 和资源额度。

完整 engine version 写入 invocation provenance。reuse/resume compatibility key
包含 provider、adapter contract 和 adapter-owned engine identity；通用 provider
至少按 major 隔离，OpenMM 的 binary checkpoint 进一步把 exact version、platform、
precision 和 deterministic properties 编入 identity，且实际 worker 版本必须与
预留 attempt 时探测的版本一致。science hash 按 stage 使用最小必要的
PreparedCase view：MD 只绑定 role-neutral MD 身份；trajectory/QC/scoring 才绑定
实体、primary binding 和 dry-scoring 投影。因此仅改变后处理解释不会错误地宣称
MD 物理输入改变，但会正确失效相关下游任务。
GROMACS `.top` 的递归 `#include` 依赖必须封存并纳入消费它的 stage hash；
外部、未声明或无法静态解析的 include 在执行前拒绝。
每条 upstream identity 还绑定已验证 artifact 的 size/SHA-256、上游 science hash、
canonical payload digest 与 attempt ID。后两项是必要输入，因为 case/project/report
会消费 payload 并把 attempt provenance 写入发布物；相同文件字节的新 attempt 不能
复用仍指向旧 attempt 的总结或报告。

wire contract 新增强制证据时必须显式 bump adapter contract。当前 builtin scoring
使用 `mmpbsa.backends.scoring.v3`，要求 observed frame IDs 的 count/range/digest；
core case finalizer 使用 `stage.finalize_case.v3`，要求同 replica 的 GB/PB planned
manifest 与 observed frame identity 一致。旧 scoring/finalizer v2 attempt 不可复用；
比较资格和介电分组改变分析含义后，analysis 使用新的 contract；旧分析结果可
只读审计，不作为新报告的可复用统计输入。

## 5. Workspace 与 attempt

当前 canonical workspace 仍使用 workspace/attempt v2；产品版本 0.7、workflow v3
和 provider ABI v2 是三个独立版本轴。

```text
workspace.json
cases/<case>/admit/attempts/0001/
cases/<case>/replicas/<replica>/<md|trajectory|trajectory_qc>/attempts/0001/
cases/<case>/replicas/<replica>/scoring/models/<model_id>/attempts/0001/
cases/<case>/finalize_case/attempts/0001/
project/<analysis|assets|report>/attempts/0001/
report/                            atomically exchanged stable tree on Linux
```

每个 attempt 有独立 `state.json`、`output/` 和 invocation ledger。完成、失败和取消
attempt 封存；活动 attempt 只能显式 `resume`，失败 attempt 只能显式 `retry`，
science 改变或 backend major 不兼容时显式创建新 attempt。产物逐个记录路径、尺寸
和 SHA-256；reuse、upstream hand-off 与 report publication 都会重新验证。

## 6. Explicit-solvent MD 与 dry scoring

标准 MD 使用 explicit solvent。结构水是否进入 MD 是准备阶段的物理体系决策；若
保留，它们属于 MD 原子序和相应实体/metadata。baseline GB/PB 采用 dry scoring：
scoring 投影移除 bulk water、ions 和所有被列为 excluded 的水实体，只覆盖 primary
receptor + ligand。wet MM/PBSA 或 Nwat 不属于 0.7 baseline，不能通过不透明 mask
偷偷启用。

## 7. 报告数据边界

`mmpbsa.report` 不扫描 workspace，不调用科学工具，只消费显式 analysis payload、
project inventory 和声明过的 asset artifacts。报告覆盖 project/case inventory、
replica 状态、GB/PB 统计与不确定性、逐帧趋势、收敛、QC、分解、相关性、结构资产
和 provenance，同时遵守“一个事实、一个权威展示位置”：

- project overview 只给 inventory 与跨 case 比较，不复制 case 明细；
- case score table 只给聚合，replica/frame 细节只在唯一的 replica summary/trend；
- QC 区只列真实 issue，PASS 状态不重复生成告警表；
- 缺失、被 QC 阻断或未选择的模型明确显示 unavailable/N/A，不补造数值。

`mmpbsa.analysis.v3` 为每个 case/model 单独判定比较资格。所有声明 replica 均完成、
QC 为 PASS/WARN、该 model 的分数均有效时才进入名次与相关性；partial 或 BLOCK 的
现有分数仍可展示，但不授予名次。每项都输出 `comparison_eligible`、
`comparison_exclusion_reasons`、`resolved_internal_dielectric`、`comparison_group`
与组内 `rank`，`cases.csv` 同步提供这些审计列。自动或固定的内部介电常数都记录
实际 ε；跨 case 比较按 model 和 ε 分组，`correlation_groups` 也分别计算，不生成
混合 ε 的总名次或 pooled correlation。报告只呈现 analysis 的资格结果，不另行推断；
旧 analysis v2 仅用于既有报告的只读展示。

validation 可在 `report.case_catalog` 中提供通用的 eligible/preflight-rejected
inventory。catalog 只增加展示页面和 provenance，不进入 score aggregation 或
correlation；preflight-rejected 页面不生成伪 replica、trajectory、energy 或
“correlation unavailable”区块。

报告由 typed blocks 组成，扩展只能向命名 slot 提交 JSON-typed contributor，不能
注入 raw HTML。assets 与 HTML 在新 report attempt 内构建并审计链接，再事务发布。

展示资源默认使用 `[receptor, ligand]`，也可在 `report.assets.components` 中显式
选择 PreparedCase 的实体 ID，例如 `[receptor_A, receptor_B, heme_A, heme_B, ligand]`。
选择使用 dry scoring 原子索引；`receptor` 是 primary binding 的 receptor 实体并集。
角色组和其实体重叠时拒绝，未知 ID 拒绝，MD-only excluded 实体在 dry display
中显示 unavailable。嵌套 `report.assets` 设置覆盖历史顶层 display 设置；自定义文件
链接使用独立的 `report.asset_links` 描述符，历史 `report.assets` 描述符序列仍兼容。
这些设置仅改变展示，不修改 MD、能量分区或跨 binding 共价键的兼容性。
调整组件后用 `report` 重新生成资源和 HTML；`--reuse-assets` 仅复用已验证兼容的
资源并重建 HTML，不能用于修复错误的组件选择。

Preflight-only 页面使用 `preflight_pending`、`preflight_skipped` 或
`preflight_rejected`，每项必须有明确原因，不能覆盖 workflow 的真实 case，也不
生成伪 replica、轨迹、能量、QC 或空排名列。

0.7 的 portable report boundary 是 `mmpbsa.report_assets.v3`，其中每个
descriptor 显式声明 `visibility=visible|hidden`。display bundle 使用
`mmpbsa.display_assets.v4`：每个 replica 含一帧 topology PDB、完整的对齐 XTC、
独立 PyMOL loader，以及可选的隐藏 RMSD/reference diagnostics。原始 cpptraj
input 含 host-local 路径，只保留在 sealed assets attempt，绝不发布。旧的
report-assets v1/v2 和 display-assets v3 不迁移、不兼容。

报告内 SVG 最多展示 400 个保端点/局部极值的数据点；完整 analysis payload 以
`data/analysis.json.gz` 发布。stable report 发布时按 sealed artifact inventory
物化，并生成含 size/SHA-256 的 `publication_inventory.json`。
替换已有报告目录时，发布者在同一文件系统上以 Linux `renameat2(RENAME_EXCHANGE)`
交换新旧树；并发读取者始终能看到一个完整目录。文件系统不支持原子交换时，替换在
移动旧目录前失败。首次发布只需单次 rename。

`services.py`、`runner.py` 和两个 preparation builder 是已知拆分边界，不允许继续
吸收新的 schema parser 或 renderer。PreparedCase 的 stage-scoped science projection
已从 runner 拆到 `prepared_projection.py`；GROMACS phase lowering、OpenMM worker 和
trajectory v3 也各自隔离。新 filesystem publication primitive 位于 utils。当前
line-count budgets 由 architecture test 锁定，避免模块在进一步拆分前继续恶化。

## 8. Breaking 与 legacy 边界

0.7 只执行 `mmpbsa.workflow.v3`。`mmpbsa.workflow.v2`、更旧配置和 legacy
workspace 不可执行，不能迁移为当前 attempt，也不能进入 science reuse 链。
`workspace inspect --legacy-workflow ... --legacy-workspace ...` 只读取并列出已知
inventory；它不改写源目录、不创建 backup、不生成 v2/v3 state，也不推断缺失的
PreparedCase entity 语义。需要重新计算时，应显式创建 workflow v3、PreparedCase
v2 和新的 workspace，旧结果继续作为独立只读档案保存。
