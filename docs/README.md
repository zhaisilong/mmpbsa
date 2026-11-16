# 0.7 documentation

- [architecture.md](architecture.md)：workflow v3、PreparedCase v2、实体与
  primary binding、固定 stage graph、science/runtime 和 legacy 边界。
- [scientific-defaults.md](scientific-defaults.md)：四个 production presets、精确
  frame 计划、MD/GB/PB 参数、四个 QC gate、介电常数策略与文献依据。
- [environment.md](environment.md)：隔离环境、工具版本记录与 clean-wheel release gate。
- [scheduling.md](scheduling.md)：物理核分配、阶段流水线、合作式跨项目租约、新项目
  控制器、只读状态、ETA、公共导出与兼容恢复；不改变科学参数。
- [openmm.md](openmm.md)：可选 OpenMM MD provider、Amber 输入表示、`@2` phase
  plan、artifact、恢复边界与当前验证状态。
- [`validations/tyk2_3x5ns/README.md`](../validations/tyk2_3x5ns/README.md)：
  公开小分子 benchmark 输入、来源和验证步骤。
- [`validations/sp2016_3x5ns/README.md`](../validations/sp2016_3x5ns/README.md)：
  19 个 protein–peptide case 的公开输入、source audit 和建模说明。

0.7 的可执行 contract 是 `mmpbsa.workflow.v3` +
`mmpbsa.prepared_case.v2`。GB/PB 是共享 frame manifest 的独立 model task，默认
同时选择并采用 `completion_policy=all`。报告可以很全，但每项指标只能有一个
权威展示位置；overview 负责索引，case page 负责细节，缺失值明确显示 unavailable。

`mmpbsa.workflow.v2` 和 legacy workspace 仅支持只读 inspect。0.7 没有 migration
路径，也不会把 legacy evidence 放入当前 science-reuse 链。
