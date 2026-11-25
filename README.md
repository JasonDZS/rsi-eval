# rsi-eval

递归自改进（RSI）框架的三层能力评测框架：在同一套协议下回答三个问题。

1. **目标任务性能**：对齐成本后，自改进是否比“多试几次”更好。
2. **未见任务泛化**：增益换题、换模型后是否还在。
3. **改进器自身进化**：被改过的改进器是否真的更会改进。

实验计划里的 E1–E11 全部可以用一条命令跑，可中断续跑，跑完自动按开跑前写死的标准给出判定。

## 快速开始

```bash
pip install -e ".[dev]"
pytest                                                   # 离线测试，不联网不花钱
rsi-eval run --suite configs/suites/smoke.yaml           # 假模型跑完全部 11 项实验，约 1 分钟
cat runs/smoke/report.md
```

用真实模型：

```bash
pip install -e ".[anthropic,data]"                       # 其他厂商加 openai
export ANTHROPIC_API_KEY=...
rsi-eval data prepare gsm8k gsm_hard mmlu drop mgsm gpqa # 导出固定的数据文件到 data/
rsi-eval protocol freeze                                 # 写死判定标准
rsi-eval plan --suite configs/suites/pilot.yaml          # 看要跑多少次，不调用模型
rsi-eval run  --suite configs/suites/pilot.yaml --yes    # 付费模型必须显式 --yes
rsi-eval report --suite configs/suites/pilot.yaml
```

## 实验与代码的对应

| 编号 | 实验 | 实现（`rsi_eval/experiments.py`） |
|---|---|---|
| E1 | 基线复现 | 初始执行者在报告集和未见基准上计分 |
| E2 | 自改进演化 | `evolve()` 跑 T 轮，每一代的改进器、执行者、存档冻结存盘 |
| E3 | 同预算重复采样对照 | 初始系统用 E2 花掉的 token 多采样，多数投票（代码题用“任一次通过”作上界） |
| E4 | 未见 Benchmark 迁移 | 冻结 E2 选出的系统，在未见基准上计分 |
| E5 | 换模型迁移 | 同一系统把执行模型换成 `transfer` 角色的模型 |
| E6 | 新旧改进器对照 | 第 0 代、中间代、最终代改进器各做一次 `improvement_run()`，外加“第 0 代多次重启取最好” |
| E7 | 只累积经验对照 | 改进器固定为第 0 代但保留存档，预算与 E2 对齐；可指定另一个框架当对照（ADAS 对 Gödel Agent） |
| E8 | 演化成本入账对照 | 第 0 代改进器的预算加上演化出最终代所花的 token 再比 |
| E9 | 相邻代对比 | 每一代都做改进运行，看增益斜率 |
| E10 | 改进器改改进器 | 各代改进器先改一轮改进器，再比子代带来的增益 |
| E11 | 换模型配置重跑 | 用另一个模型配置重做 E2 和 E6 |

协议的硬约束都写在代码里：

- **同起点**：每次改进运行都从初始执行者、空存档出发，只换改进器。
- **预算对齐**：所有模型调用经过 `Meter` 记账（`runs/<套件>/ledger.jsonl`），预算按 token 截断；演化、选择、评测三个阶段分开记。
- **留出集**：数据确定性地切成搜索、选择、报告三份，演化和选择阶段读报告集会直接报错。
- **重复与统计**：每个条件按档位做 3 或 5 次独立运行，报告给出配对差值的 95% 置信区间和置换检验 p 值，并按实测方差估计需要的重复次数。
- **判定标准写死**：`configs/protocol.yaml` 的哈希记在每个结果里，`rsi-eval protocol freeze` 之后再改会被拒绝。
- **花费上限**：套件里的 `hard_cap_usd` 到了立即中止。

## 可插拔的四个部分

| 部分 | 位置 | 怎么加 |
|---|---|---|
| 框架适配器 | `rsi_eval/frameworks/` | 继承 `FrameworkAdapter`，实现 `initial / step / solve / candidates`，见 [docs/adapters.md](docs/adapters.md) |
| Benchmark | `rsi_eval/benchmarks/` | 继承 `Benchmark`，实现 `load / score`（可选 `prepare`） |
| 模型 | `configs/models.yaml`、`rsi_eval/llm.py` | 在 yaml 里加一项；新厂商写一个 `LLMClient` 子类 |
| 预算与档位 | `configs/protocol.yaml`、`configs/suites/` | 改 `tiers` 或在套件的 `params` 里覆盖 |

## 目前的状态

已实现并在离线假模型上跑通：全部 11 项实验的编排、记账、缓存、续跑、统计和报告；两个参考框架 `ref_l2`（改进器会改自己）和 `ref_l1`（改进器固定）；问答类基准的评分；LiveCodeBench 的评分；计量代理。

还没做、需要注意的：

- **四个上游框架的适配器只有骨架**（`frameworks/external.py`）。公共部分（在计量代理后面跑上游子进程、冻结目录）已有，但每个框架怎么取出改进器和执行者、怎么跑一轮，需要对着上游代码写。运行到这里会报 `NotWired` 并说明要接什么。
- **真实数据和真实模型都还没跑过**。`prepare()` 里的数据集 id 和字段名、Anthropic 与 OpenAI 兼容客户端都没有实际调用验证。
- **Polyglot 和 SWE-bench 的评分**交给框架适配器调用上游的 docker 评测脚本，本仓库不重复实现。
- `ref_l2` / `ref_l1` 是协议自带的最小参考实现，不是任何论文的复现，用来验证协议和估方差。

## 目录

```
configs/            模型、协议、套件
rsi_eval/
  budget.py         token / 美元记账与预算
  llm.py            模型客户端（mock、anthropic、openai 兼容）
  core.py           上下文、产物存档、评测缓存、配置
  benchmarks/       基准与切分
  frameworks/       适配器接口、参考框架、上游框架骨架
  evaluation.py     并发评测
  runner.py         evolve() 与 improvement_run()
  experiments.py    E1–E11
  stats.py report.py
  proxy.py          给上游子进程用的计量代理
runs/<套件>/        cells/ 每个格子的结果，artifacts/ 冻结的各代产物，ledger.jsonl，report.md
```
