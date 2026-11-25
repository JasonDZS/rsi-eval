# 接入一个上游框架

评测协议把每个自改进框架看成三样可以冻结、导出、重新组合的东西：

| 概念 | 含义 | 例子 |
|---|---|---|
| 改进器 improver | 负责“改”的部分。L2 框架里它自己也会被改 | HyperAgents 的元 agent 代码；ADAS 的固定元提示词 |
| 执行者 executor | 负责答题或写代码的部分 | ADAS 存档里的 `forward` 函数；HGM 的编码 agent 快照 |
| 存档 archive | 累积的候选、分数、历史 | ADAS 的 archive json |

三样都以 `Artifact` 的形式存在 `runs/<套件>/artifacts/` 下，按内容寻址。小的用 `ctx.store.put_json()`，
代码快照或 checkpoint 用 `ctx.store.put_dir()`。

## 要实现的方法

```python
class MyFramework(FrameworkAdapter):
    name = "my_framework"
    level = "L2"            # 改进器固定的框架写 "L1"

    def initial(self, ctx, bench) -> SystemState: ...
    def step(self, state, ctx, bench, tasks, *, modify_improver) -> SystemState: ...
    def solve(self, executor, bench, task, ctx, sample) -> str: ...
    def candidates(self, state, k) -> list[Artifact]: ...
    # 可选
    def improve_improver(self, improver, ctx, bench, tasks) -> Artifact: ...   # E10
    def run_task(self, executor, bench, task, ctx, sample) -> TaskResult: ...   # Polyglot / SWE-bench 用上游评测脚本
```

必须满足的四条约定，协议的结论靠它们成立：

1. **`initial()` 每次返回同一个第 0 代**。所有条件都从它出发。
2. **`modify_improver=False` 时改进器一个字节都不能变**。冻结改进器的改进运行和“只累积经验”对照都靠这个开关。
   运行器会把任意一代的改进器通过 `with_improver()` 装到初始状态上，所以改进器必须能脱离它演化时的存档单独工作。
3. **`step()` 只能看传进来的 `tasks`**（搜索集）。不要自己去读数据文件。
4. **所有模型调用都要入账**。进程内调用用 `ctx.llm("improver")` / `ctx.llm("executor")`；上游代码在子进程里跑时用
   `ExternalRepoAdapter.run()`，它会起一个计量代理并把 `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` 指过去。
   预算用完时会抛 `BudgetExhausted`，不要捕获，让它冒到运行器。

## 上游子进程的做法

```python
class HyperAgents(ExternalRepoAdapter):
    def step(self, state, ctx, bench, tasks, *, modify_improver):
        work = self.workspace(ctx, f"seed{ctx.seed}-gen{state.generation}")   # 干净的上游代码副本
        # 1. 把 state.improver / state.executor / state.archive 的目录内容铺到 work 里上游期望的位置
        # 2. 把 tasks 的 id 写成上游能读的任务清单
        # 3. self.run([...上游的单轮命令...], ctx, cwd=work)   # 模型调用自动入账
        # 4. 把新的改进器、执行者、存档目录用 self.freeze() 冻结，组装成新的 SystemState
```

接好之后先用 `rsi-eval run --suite ... --only E1,E2` 在最小配置上跑一遍，核对 `ledger.jsonl` 里的 token 数
和上游自己报告的用量一致，再放开其余实验。

## 四个入选框架的接入要点

- **ADAS**（L1，对照）：改进器是固定的元 agent 提示词；一轮 = 元 agent 生成一个新 agent 并在搜索集上评测入档。
  上游按基准分目录，沿用它的验证集做搜索集时注意和本仓库的切分对齐。
- **Gödel Agent**（L2，试点）：agent 在运行时改写自己的逻辑，要把运行时状态落盘成目录才能冻结每一代，这是它导出难度为“中”的原因。
- **HyperAgents**（L2，主力）：元 agent 和任务 agent 在同一程序里，按文件路径拆成改进器和执行者；
  `modify_improver=False` 直接用上游自带的“去掉自改进”消融；E10 对应只让元 agent 改元 agent。
- **HGM**（L2）：编码 agent 改自己，改进器和执行者是同一份代码的不同快照；同起点对照时用第 k 代快照去改第 0 代快照。
