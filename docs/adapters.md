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

## 已接入：ADAS 和 Gödel Agent

两者的上游代码都只会说 OpenAI Chat Completions，并且把模型型号写死在代码里。接入方式相同：

```
适配器（本进程）──run_driver──▶ 驱动脚本（子进程，导入上游代码）──▶ 计量代理 /_role/<角色>/v1 ──▶ 模型
```

- `rsi-eval vendor fetch` 把上游拉到 `vendor/<名字>/` 并检出适配器里固定的 commit；运行时 HEAD 不对或文件被改过就拒绝。
- 驱动脚本（`rsi_eval/frameworks/drivers/`）导入上游模块后只替换内存里的对象（模型客户端、评测题目），不改上游文件。
- 代理的 `/_role/executor/v1`、`/_role/improver/v1` 两个入口决定这次调用记到谁头上，并把型号强制换成该角色配置的型号。
- 子进程的环境变量是白名单，真实密钥只在代理进程里。
- 模型拒答会在流水里标 `refusal: true`；拒答率过半（`abort_refusal_rate`，满 `refusal_min_calls` 次后生效）适配器抛 `UpstreamRefused` 中止。

### ADAS（L1，对照）

| 协议概念 | 上游里的东西 |
|---|---|
| 改进器 | 该领域的元 agent 提示词文件 `<领域>_prompt.py`（固定） |
| 执行者 | 一个 agent：`{name, thought, code}`，`code` 是 `def forward(self, taskInfo)`。第 0 代是种子库里的 Chain-of-Thought |
| 存档 | 上游的 archive：种子库 7 个 agent + 每轮新生成的，各带搜索集分数 |
| 一轮 | 上游 `search()` 循环体一次：首轮先评种子库 → 元 agent 提案 + 两次反思 → 搜索集评测 → 报错或全错时 debug，最多 3 次 → 入档 |

基准到上游领域目录的对应：`drop→_drop`、`mgsm→_mgsm`、`mmlu→_mmlu`、`gpqa→_gpqa`，未见任务 `gsm8k`、`gsm_hard` 用 `_mgsm`（上游的迁移实验也这么做）。
可调：`frameworks_cfg.adas.{eval_batch, debug_max, solve_timeout, propose_timeout, domains}`。

### Gödel Agent（L2，试点）

上游是一个进程里 `evolve()` 和 `execute_action()` 互相递归、用 monkey patching 改自己，状态只在内存里。

| 协议概念 | 上游里的东西 | 怎么冻结 |
|---|---|---|
| 改进器 | agent 自身的逻辑：`Agent` 类的方法、`action_*` 函数、`goal_prompt`、动作列表 | 动作日志里改动了这些对象的条目。第 0 代是空列表 |
| 执行者 | `solver` 以及它依赖的运行时对象 | 完整的动作日志；新进程里按顺序重放 |
| 存档 | 对话历史 `optimize_history`、最新运行时状态、每轮末的执行者和分数 | json |
| 一轮 | 至多 `calls_per_step`（默认 10）次 `evolve()`，然后把轮末的 solver 在搜索集上评一次 | |

- **动作日志**：每次 `action_adjust_logic` / `action_run_code` 的参数，连同期间发生的模型调用的应答一起记下。重放时用记下的应答，不花钱、结果一致。
- **是否改了改进器**：动作前后对比上面那组对象。`modify_improver=False` 时这类改动被撤销，并向 agent 返回一条说明。
- **角色**：带 `tools` 的调用（`evolve()` 自己那次）算改进器，其余全部算执行者。agent 把 solver 里的 `model` 改成更强的型号不起作用，提示词里也如实告诉了它。
- **初始 solver**：取上游 `task_<基准>.py` 里对应基准的那个（上游是手工改 `agent_module.py` 来换基准的）。
- **评测反馈**：agent 调 `action_evaluate_on_task` 时，从协议传进来的搜索集里抽 `feedback_batch`（默认 20）题，反馈格式照上游；上游“验证集过线就测测试集”的逻辑去掉了。
- E10 `improve_improver`：带着给定的改进器放开自我修改跑几轮，取结束时的改进器。

已知的不完全等价之处：重放依赖动作本身可重复（读时钟、随机数、写到工作目录之外的文件的动作重放结果可能不同）；
上游的 `openai==1.2.0` 太旧，这里用当前版本的 openai 包；`run_code` 里的 bash 动作重放时会再执行一遍。

### 把 OpenAI 格式的请求发给 Claude

角色对应的模型是 `provider: anthropic` 时，代理把请求转到 Anthropic 的 OpenAI 兼容端点，并抹平三处差异：去掉 `temperature`（新模型不接受）；
去掉 `response_format: json_object`，补一句“只输出一个 JSON 对象”，并把回复里的代码围栏、单引号字典修成合法 JSON；`n > 1` 拆成多次请求再合并。
**但 Claude 5.5 会拒答这两个框架的大部分提示词**（见 README“目前的状态”），所以实际要跑得用 `provider: openai_compat` 的模型。

## 还没接入的两个框架

- **HyperAgents**（L2，主力）：元 agent 和任务 agent 在同一程序里，按文件路径拆成改进器和执行者；
  `modify_improver=False` 直接用上游自带的“去掉自改进”消融；E10 对应只让元 agent 改元 agent。
- **HGM**（L2）：编码 agent 改自己，改进器和执行者是同一份代码的不同快照；同起点对照时用第 k 代快照去改第 0 代快照。
