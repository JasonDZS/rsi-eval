"""框架适配器接口。

评测协议把每个自改进框架看成三样可以冻结、导出、重新组合的东西：

- 改进器 improver：负责“改”的那部分（L2 框架里它自己也会被改）
- 执行者 executor：负责答题 / 写代码的那部分
- 存档 archive：累积的候选、分数、历史经验

适配器要做的就是把上游框架映射到这三样，并实现下面几个操作。
运行器(runner.py)用这些操作拼出全部 11 项实验，适配器不需要知道实验逻辑。

适配器内所有模型调用必须走 ctx.llm(role)（或上游子进程走计量代理），否则 token 不会入账。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..benchmarks import Benchmark, Task
from ..core import Artifact, Ctx


class NotSupported(Exception):
    """该框架不支持这个操作（例如 L1 框架没有“改进器改改进器”）。"""


@dataclass
class SystemState:
    improver: Artifact
    executor: Artifact  # 当前最优执行者（按搜索集分数）
    archive: Artifact
    generation: int = 0

    def to_dict(self) -> dict:
        return {"improver": self.improver.to_dict(), "executor": self.executor.to_dict(),
                "archive": self.archive.to_dict(), "generation": self.generation}


@dataclass
class TaskResult:
    output: str
    score: float


class FrameworkAdapter(ABC):
    name: str = ""
    #: "L2" = 改进器本体会被修改；"L1" = 改进器固定、只累积存档
    level: str = "L2"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    # ---- 必须实现 ----
    @abstractmethod
    def initial(self, ctx: Ctx, bench: Benchmark) -> SystemState:
        """第 0 代：初始改进器 + 初始执行者 + 空存档。所有条件都从这里出发。"""

    @abstractmethod
    def step(self, state: SystemState, ctx: Ctx, bench: Benchmark, tasks: list[Task], *, modify_improver: bool) -> SystemState:
        """一轮演化：改进器看搜索集上的表现，产出新的执行者候选并入档。

        modify_improver=True 且框架是 L2 时，这一轮也允许改进器修改自己，返回的
        state.improver 就是下一代改进器。modify_improver=False 时改进器必须保持不变
        （冻结改进器的改进运行、“只累积经验”对照都靠这个开关）。
        只能看传进来的 tasks（搜索集）；预算用完时 ctx.llm 会抛 BudgetExhausted，直接让它冒出去。
        """

    @abstractmethod
    def solve(self, executor: Artifact, bench: Benchmark, task: Task, ctx: Ctx, sample: int) -> str:
        """用执行者答一道题，返回原始输出。模型用 ctx.llm("executor")。"""

    @abstractmethod
    def candidates(self, state: SystemState, k: int) -> list[Artifact]:
        """存档里按搜索集分数最好的 k 个执行者，供运行器在选择集上挑。"""

    # ---- 可选 ----
    def with_improver(self, state: SystemState, improver: Artifact) -> SystemState:
        """把一个冻结的改进器装到给定状态上（同起点对照用）。"""
        return SystemState(improver, state.executor, state.archive, state.generation)

    def improve_improver(self, improver: Artifact, ctx: Ctx, bench: Benchmark, tasks: list[Task]) -> Artifact:
        """让改进器改一轮改进器，返回下一代（E10）。预算由 ctx 控制。"""
        raise NotSupported(f"{self.name} does not support improver-improves-improver")

    def run_task(self, executor: Artifact, bench: Benchmark, task: Task, ctx: Ctx, sample: int) -> TaskResult:
        """答题并评分。adapter_scored 的基准（Polyglot、SWE-bench）在子类里重写，调用上游评测脚本。"""
        if bench.adapter_scored:
            raise NotSupported(f"{self.name} must override run_task() for adapter-scored benchmark {bench.name}")
        out = self.solve(executor, bench, task, ctx, sample)
        return TaskResult(out, bench.score(task, out))


ADAPTERS: dict[str, type[FrameworkAdapter]] = {}


def register(cls):
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str, cfg: dict | None = None) -> FrameworkAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown framework '{name}' (have: {sorted(ADAPTERS)})")
    return ADAPTERS[name](cfg)
