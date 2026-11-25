"""协议的两个基本动作：一次演化(evolve)和一次改进运行(improvement_run)。

11 项实验都是这两个动作加上不同的起点、预算和开关拼出来的。
"""
from __future__ import annotations

from .benchmarks import Benchmark
from .budget import BudgetExhausted
from .core import Artifact, Ctx
from .evaluation import evaluate
from .frameworks.base import FrameworkAdapter


def report_score(adapter: FrameworkAdapter, executor: Artifact, bench: Benchmark, ctx: Ctx, sample: int = 0) -> float:
    """报告集分数。只有这里会读报告集。"""
    rctx = ctx.with_(phase="report")
    return evaluate(adapter, executor, bench, bench.tasks("report", "report"), rctx, sample=sample).mean


def evolve(
    adapter: FrameworkAdapter,
    bench: Benchmark,
    ctx: Ctx,
    *,
    rounds: int | None = None,
    budget: float | None = None,
    modify_improver: bool = True,
    improver: Artifact | None = None,
    tags: dict | None = None,
) -> dict:
    """从第 0 代出发演化，直到跑满 rounds 轮或预算(只限 evolve 阶段)用完。

    每一代的改进器、最优执行者、存档都冻结存盘。结束后在选择集上从候选里挑出最终系统。
    improver 不为空时用它替换初始改进器（冻结改进器的同起点运行）。
    返回的全是可序列化的内容，实验层直接落盘。
    """
    if rounds is None and budget is None:
        raise ValueError("evolve needs rounds or budget")
    meter = ctx.meter.child(limit=budget, **(tags or {}))
    ectx = ctx.with_(phase="evolve", meter=meter)
    search = bench.tasks("search", "evolve")

    state = adapter.initial(ectx, bench)
    initial_executor = state.executor
    if improver is not None:
        state = adapter.with_improver(state, improver)

    def snap(s):
        return {**s.to_dict(), "tokens_cum": meter.totals(("evolve",)).tokens, "usd_cum": meter.totals(("evolve",)).usd}

    generations = [snap(state)]
    stopped = "rounds"
    while rounds is None or state.generation < rounds:
        try:
            state = adapter.step(state, ectx, bench, search, modify_improver=modify_improver)
        except BudgetExhausted:
            stopped = "budget"  # 半途而废的这一轮作废，但花掉的 token 照记
            break
        generations.append(snap(state))

    # 选择：存档里搜索集分数最好的 k 个 + 初始执行者，在选择集上比
    sctx = ctx.with_(phase="select", meter=meter)
    k = int(ctx.protocol.get("select_top_k", 3))
    cands = {a.digest: a for a in [initial_executor, *adapter.candidates(state, k)]}
    select_tasks = bench.tasks("select", "select")
    select_scores = {d: evaluate(adapter, a, bench, select_tasks, sctx).mean for d, a in cands.items()}
    best = max(cands, key=lambda d: (select_scores[d], d == initial_executor.digest))

    return {
        "initial_executor": initial_executor.to_dict(),
        "selected_executor": cands[best].to_dict(),
        "select_scores": select_scores,
        "generations": generations,
        "final_improver": state.improver.to_dict(),
        "rounds_done": state.generation,
        "stopped": stopped,
        "cost": meter.summary(),
        "evolve_tokens": meter.totals(("evolve",)).tokens,
        "total_tokens": meter.totals(("evolve", "select")).tokens,
        "total_usd": meter.totals(("evolve", "select")).usd,
    }


def improvement_run(adapter: FrameworkAdapter, improver: Artifact, bench: Benchmark, ctx: Ctx, budget: float, *, tags: dict | None = None) -> dict:
    """一次“改进运行”：冻结的改进器从初始执行者、空存档出发，用固定预算改一轮，
    在留出集（该基准的报告集）上看执行者相对起点涨了多少。改进器全程不许改自己。"""
    ev = evolve(adapter, bench, ctx, budget=budget, modify_improver=False, improver=improver, tags=tags)
    init, sel = Artifact.from_dict(ev["initial_executor"]), Artifact.from_dict(ev["selected_executor"])
    base = report_score(adapter, init, bench, ctx)
    score = report_score(adapter, sel, bench, ctx)
    out = {
        "improver": improver.digest, "budget": budget, "base_score": base, "score": score, "gain": score - base,
        "rounds_done": ev["rounds_done"], "evolve_tokens": ev["evolve_tokens"], "total_tokens": ev["total_tokens"],
        "total_usd": ev["total_usd"], "selected_executor": ev["selected_executor"],
    }
    if ctx.protocol.get("gain_curve", True):
        # 增益随预算变化的曲线：每一代的当前最优执行者在留出集上的分数（有缓存，重复的不多花钱）
        out["curve"] = [
            {"tokens": g["tokens_cum"], "gain": report_score(adapter, Artifact.from_dict(g["executor"]), bench, ctx) - base}
            for g in ev["generations"]
        ]
    return out
