"""在一组题上评测一个执行者：并发、缓存、记账。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .benchmarks import Benchmark, Task
from .budget import Usage
from .core import Artifact, Ctx


#: 整道题超时是线路问题（请求挂死），不是执行者答错：重试几次，实在不行才记 0 分
INFRA_ERROR = "[upstream-error] timeout after"
INFRA_RETRIES = 2


@dataclass
class EvalResult:
    mean: float
    scores: dict = field(default_factory=dict)  # task_id -> score
    outputs: dict = field(default_factory=dict)  # task_id -> 原始输出
    tokens: int = 0


def _one(adapter, executor: Artifact, bench: Benchmark, task: Task, ctx: Ctx, sample: int) -> dict:
    model = ctx.roles["executor"]
    key = (adapter.name, executor.digest, model, bench.name, task.id, sample)
    hit = ctx.cache.get(key)
    if hit is not None:
        # 命中缓存：不再真实调用，但按原始消耗记账，保证各条件预算口径一致
        ctx.meter.check(ctx.phase)
        ctx.meter.charge(ctx.phase, "executor", model, Usage(hit["input_tokens"], hit["output_tokens"]), hit["usd"], cached=True)
        return hit
    m = ctx.meter.child()
    for attempt in range(1 + INFRA_RETRIES):
        r = adapter.run_task(executor, bench, task, ctx.with_(meter=m), sample)
        if not r.output.startswith(INFRA_ERROR):
            break
    t = m.totals()
    rec = {"output": r.output, "score": float(r.score), "input_tokens": t.input_tokens, "output_tokens": t.output_tokens, "usd": t.usd}
    if not r.output.startswith(INFRA_ERROR):  # 线路问题不进缓存：续跑时这道题会重新评
        ctx.cache.put(key, rec)
    return rec


def evaluate(adapter, executor: Artifact, bench: Benchmark, tasks: list[Task], ctx: Ctx, sample: int | None = None) -> EvalResult:
    """sample 缺省：报告阶段固定为 0（同一执行者的报告集分数全套件共用一次评测）；
    演化/选择阶段用 ctx.seed，使不同的独立运行互不共享采样。"""
    if sample is None:
        sample = 0 if ctx.phase == "report" else 1000 + ctx.seed
    if not tasks:
        return EvalResult(0.0)
    workers = int(ctx.settings.suite.get("max_workers", ctx.protocol.get("max_workers", 8)))
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [pool.submit(_one, adapter, executor, bench, t, ctx, sample) for t in tasks]
        recs = [f.result() for f in futs]  # 第一个异常（如 BudgetExhausted）直接抛出
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    scores = {t.id: r["score"] for t, r in zip(tasks, recs)}
    outputs = {t.id: r["output"] for t, r in zip(tasks, recs)}
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in recs)
    return EvalResult(sum(scores.values()) / len(scores), scores, outputs, tokens)
