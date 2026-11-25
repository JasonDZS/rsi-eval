"""参考适配器：协议自带的两个最小自改进框架。

它们不是任何论文的复现，用途是：
1. 离线（mock 模型）验证整条协议、记账、统计流程；
2. 用真实模型在问答基准上做最便宜的端到端试跑；
3. 给接入上游框架的人当样板。

- ref_l2：改进器是一份规格（元提示词 + 指导原则 + 参数），每轮既产出新的执行者，
  也可以重写自己的规格（改进器本体被修改）。
- ref_l1：同一套逻辑，但改进器永远不变，只累积存档（对应“只累积经验”对照）。

执行者也是一份声明式规格（系统提示词 + 指令列表 + 采样次数），由固定解释器执行，
不执行模型生成的代码。
"""
from __future__ import annotations

import json

from ..benchmarks import Benchmark, Task
from ..core import Artifact, Ctx
from ..evaluation import evaluate
from ..llm import extract_json
from .base import FrameworkAdapter, SystemState, register

INITIAL_EXECUTOR = {"system": "You are a careful problem solver.", "instructions": [], "samples": 1}
INITIAL_IMPROVER = {
    "meta_prompt": "You improve an AI agent's instructions so that it scores higher on a benchmark.",
    "guidance": [],
    "eval_batch": 20,
    "show_failures": 3,
    "show_archive": 3,
}


def _clean_executor(obj: dict | None, fallback: dict) -> dict | None:
    if not isinstance(obj, dict):
        return None
    ins = obj.get("instructions", fallback.get("instructions", []))
    if not isinstance(ins, list):
        return None
    try:
        samples = max(1, min(5, int(obj.get("samples", 1))))
    except (TypeError, ValueError):
        samples = 1
    return {"system": str(obj.get("system", fallback.get("system", "")))[:4000],
            "instructions": [str(x)[:500] for x in ins][:30], "samples": samples}


def _clean_improver(obj: dict | None, cur: dict) -> dict | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("guidance", []), list):
        return None

    def _int(key, lo, hi):
        try:
            return max(lo, min(hi, int(obj.get(key, cur[key]))))
        except (TypeError, ValueError):
            return cur[key]

    return {
        "meta_prompt": str(obj.get("meta_prompt", cur["meta_prompt"]))[:4000],
        "guidance": [str(x)[:500] for x in obj.get("guidance", [])][:30],
        "eval_batch": cur["eval_batch"],  # 评测批大小不让改进器自己改，保证各代可比
        "show_failures": _int("show_failures", 0, 10),
        "show_archive": _int("show_archive", 0, 10),
    }


@register
class RefL2(FrameworkAdapter):
    name = "ref_l2"
    level = "L2"

    # ---- 状态 ----
    def initial(self, ctx, bench):
        imp = {**INITIAL_IMPROVER, **self.cfg.get("improver", {})}
        return SystemState(
            improver=ctx.store.put_json("improver", imp),
            executor=ctx.store.put_json("executor", {**INITIAL_EXECUTOR, **self.cfg.get("executor", {})}),
            archive=ctx.store.put_json("archive", {"entries": [], "history": []}),
        )

    def candidates(self, state, k):
        entries = sorted(state.archive.load_json()["entries"], key=lambda e: -e["search_score"])
        seen, out = set(), []
        for e in entries:
            if e["executor"]["digest"] not in seen:
                seen.add(e["executor"]["digest"])
                out.append(Artifact.from_dict(e["executor"]))
        return out[:k]

    # ---- 执行者 ----
    def solve(self, executor, bench, task, ctx, sample):
        spec = executor.load_json()
        prompt = "\n".join(["Instructions:"] + [f"- {i}" for i in spec["instructions"]] + ["", task.input, "", bench.instructions])
        llm = ctx.llm("executor")
        outs = [llm.complete(prompt, system=spec["system"], sample=sample * 100 + i).text for i in range(spec.get("samples", 1))]
        return outs[0] if len(outs) == 1 else bench.vote(outs)

    # ---- 一轮演化 ----
    def step(self, state, ctx, bench, tasks, *, modify_improver):
        imp = state.improver.load_json()
        arch = state.archive.load_json()
        batch = tasks[: imp["eval_batch"]]

        def add(executor: Artifact, gen: int) -> dict:
            r = evaluate(self, executor, bench, batch, ctx)
            failures = [tid for tid, s in r.scores.items() if s < 1.0]
            e = {"executor": executor.to_dict(), "search_score": r.mean, "generation": gen,
                 "failures": [{"id": tid, "output": r.outputs[tid][-300:]} for tid in failures[:10]]}
            arch["entries"].append(e)
            return e

        if not arch["entries"]:
            add(state.executor, state.generation)
        parent = max(arch["entries"], key=lambda e: e["search_score"])

        # 1) 改进器产出新的执行者候选
        cand_spec = self._propose_executor(imp, arch, parent, bench, batch, ctx)
        gen = state.generation + 1
        if cand_spec is not None:
            child = add(ctx.store.put_json("executor", cand_spec), gen)
            arch["history"].append({"generation": gen, "parent_score": parent["search_score"], "child_score": child["search_score"],
                                    "added": [i for i in cand_spec["instructions"] if i not in Artifact.from_dict(parent["executor"]).load_json()["instructions"]]})
        else:
            arch["history"].append({"generation": gen, "parent_score": parent["search_score"], "child_score": None, "added": []})

        # 2) L2：改进器重写自己
        improver = state.improver
        if modify_improver and self.level == "L2":
            new_imp = self._propose_improver(imp, arch["history"], ctx)
            if new_imp is not None:
                improver = ctx.store.put_json("improver", new_imp)

        best = max(arch["entries"], key=lambda e: e["search_score"])
        return SystemState(improver, Artifact.from_dict(best["executor"]), ctx.store.put_json("archive", arch), gen)

    def _propose_executor(self, imp, arch, parent, bench, batch, ctx) -> dict | None:
        by_id = {t.id: t for t in batch}
        top = sorted(arch["entries"], key=lambda e: -e["search_score"])[: imp["show_archive"]]
        parent_spec = Artifact.from_dict(parent["executor"]).load_json()
        parts = [
            imp["meta_prompt"],
            "<guidance>\n" + "\n".join(f"- {g}" for g in imp["guidance"]) + "\n</guidance>",
            f"## Output format the agent must follow\n{bench.instructions}",
            "## Archive (best first)\n" + "\n".join(
                json.dumps({"score": round(e["search_score"], 3), "spec": Artifact.from_dict(e["executor"]).load_json()}) for e in top),
            f"## Current best agent (score {parent['search_score']:.3f})\n<current_executor>{json.dumps(parent_spec)}</current_executor>",
            "## Examples it got wrong\n" + "\n\n".join(
                f"Question: {by_id[f['id']].input[:1500]}\nExpected: {by_id[f['id']].target}\nAgent output (tail): {f['output']}"
                for f in parent["failures"][: imp["show_failures"]] if f["id"] in by_id),
            "## Your job\nPropose a NEW agent spec that will score higher. Reply with one JSON object with keys "
            '"system" (string), "instructions" (list of strings) and "samples" (integer 1-5, number of votes per question). <<EXECUTOR_SPEC>>',
        ]
        text = ctx.llm("improver").complete("\n\n".join(parts)).text
        return _clean_executor(extract_json(text), parent_spec)

    def _propose_improver(self, imp, history, ctx) -> dict | None:
        parts = [
            "You are a self-improving optimizer. Below is your own current configuration and the record of what your recent proposals achieved.",
            f"<current_improver>{json.dumps(imp)}</current_improver>",
            "## History of your proposals (parent score -> child score, instructions you added)\n" + "\n".join(json.dumps(h) for h in history[-8:]),
            "## Your job\nRewrite your own configuration so that your future proposals improve agents faster. Reply with one JSON object with keys "
            '"meta_prompt" (string), "guidance" (list of strings), "show_failures" (int 0-10), "show_archive" (int 0-10). <<IMPROVER_SPEC>>',
        ]
        text = ctx.llm("improver").complete("\n\n".join(parts)).text
        return _clean_improver(extract_json(text), imp)

    # ---- E10：改进器改改进器 ----
    def improve_improver(self, improver, ctx, bench, tasks):
        """从初始执行者出发试跑几轮攒出历史，再让改进器据此重写自己；循环到预算用完为止。"""
        from ..budget import BudgetExhausted

        cur = improver
        state = self.with_improver(self.initial(ctx, bench), cur)
        try:
            while True:
                for _ in range(int(self.cfg.get("meta_trial_rounds", 2))):
                    state = self.step(state, ctx, bench, tasks, modify_improver=False)
                new = self._propose_improver(cur.load_json(), state.archive.load_json()["history"], ctx)
                if new is not None:
                    cur = ctx.store.put_json("improver", new)
                    state = self.with_improver(state, cur)
                if ctx.meter.remaining() is None:  # 没设预算就只做一轮
                    break
        except BudgetExhausted:
            pass
        return cur


@register
class RefL1(RefL2):
    """改进器固定为第 0 代，只累积存档。"""

    name = "ref_l1"
    level = "L1"

    def improve_improver(self, improver, ctx, bench, tasks):
        from .base import NotSupported

        raise NotSupported("ref_l1 has a fixed improver")


