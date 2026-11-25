"""Gödel Agent 适配器。L2：agent 在运行时改写自己的逻辑，包括负责“改”的那部分。

上游：https://github.com/Arvid-pku/Godel_Agent ，固定在 GodelAgent.commit。
状态怎么冻结、每一轮怎么界定、对上游做了哪些替换，见 drivers/godel_driver.py 开头的说明。

映射
- 改进器 = agent 自身逻辑的改动记录：动作日志里“改了 Agent 类、action_* 函数、goal_prompt 或动作列表”的那些条目。
  第 0 代是空列表（即上游原样的 agent）。装到初始状态上 = 新进程里先重放这些条目。
- 执行者 = 完整的动作日志（重放后得到当时的 solver 和它依赖的一切）。第 0 代是空日志（上游该基准的初始 solver）。
- 存档 = agent 的对话历史（上游的 optimize_history）、最新的运行时状态（head）、每一轮结束时的执行者和搜索集分数。

一轮 step = 从 head 恢复运行时，让 agent 做至多 calls_per_step 次 evolve()，然后把轮末的 solver 在搜索集上评一次入档。
上游一次运行最多 100 次 evolve()，默认每轮 10 次，10 轮正好对上。
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..budget import BudgetExhausted
from ..core import Artifact
from ..evaluation import evaluate
from .base import SystemState, register
from .drivers.godel_driver import FAMILIES
from .external import DriverAdapter


@register
class GodelAgent(DriverAdapter):
    name = "godel_agent"
    level = "L2"
    repo_url = "https://github.com/Arvid-pku/Godel_Agent"
    commit = "bbb508796be31c7140cdfc7106efd830a1324242"  # 2025-09-17 "fix typo and change api"
    upstream_api = "https://api.openai.com"
    driver = "rsi_eval.frameworks.drivers.godel_driver"

    # ---- 运行上游 ----
    def _drive(self, command: str, payload: dict, ctx, timeout: float, keep: str | None = None) -> dict:
        """在一份干净的上游 src/ 副本里跑驱动。keep 给了就把子进程的输出留在 runs/<套件>/work/godel_agent/ 下备查。"""
        src = self.ensure_repo(ctx) / "src"
        with tempfile.TemporaryDirectory(prefix="rsi-godel-") as ws:
            shutil.copytree(src, Path(ws) / "src", ignore=shutil.ignore_patterns("__pycache__"))
            (Path(ws) / "src" / "key.env").write_text("rsi-eval-proxy\n")  # 上游导入时会读这个文件；真实密钥只在代理里
            log_to = ctx.settings.run_dir / "work" / self.name / f"{keep}.log" if keep else None
            return self.run_driver(self.driver, command, payload, ctx, Path(ws), timeout=timeout, log_to=log_to)

    def _check(self, bench) -> None:
        if bench.name not in FAMILIES:
            raise KeyError(f"godel_agent: 基准 {bench.name} 没有对应的上游初始 solver（支持：{sorted(FAMILIES)}）")

    # ---- 状态 ----
    def initial(self, ctx, bench):
        self._check(bench)
        base = {"framework": self.name, "commit": self.commit}
        return SystemState(
            improver=ctx.store.put_json("improver", {**base, "entries": []}),
            executor=ctx.store.put_json("executor", {**base, "bench": bench.name, "log": []}),
            archive=ctx.store.put_json("archive", {"entries": [], "history": [], "head": None, "steps": []}),
        )

    # ---- 执行者 ----
    def solve(self, executor, bench, task, ctx, sample):
        spec = executor.load_json()
        # 初始 solver 跟着被评测的基准走：在未见基准上评测时，没改过 solver 的执行者用那个基准的初始 solver
        res = self._drive("solve", {"bench": bench.name, "log": spec["log"], "input": task.input}, ctx, float(self.cfg.get("solve_timeout", 900)))
        return self.as_output(res)

    # ---- 一轮演化 ----
    def step(self, state, ctx, bench, tasks, *, modify_improver):
        self._check(bench)
        arch = state.archive.load_json()
        batch = tasks[: int(self.cfg.get("eval_batch", len(tasks)))]

        def add(executor: Artifact, generation: int) -> None:
            known = {e["executor"]["digest"]: e for e in arch["entries"]}
            if executor.digest in known:  # solver 没变：沿用已有分数，不重复评测
                score = known[executor.digest]["search_score"]
            else:
                score = evaluate(self, executor, bench, batch, ctx).mean
            arch["entries"].append({"executor": executor.to_dict(), "search_score": score, "generation": generation})

        if not arch["entries"]:
            add(state.executor, state.generation)

        head = Artifact.from_dict(arch["head"]) if arch.get("head") else state.executor
        exec_log = head.load_json()["log"]
        have = {e["id"] for e in exec_log}
        # 改进器是从别处装过来的（同起点对照）：它的改动先于执行者的日志重放
        log = [e for e in state.improver.load_json()["entries"] if e["id"] not in have] + exec_log

        gen = state.generation + 1
        out = self._drive("step", {
            "bench": bench.name, "log": log, "history": arch["history"], "modify_improver": bool(modify_improver),
            "tasks": [{"id": t.id, "input": t.input, "target": t.target, "meta": t.meta} for t in tasks],
            "max_calls": int(self.cfg.get("calls_per_step", 10)), "feedback_batch": int(self.cfg.get("feedback_batch", 20)),
            "seed": f"{ctx.seed}|{gen}", "workers": int(ctx.settings.suite.get("max_workers", ctx.protocol.get("max_workers", 8))),
        }, ctx, float(self.cfg.get("step_timeout", 7200)), keep=f"seed{ctx.seed}-gen{gen}-{state.improver.digest[:6]}")

        info = {"generation": gen, "calls": out.get("calls", 0), "exit": out.get("exit"), "feedback_scores": out.get("evals", []), "llm": out["_llm"]}
        improver = state.improver
        if "log" not in out:  # 子进程超时或崩溃：这一轮什么都没改，但算一轮
            info["error"] = str(out.get("error"))[-500:]
            new_head = head
        else:
            new_log = out["log"]
            new_head = ctx.store.put_json("executor", {"framework": self.name, "commit": self.commit, "bench": bench.name,
                                                       "log": new_log, "solver_source": out.get("solver_source", "")})
            arch["history"] = out.get("history", [])
            entries = [e for e in new_log if e.get("improver")]
            info["improver_changed"] = len(entries) != len(state.improver.load_json()["entries"])
            if modify_improver and info["improver_changed"]:
                improver = ctx.store.put_json("improver", {"framework": self.name, "commit": self.commit, "entries": entries,
                                                           "sources": out.get("improver_sources", {})})
        arch["head"] = new_head.to_dict()
        add(new_head, gen)
        arch["steps"].append(info)

        best = max(arch["entries"], key=lambda e: e["search_score"])
        return SystemState(improver, Artifact.from_dict(best["executor"]), ctx.store.put_json("archive", arch), gen)

    # ---- E10：改进器改改进器 ----
    def improve_improver(self, improver, ctx, bench, tasks):
        """Gödel Agent 的自我修改就发生在它的普通演化轮里：从初始执行者出发，带着给定的改进器放开自我修改跑几轮，
        取结束时的改进器。有预算就跑到预算用完，没有就跑 meta_rounds 轮。"""
        state = self.with_improver(self.initial(ctx, bench), improver)
        try:
            for _ in range(int(self.cfg.get("meta_rounds", 2)) if ctx.meter.remaining() is None else 10**6):
                state = self.step(state, ctx, bench, tasks, modify_improver=True)
        except BudgetExhausted:
            pass
        return state.improver
