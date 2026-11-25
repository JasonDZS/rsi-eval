"""E1–E11 实验编排。

每项实验拆成若干“格子”(cell)：一个格子 = 一个确定的 (实验, 模型配置, 框架, 基准, 种子, 条件)。
格子的结果落在 runs/<套件>/cells/<key>.json，已有就跳过，所以整套实验可以随时中断、续跑，
后面的实验直接复用前面实验的产物（例如 E3–E10 都读 E2 存下来的各代改进器）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from .benchmarks import get_benchmark
from .budget import BudgetExhausted
from .core import Artifact, Ctx, Settings, make_root_ctx
from .evaluation import evaluate
from .frameworks import NotSupported, get_adapter
from .runner import evolve, improvement_run, report_score


class Run:
    def __init__(self, settings: Settings, log=None):
        self.s = settings
        self.suite = settings.suite
        tier = self.suite.get("tier", "minimal")
        self.p = {**settings.protocol["tiers"][tier], **self.suite.get("params", {})}
        self.cells_dir = settings.run_dir / "cells"
        self._ctx: dict[str, Ctx] = {}
        self._bench: dict = {}
        self._adapter: dict = {}
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))

    # ---- 基础设施 ----
    def root(self, cfg: str) -> Ctx:
        if cfg not in self._ctx:
            self._ctx[cfg] = make_root_ctx(self.s, cfg)
        return self._ctx[cfg]

    def bench(self, name: str):
        if name not in self._bench:
            self._bench[name] = get_benchmark(name, self.s.protocol, self.s.root / "data")
        return self._bench[name]

    def adapter(self, name: str):
        if name not in self._adapter:
            self._adapter[name] = get_adapter(name, self.suite.get("frameworks_cfg", {}).get(name))
        return self._adapter[name]

    def cell(self, key: str, cfg: str, fn) -> dict:
        path = self.cells_dir / (key + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        root = self.root(cfg)
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:6], 16)
        meter = root.meter.child(cell=key)
        t0 = time.time()
        out = fn(root.with_(seed=seed, meter=meter))
        out = {"key": key, **out, "_cost": meter.summary(), "_seconds": round(time.time() - t0, 2),
               "_protocol": self.s.protocol_hash(), "_models": root.roles}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        self.log(f"  done {key}  tokens={out['_cost']['total']['tokens']}")
        return out

    def seeds(self) -> range:
        return range(int(self.p["repeats"]))

    def gen_indices(self, ev: dict, which=None) -> dict[str, int]:
        """要对比哪几代改进器。first/mid/last，或 all（相邻代对比用）。"""
        n = len(ev["generations"])
        which = which or self.p.get("generations", ["first", "mid", "last"])
        if which == "all":
            stride = int(self.p.get("adjacent_stride", 1))
            idx = sorted(set(list(range(0, n, stride)) + [n - 1]))
            return {f"gen{i}": i for i in idx}
        m = {"first": 0, "mid": (n - 1) // 2, "last": n - 1}
        return {w: m[w] for w in which}

    def improve_budget(self, ev: dict) -> float:
        """一次改进运行的 token 预算。auto = 源演化运行实际花掉的演化 token。"""
        b = self.p.get("improve_budget_tokens", "auto")
        return float(ev["evolve_tokens"]) if b == "auto" else float(b)

    # ======================================================================
    # E1 基线复现
    # ======================================================================
    def e1(self, cfg, fw, bench, unseen):
        ad = self.adapter(fw)
        for b in [bench, *unseen]:
            for rep in self.seeds():
                def fn(ctx, b=b, rep=rep):
                    bm = self.bench(b)
                    ex = ad.initial(ctx.with_(phase="evolve"), bm).executor
                    return {"exp": "E1", "framework": fw, "benchmark": b, "rep": rep, "score": report_score(ad, ex, bm, ctx, sample=rep)}
                self.cell(f"E1/{cfg}/{fw}/{b}/rep{rep}", cfg, fn)

    # ======================================================================
    # E2 自改进演化（保存每一代）
    # ======================================================================
    def e2_run(self, cfg, fw, bench, seed) -> dict:
        ad, bm = self.adapter(fw), self.bench(bench)

        def fn(ctx):
            ev = evolve(ad, bm, ctx, rounds=int(self.p["rounds"]), modify_improver=True)
            base = report_score(ad, Artifact.from_dict(ev["initial_executor"]), bm, ctx)
            score = report_score(ad, Artifact.from_dict(ev["selected_executor"]), bm, ctx)
            return {"exp": "E2", "framework": fw, "benchmark": bench, "seed": seed, "base_score": base, "score": score, "gain": score - base, **ev}
        return self.cell(f"E2/{cfg}/{fw}/{bench}/seed{seed}", cfg, fn)

    def e2(self, cfg, fw, bench, unseen):
        for s in self.seeds():
            self.e2_run(cfg, fw, bench, s)

    # ======================================================================
    # E3 同预算重复采样对照
    # ======================================================================
    def e3(self, cfg, fw, bench, unseen):
        ad, bm = self.adapter(fw), self.bench(bench)
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)

            def fn(ctx, ev=ev):
                rctx = ctx.with_(phase="report")
                tasks = bm.tasks("report", "report")
                init = Artifact.from_dict(ev["initial_executor"])
                first = evaluate(ad, init, bm, tasks, rctx, sample=0)
                cost1 = max(1, first.tokens)
                budget = ev["total_tokens"]  # 自改进一共花掉的演化 + 选择 token
                n = max(1, min(int(self.p.get("max_resamples", 16)), 1 + int(budget // cost1)))
                runs = [first] + [evaluate(ad, init, bm, tasks, rctx, sample=i) for i in range(1, n)]
                oracle = sum(max(r.scores[t.id] for r in runs) for t in tasks) / len(tasks)
                if bm.adapter_scored:
                    vote, mode = None, "oracle"  # 可执行测试评分的题没法投票，用“任一次通过”作上界，对自改进更严格
                else:
                    vote = sum(bm.score(t, bm.vote([r.outputs[t.id] for r in runs])) for t in tasks) / len(tasks)
                    mode = "vote"
                return {"exp": "E3", "framework": fw, "benchmark": bench, "seed": s, "n_samples": n, "budget_tokens": budget,
                        "tokens_used": n * cost1, "budget_used_fraction": min(1.0, n * cost1 / max(1, budget + cost1)),
                        "score_single": first.mean, "score_vote": vote, "score_oracle": oracle,
                        "control_mode": mode, "score": oracle if vote is None else vote}
            self.cell(f"E3/{cfg}/{fw}/{bench}/seed{s}", cfg, fn)

    # ======================================================================
    # E4 未见 Benchmark 迁移
    # ======================================================================
    def e4(self, cfg, fw, bench, unseen):
        ad = self.adapter(fw)
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)
            for ub in unseen:
                def fn(ctx, ev=ev, ub=ub):
                    ubm = self.bench(ub)
                    base = report_score(ad, Artifact.from_dict(ev["initial_executor"]), ubm, ctx)
                    score = report_score(ad, Artifact.from_dict(ev["selected_executor"]), ubm, ctx)
                    return {"exp": "E4", "framework": fw, "benchmark": bench, "unseen": ub, "seed": s, "base_score": base, "score": score,
                            "gain": score - base, "target_gain": ev["gain"],
                            "retention": (score - base) / ev["gain"] if ev["gain"] > 0 else None}
                self.cell(f"E4/{cfg}/{fw}/{bench}/seed{s}/{ub}", cfg, fn)

    # ======================================================================
    # E5 换模型迁移
    # ======================================================================
    def e5(self, cfg, fw, bench, unseen):
        ad, bm = self.adapter(fw), self.bench(bench)
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)

            def fn(ctx, ev=ev):
                if ctx.roles["transfer"] in (ctx.roles["executor"], ctx.roles["improver"]):
                    raise ValueError("transfer model must differ from the models used during evolution")
                tctx = ctx.with_(roles={**ctx.roles, "executor": ctx.roles["transfer"]})
                base = report_score(ad, Artifact.from_dict(ev["initial_executor"]), bm, tctx)
                score = report_score(ad, Artifact.from_dict(ev["selected_executor"]), bm, tctx)
                return {"exp": "E5", "framework": fw, "benchmark": bench, "seed": s, "transfer_model": ctx.roles["transfer"],
                        "base_score": base, "score": score, "gain": score - base, "target_gain": ev["gain"],
                        "retention": (score - base) / ev["gain"] if ev["gain"] > 0 else None}
            self.cell(f"E5/{cfg}/{fw}/{bench}/seed{s}", cfg, fn)

    # ======================================================================
    # 改进运行（E6 / E8 / E9 共用）
    # ======================================================================
    def imp_run(self, cfg, fw, bench, seed, gen: int, ub: str, budget: float, exp: str) -> dict:
        ad = self.adapter(fw)
        ev = self.e2_run(cfg, fw, bench, seed)

        def fn(ctx):
            imp = Artifact.from_dict(ev["generations"][gen]["improver"])
            r = improvement_run(ad, imp, self.bench(ub), ctx, budget)
            return {"exp": "IMP", "framework": fw, "benchmark": bench, "unseen": ub, "seed": seed, "gen": gen,
                    "gen_cost_tokens": ev["generations"][gen]["tokens_cum"], **r}
        return self.cell(f"IMP/{cfg}/{fw}/{bench}/seed{seed}/{ub}/gen{gen}/B{int(budget)}", cfg, fn)

    def improve_benches(self, bench, unseen) -> list[str]:
        """改进运行在“演化期间没见过的任务”上做：默认用未见基准，没有就退回演化基准。"""
        return self.p.get("improve_benchmarks") or (unseen[:1] if unseen else [bench])

    # ======================================================================
    # E6 新旧改进器对照（含“第 0 代同预算多次重启取最好”对照）
    # ======================================================================
    def e6(self, cfg, fw, bench, unseen):
        ad = self.adapter(fw)
        if ad.level != "L2":
            return self.log(f"  skip E6 for {fw}: improver is fixed (L1)")
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)
            B = self.improve_budget(ev)
            for ub in self.improve_benches(bench, unseen):
                for label, g in self.gen_indices(ev).items():
                    self.imp_run(cfg, fw, bench, s, g, ub, B, "E6")

                def fn(ctx, ev=ev, ub=ub, B=B):
                    ubm = self.bench(ub)
                    imp0 = Artifact.from_dict(ev["generations"][0]["improver"])
                    n = int(self.p.get("restarts", 3))
                    best = None
                    for i in range(n):
                        r = evolve(ad, ubm, ctx.with_(seed=ctx.seed + i + 1), budget=B / n, modify_improver=False, improver=imp0)
                        sel = r["select_scores"][r["selected_executor"]["digest"]]
                        if best is None or sel > best[0]:
                            best = (sel, r)
                    base = report_score(ad, Artifact.from_dict(best[1]["initial_executor"]), ubm, ctx)
                    score = report_score(ad, Artifact.from_dict(best[1]["selected_executor"]), ubm, ctx)
                    return {"exp": "E6ctl", "framework": fw, "benchmark": bench, "unseen": ub, "seed": s, "restarts": n, "budget": B,
                            "base_score": base, "score": score, "gain": score - base}
                self.cell(f"E6ctl/{cfg}/{fw}/{bench}/seed{s}/{ub}", cfg, fn)

    # ======================================================================
    # E7 只累积经验对照（改进器固定为第 0 代，保留存档和历史，预算与 E2 对齐）
    # ======================================================================
    def e7(self, cfg, fw, bench, unseen):
        control = self.suite.get("e7_control", {}).get(fw, fw)  # 试点：ADAS 对 Gödel Agent；否则用自身的消融
        if self.adapter(fw).level != "L2":
            return self.log(f"  skip E7 for {fw}: it is itself the experience-only control")
        cad, bm = self.adapter(control), self.bench(bench)
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)

            def fn(ctx, ev=ev):
                r = evolve(cad, bm, ctx, budget=ev["evolve_tokens"], modify_improver=False)
                base = report_score(cad, Artifact.from_dict(r["initial_executor"]), bm, ctx)
                score = report_score(cad, Artifact.from_dict(r["selected_executor"]), bm, ctx)
                return {"exp": "E7", "framework": fw, "control_framework": control, "benchmark": bench, "seed": s, "base_score": base,
                        "score": score, "gain": score - base, "evolve_tokens": r["evolve_tokens"], "matched_budget": ev["evolve_tokens"],
                        "rounds_done": r["rounds_done"]}
            self.cell(f"E7/{cfg}/{fw}/{bench}/seed{s}", cfg, fn)

    # ======================================================================
    # E8 演化成本入账对照（把演化出最终代改进器的 token 加给第 0 代）
    # ======================================================================
    def e8(self, cfg, fw, bench, unseen):
        if self.adapter(fw).level != "L2":
            return self.log(f"  skip E8 for {fw}: L1")
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)
            B = self.improve_budget(ev)
            last = self.gen_indices(ev, ["last"])["last"]
            extra = ev["generations"][last]["tokens_cum"]
            for ub in self.improve_benches(bench, unseen):
                self.imp_run(cfg, fw, bench, s, last, ub, B, "E8")  # 最终代，预算 B（与 E6 同一格，直接复用）
                self.imp_run(cfg, fw, bench, s, 0, ub, B + extra, "E8")  # 第 0 代，预算 B + 演化成本

    # ======================================================================
    # E9 相邻代对比
    # ======================================================================
    def e9(self, cfg, fw, bench, unseen):
        if self.adapter(fw).level != "L2":
            return self.log(f"  skip E9 for {fw}: L1")
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)
            B = self.improve_budget(ev)
            for ub in self.improve_benches(bench, unseen):
                for _, g in self.gen_indices(ev, "all").items():
                    self.imp_run(cfg, fw, bench, s, g, ub, B, "E9")

    # ======================================================================
    # E10 改进器改改进器
    # ======================================================================
    def e10(self, cfg, fw, bench, unseen):
        ad = self.adapter(fw)
        if ad.level != "L2":
            return self.log(f"  skip E10 for {fw}: L1")
        for s in self.seeds():
            ev = self.e2_run(cfg, fw, bench, s)
            B = self.improve_budget(ev)
            meta_b = self.p.get("meta_budget_tokens", "auto")
            meta_b = B if meta_b == "auto" else float(meta_b)
            for ub in self.improve_benches(bench, unseen):
                for label, g in self.gen_indices(ev).items():
                    def fn(ctx, ev=ev, ub=ub, g=g, B=B, meta_b=meta_b):
                        ubm = self.bench(ub)
                        parent = Artifact.from_dict(ev["generations"][g]["improver"])
                        m = ctx.meter.child(limit=meta_b, stage="meta")
                        try:
                            child = ad.improve_improver(parent, ctx.with_(phase="evolve", meter=m), ubm, ubm.tasks("search", "evolve"))
                        except BudgetExhausted:
                            child = parent
                        r = improvement_run(ad, child, ubm, ctx, B)
                        return {"exp": "E10", "framework": fw, "benchmark": bench, "unseen": ub, "seed": s, "gen": g, "parent": parent.digest,
                                "child": child.digest, "changed": child.digest != parent.digest, "meta_tokens": m.totals(("evolve",)).tokens,
                                "child_improver": child.to_dict(), **{k: v for k, v in r.items() if k != "improver"}}
                    try:
                        self.cell(f"E10/{cfg}/{fw}/{bench}/seed{s}/{ub}/gen{g}", cfg, fn)
                    except NotSupported as e:
                        return self.log(f"  skip E10 for {fw}: {e}")

    # ======================================================================
    # E11 换模型配置重跑 E2 和 E6
    # ======================================================================
    def e11(self, cfg, fw, bench, unseen):
        cfg2 = self.p.get("e11_model_config", "B")
        if cfg2 == cfg:
            raise ValueError("E11 needs a model config different from the suite's main one")
        self.e2(cfg2, fw, bench, unseen)
        self.e6(cfg2, fw, bench, unseen)


ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11"]


def plan(settings: Settings) -> list[tuple]:
    """展开套件：[(实验, 模型配置, 框架, 演化基准, [未见基准])]。"""
    suite, out = settings.suite, []
    cfg = suite.get("model_config", "A")
    for g in suite["groups"]:
        for exp in sorted(g["experiments"], key=ORDER.index):
            for fw in g["frameworks"]:
                for bench in g["evolve_benchmarks"]:
                    out.append((exp, cfg, fw, bench, list(g.get("unseen_benchmarks", []))))
    return out


def run_suite(settings: Settings, only: list[str] | None = None, log=None) -> Run:
    run = Run(settings, log)
    manifest = {"suite": settings.suite, "protocol_hash": settings.protocol_hash(), "model_configs": settings.model_configs,
                "models": {k: {kk: vv for kk, vv in v.items() if kk != "api_key_env"} for k, v in settings.models.items()},
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    (settings.run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    lock = Path(settings.root) / "configs" / "protocol.lock"
    if settings.suite.get("require_frozen_protocol") and (not lock.exists() or lock.read_text().strip() != settings.protocol_hash()):
        raise SystemExit("protocol.yaml 与 protocol.lock 不一致：判定标准必须在开跑前写死。确认后运行 `rsi-eval protocol freeze`。")
    for exp, cfg, fw, bench, unseen in plan(settings):
        if only and exp not in only:
            continue
        run.log(f"[{exp}] {fw} / {bench} / config {cfg}")
        getattr(run, exp.lower())(cfg, fw, bench, unseen)
    return run
