"""上游框架的真实模型冒烟：在 GSM8K 的几道题上跑一轮演化，看链路、记账和拒答情况。花费很小。

用法：python scripts/upstream_smoke.py <adas|godel_agent> [--config C] [--search 3] [--calls 4] [--cap 0.5]
结果在 runs/upstream-smoke-<框架>-<配置>/。
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from rsi_eval.benchmarks import get_benchmark
from rsi_eval.core import Settings, make_root_ctx
from rsi_eval.frameworks import get_adapter
from rsi_eval.runner import evolve

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("framework", choices=["adas", "godel_agent"])
ap.add_argument("--config", default="C")
ap.add_argument("--search", type=int, default=3)
ap.add_argument("--calls", type=int, default=4, help="Gödel Agent 每轮的 evolve 次数")
ap.add_argument("--cap", type=float, default=0.5, help="花费硬上限（美元；模型没填价格时不起作用）")
a = ap.parse_args()

s = Settings.load(ROOT / "configs/suites/smoke.yaml", out_root=ROOT / "runs")
s.suite = {"name": f"upstream-smoke-{a.framework}-{a.config}", "hard_cap_usd": a.cap, "max_workers": 8}
s.run_dir = ROOT / "runs" / s.suite["name"]
s.protocol["benchmarks"]["gsm8k"] = {"sizes": {"search": a.search, "select": 2, "report": 3}}
s.protocol["select_top_k"] = 1
ctx = make_root_ctx(s, a.config)
bench = get_benchmark("gsm8k", s.protocol, ROOT / "data")
cfg = {"adas": {}, "godel_agent": {"calls_per_step": a.calls, "feedback_batch": a.search}}[a.framework]
t = time.time()
ev = evolve(get_adapter(a.framework, cfg), bench, ctx, rounds=1)
print("models", ctx.roles, "seconds", round(time.time() - t))
print(json.dumps({k: ev[k] for k in ("select_scores", "rounds_done", "stopped", "evolve_tokens", "total_usd")}, indent=1))
arch = json.loads((Path(ev["generations"][-1]["archive"]["path"]) / "spec.json").read_text())
if a.framework == "adas":
    print(json.dumps(arch["history"], indent=1, ensure_ascii=False))
    print([(e["solution"]["name"], e["search_score"]) for e in arch["entries"]])
else:
    print(json.dumps(arch["steps"], indent=1, ensure_ascii=False))
    print("search scores", [e["search_score"] for e in arch["entries"]])
    head = json.loads((Path(arch["head"]["path"]) / "spec.json").read_text())
    print("action log", [(e["kind"], e["improver"], e["ok"], len(e["llm"])) for e in head["log"]])
    print(head.get("solver_source", "")[:2000])
rows = [json.loads(l) for l in (s.run_dir / "ledger.jsonl").read_text().splitlines()]
print("ledger (role, phase, refusal) ->", dict(Counter((r["role"], r["phase"], bool(r.get("refusal"))) for r in rows if not r["cached"])))
