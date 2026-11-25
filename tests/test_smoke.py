"""离线测试：假模型 + 假基准。不联网、不花钱。"""
import json
import shutil
import urllib.request
from pathlib import Path

import pytest

from rsi_eval.benchmarks import get_benchmark
from rsi_eval.budget import BudgetExhausted, Ledger, Meter, Usage
from rsi_eval.core import ProtocolViolation, Settings, make_root_ctx
from rsi_eval.experiments import run_suite
from rsi_eval.frameworks import get_adapter
from rsi_eval.proxy import MeteringProxy
from rsi_eval.report import build, write
from rsi_eval.runner import evolve, improvement_run
from rsi_eval.stats import signflip_p, slope, summarize, t_ci

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def settings(tmp_path):
    s = Settings.load(ROOT / "configs" / "suites" / "smoke.yaml", out_root=tmp_path)
    s.suite = {**s.suite, "params": {"rounds": 3, "e11_model_config": "mockB", "repeats": 2}}
    return s


def test_meter_budget_and_nesting():
    root = Meter(Ledger(None))
    child = root.child(limit=100)
    child.charge("evolve", "improver", "m", Usage(60, 50), 0.0)
    assert root.totals().tokens == 110
    with pytest.raises(BudgetExhausted):
        child.check("evolve")
    child.check("select")  # 选择和报告阶段不受演化预算限制
    child.check("report")


def test_report_split_is_locked(settings):
    b = get_benchmark("mock_arith", settings.protocol, settings.root / "data")
    sp = b.splits()
    ids = [set(t.id for t in sp[k]) for k in ("search", "select", "report")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    with pytest.raises(ProtocolViolation):
        b.tasks("report", "evolve")
    with pytest.raises(ProtocolViolation):
        b.tasks("search", "report")


def test_evolve_freezes_generations_and_respects_budget(settings):
    ctx = make_root_ctx(settings, "mockA")
    ad, b = get_adapter("ref_l2"), get_benchmark("mock_arith", settings.protocol, settings.root / "data")
    ev = evolve(ad, b, ctx, rounds=3)
    assert len(ev["generations"]) == 4 and ev["rounds_done"] == 3
    assert all(Path(g["improver"]["path"]).exists() for g in ev["generations"])
    # 冻结改进器 + 固定预算：改进器不变，演化 token 不会明显超出预算
    imp0 = ev["generations"][0]["improver"]
    from rsi_eval.core import Artifact

    budget = ev["evolve_tokens"] / 2
    r = improvement_run(ad, Artifact.from_dict(imp0), b, ctx, budget)
    assert r["evolve_tokens"] < budget * 1.5
    assert r["improver"] == imp0["digest"]
    ev2 = evolve(ad, b, ctx, rounds=2, modify_improver=False)
    assert len({g["improver"]["digest"] for g in ev2["generations"]}) == 1


def test_l1_never_changes_improver(settings):
    ctx = make_root_ctx(settings, "mockA")
    ev = evolve(get_adapter("ref_l1"), get_benchmark("mock_arith", settings.protocol, settings.root / "data"), ctx, rounds=3)
    assert len({g["improver"]["digest"] for g in ev["generations"]}) == 1


def test_full_suite_runs_resumes_and_reports(settings):
    run_suite(settings, log=lambda m: None)
    cells = list((settings.run_dir / "cells").rglob("*.json"))
    exps = {p.relative_to(settings.run_dir / "cells").parts[0] for p in cells}
    assert {"E1", "E2", "E3", "E4", "E5", "E6ctl", "E7", "IMP", "E10"} <= exps
    assert any("mockB" in str(p) for p in cells)  # E11
    # 续跑：所有格子都已存在，不应再有任何模型调用
    before = (settings.run_dir / "ledger.jsonl").stat().st_size
    run_suite(settings, log=lambda m: None)
    assert (settings.run_dir / "ledger.jsonl").stat().st_size == before
    r = build(settings)
    g = next(x for x in r["groups"] if x["framework"] == "ref_l2" and x["model_config"] == "mockA")
    assert g["L1"]["vs_repeated_sampling"]["n"] == 2
    l3 = g["L3"]["mock_arith_unseen"]
    assert l3["complete"] and set(l3["gain_by_generation"]) == {"first", "mid", "last"}
    assert "第三层" in write(settings).read_text(encoding="utf-8")
    shutil.rmtree(settings.run_dir)


def test_metering_proxy_charges_and_enforces_budget(settings):
    ctx = make_root_ctx(settings, "mockA")
    m = ctx.meter.child(limit=400)
    ctx = ctx.with_(meter=m, phase="evolve")

    def post(url, path, body):
        req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req).read())

    with MeteringProxy(ctx, "mock") as px:
        a = post(px.url, "/v1/messages", {"model": "mock-strong", "messages": [{"role": "user", "content": "What is 2 + 3? " * 50}]})
        assert a["content"][0]["type"] == "text" and m.totals().tokens > 0
        o = post(px.url, "/v1/chat/completions", {"model": "mock-weak", "messages": [{"role": "user", "content": "What is 2 + 3? " * 50}]})
        assert o["choices"][0]["message"]["content"]
        with pytest.raises(urllib.error.HTTPError) as e:
            post(px.url, "/v1/messages", {"model": "mock-strong", "messages": [{"role": "user", "content": "hi"}]})
        assert e.value.code == 429 and px.exhausted


def test_stats():
    assert summarize([0.1, 0.12, 0.09, 0.11, 0.1])["pass"]
    assert not summarize([0.1, -0.12, 0.09, -0.11, 0.02])["pass"]
    lo, hi = t_ci([1.0, 2.0, 3.0])
    assert lo < 2.0 < hi
    assert signflip_p([1, 1, 1, 1, 1]) == pytest.approx(1 / 32)
    assert slope([0, 1, 2, 3]) == pytest.approx(1.0)


def test_anthropic_key_env_order(monkeypatch):
    from rsi_eval.llm import anthropic_key

    for k in ("RSI_EVAL_ANTHROPIC_KEY", "ANTHROPIC_API_KEY", "MY_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert anthropic_key() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "c")
    assert anthropic_key() == "c"
    monkeypatch.setenv("RSI_EVAL_ANTHROPIC_KEY", "b")
    assert anthropic_key() == "b"
    monkeypatch.setenv("MY_KEY", "a")
    assert anthropic_key({"api_key_env": "MY_KEY"}) == "a"
