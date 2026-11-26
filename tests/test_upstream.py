"""ADAS 和 Gödel Agent 适配器的离线测试。

用的是真的上游代码（vendor/ 下固定 commit 的那份）和真的数据文件，只有模型是假的：
把计量代理的 forward 换成一个按脚本应答的函数。不联网、不花钱。
vendor/ 或 data/ 没准备好时整组跳过：先运行 `rsi-eval vendor fetch` 和 `rsi-eval data prepare gsm8k`。
"""
import ast
import json
import re
import threading
import urllib.request
from pathlib import Path

import pytest

from rsi_eval.benchmarks import get_benchmark
from rsi_eval.core import Settings, make_root_ctx
from rsi_eval.frameworks import get_adapter
from rsi_eval.frameworks.external import ERROR_PREFIX, NotWired, repo_head
from rsi_eval.proxy import MeteringProxy, repair_json
from rsi_eval.runner import evolve

ROOT = Path(__file__).resolve().parent.parent
READY = all((ROOT / p).exists() for p in ("vendor/adas/_mgsm/search.py", "vendor/godel_agent/src/agent_module.py", "data/gsm8k.jsonl"))
needs_upstream = pytest.mark.skipif(not READY, reason="run `rsi-eval vendor fetch` and `rsi-eval data prepare gsm8k` first")


# --------------------------------------------------------------------------
# 不需要上游代码的部分
# --------------------------------------------------------------------------
def test_repair_json():
    assert json.loads(repair_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(repair_json("```json\n{'thinking': 'x', 'answer': '5'}\n```")) == {"thinking": "x", "answer": "5"}
    assert json.loads(repair_json('Sure! {"a": "b"} hope it helps')) == {"a": "b"}
    assert repair_json("no json here") == "no json here"


@pytest.fixture()
def settings(tmp_path):
    s = Settings.load(ROOT / "configs" / "suites" / "smoke.yaml", out_root=tmp_path)
    s.protocol = json.loads(json.dumps(s.protocol))
    s.protocol["benchmarks"]["gsm8k"] = {"sizes": {"search": 4, "select": 3, "report": 5}}
    s.protocol["select_top_k"] = 2
    return s


def test_role_prefix_pins_model_and_role(settings):
    """/_role/<role>/ 前缀决定记到谁头上、用哪个型号；请求里写的 gpt-4o 不起作用。"""
    ctx = make_root_ctx(settings, "mockA")
    seen = []
    with MeteringProxy(ctx, "mock") as px:
        real = px._mock
        px._mock = lambda path, req, role: (seen.append((path, req["model"], role)), real(path, req, role))[1]
        body = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "What is 2 + 3?"}]}).encode()
        for role in ("executor", "improver"):
            req = urllib.request.Request(f"{px.url}/_role/{role}/v1/chat/completions", data=body, headers={"content-type": "application/json"})
            assert json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"]
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(f"{px.url}/_role/nobody/v1/chat/completions", data=body))
        assert e.value.code == 400
        assert px.env()["RSI_MODEL_EXECUTOR"] == "mock-weak" and "RSI_EVAL_ANTHROPIC_KEY" not in px.child_env(minimal=True)
    assert seen == [("/v1/chat/completions", "mock-weak", "executor"), ("/v1/chat/completions", "mock-strong", "improver")]
    rows = [json.loads(l) for l in (settings.run_dir / "ledger.jsonl").read_text().splitlines()]
    assert [(r["role"], r["model"]) for r in rows] == [("executor", "mock-weak"), ("improver", "mock-strong")]


def test_openai_requests_are_adapted_for_claude(settings, monkeypatch):
    """发给 Claude 的 OpenAI 格式请求：去掉 temperature 和 json_object，n 拆成多次，回复修成合法 JSON。"""
    settings.models["mock-weak"] = {"provider": "anthropic", "model": "claude-x", "max_tokens": 1000, "price": {"input": 1, "output": 1}}
    monkeypatch.setenv("RSI_EVAL_ANTHROPIC_KEY", "k")
    ctx = make_root_ctx(settings, "mockA")
    px = MeteringProxy(ctx, "https://api.openai.com")
    sent = []

    def fake_post(url, body, headers, tries=5):
        sent.append((url, body, headers))
        out = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "```json\n{'answer': '5'}\n```"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        return 200, json.dumps(out).encode(), "application/json"

    px._post = fake_post
    try:
        req = {"model": "gpt-4o", "temperature": 0.5, "n": 2, "max_tokens": 4096, "stop": None, "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": "hi"}]}
        code, raw, _ = px.forward("/v1/chat/completions", b"", req, {}, "executor")
    finally:
        px.server.server_close()
    out = json.loads(raw)
    assert code == 200 and len(sent) == 2 and len(out["choices"]) == 2
    url, body, headers = sent[0]
    assert url == "https://api.anthropic.com/v1/chat/completions" and headers["authorization"] == "Bearer k"
    assert body["model"] == "claude-x" and body["max_tokens"] == 1000
    assert not {"temperature", "n", "response_format", "stop"} & body.keys()
    assert body["messages"][0]["role"] == "system" and "JSON" in body["messages"][0]["content"]
    assert [json.loads(c["message"]["content"]) for c in out["choices"]] == [{"answer": "5"}] * 2
    assert out["usage"] == {"prompt_tokens": 14, "completion_tokens": 6, "total_tokens": 20}


def test_unpinned_upstream_is_rejected(settings, tmp_path):
    ad = get_adapter("adas", {"repo_dir": str(tmp_path / "nope")})
    with pytest.raises(NotWired):
        ad.ensure_repo(make_root_ctx(settings, "mockA"))


# --------------------------------------------------------------------------
# 按脚本应答的假模型
# --------------------------------------------------------------------------
NEW_FORWARD = '''def forward(self, taskInfo):
    agent = LLMAgentBase(['thinking', 'answer'], 'Careful Agent')
    thinking, answer = agent([taskInfo], "Please think meticulously and then solve the task.")
    return answer
'''
BROKEN_FORWARD = '''def forward(self, taskInfo):
    raise ValueError("boom")
'''
NEW_SOLVER = '''def solver(agent, task: str):
    messages = [{"role": "user", "content": f"# Your Task:\\n{task}"}]
    response = agent.action_call_json_format_llm(
        model="gpt-5",
        messages=messages,
        temperature=0.8,
        num_of_response=1,
        role="math expert",
        return_dict_keys=["reasoning", "answer"],
        requirements=HELPER_REQUIREMENTS,
    )
    return_dict = response[0]
    return_dict["answer"] = str(return_dict.get("answer", ""))
    return return_dict
'''
NEW_ANALYSIS = '''def action_display_analysis(analysis):
    return "Analysis Received (v2)."
'''


class FakeLLM:
    """执行者：题目能在题库里找到、且提示里出现 meticulously 时答对，否则答 0。改进器：按脚本出牌。"""

    def __init__(self, tasks):
        self.answers = {t.input: t.target for t in tasks}
        self.improver_calls = 0
        self.models = []
        self.lock = threading.Lock()

    @staticmethod
    def _text(req):
        return "\n".join(str(m.get("content") or "") for m in req.get("messages", []))

    @staticmethod
    def _reply(message):
        body = {"id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                "choices": [{"index": 0, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop", "message": {"role": "assistant", **message}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        return 200, json.dumps(body).encode(), "application/json"

    def forward(self, path, raw, req, headers, role):
        text = self._text(req)
        with self.lock:
            self.models.append((role, req.get("model")))
        if role == "executor":
            ans = next((a for q, a in self.answers.items() if q in text), 0)
            ans = f"{ans:g}" if "meticulously" in text else "0"
            m = re.search(r"\{'.*?'\}", text, re.S)  # ADAS: 系统消息里用 Python 字典写的字段说明
            keys = list(ast.literal_eval(m.group(0))) if m and "Reply EXACTLY" in text else ["reasoning", "answer"]
            return self._reply({"content": json.dumps({k: (ans if "answer" in k else "ok") for k in keys})})
        if req.get("tools"):
            return self._reply({"content": None, "tool_calls": self._godel_script()})
        code = BROKEN_FORWARD if "Error during evaluation" not in text else NEW_FORWARD  # ADAS 元 agent：先交一个有 bug 的，报错后改好
        return self._reply({"content": json.dumps({"thought": "be careful", "name": "Careful Agent", "code": code})})

    def _godel_script(self):
        with self.lock:
            self.improver_calls += 1
            n = self.improver_calls
        call = lambda name, **args: {"id": f"call_{n}_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}  # noqa: E731
        if n == 1:
            return [call("action_display_analysis", analysis="read the solver first"), call("action_read_logic", module_name="agent_module", target_name="solver")]
        if n == 2:
            return [call("action_run_code", code_type="python", code="HELPER_REQUIREMENTS = 'Work meticulously. The answer MUST be a number.'"),
                    call("action_adjust_logic", module_name="agent_module", target_name="solver", new_code=NEW_SOLVER, target_type="function", operation="modify")]
        if n == 3:
            return [call("action_evaluate_on_task")]
        if n == 4:  # 改自己：换掉一个 action 函数
            return [call("action_adjust_logic", module_name="agent_module", target_name="action_display_analysis", new_code=NEW_ANALYSIS, target_type="function", operation="modify")]
        return [call("action_display_analysis", analysis="nothing more to do")]


@pytest.fixture()
def world(settings, monkeypatch):
    ctx = make_root_ctx(settings, "mockA")
    bench = get_benchmark("gsm8k", settings.protocol, ROOT / "data")
    fake = FakeLLM([t for split in bench.splits().values() for t in split])
    monkeypatch.setattr(MeteringProxy, "forward", lambda self, *a: fake.forward(*a))
    return ctx, bench, fake


# --------------------------------------------------------------------------
# ADAS
# --------------------------------------------------------------------------
@needs_upstream
def test_adas_pinned_and_one_generation(world):
    ctx, bench, fake = world
    ad = get_adapter("adas")
    assert repo_head(ad.ensure_repo(ctx)) == ad.commit
    ev = evolve(ad, bench, ctx, rounds=1)

    arch = json.loads((Path(ev["generations"][-1]["archive"]["path"]) / "spec.json").read_text())
    # 上游种子库 7 个 + 元 agent 新出的 1 个；第一版有 bug，经过一轮 debug 才入档
    assert [e["generation"] for e in arch["entries"]] == ["initial"] * 7 + [1]
    assert arch["history"] == [{"generation": 1, "added": True, "debug_rounds": 1, "error": "ValueError('boom')", "meta_refusals": 0, "name": "Careful Agent", "search_score": 1.0}]
    assert all("Bootstrap Confidence Interval" in e["solution"]["fitness"] for e in arch["entries"][:7])
    # L1：改进器不变；选出来的是新 agent；初始执行者是上游的 Chain-of-Thought
    assert len({g["improver"]["digest"] for g in ev["generations"]}) == 1
    selected = json.loads((Path(ev["selected_executor"]["path"]) / "spec.json").read_text())
    initial = json.loads((Path(ev["initial_executor"]["path"]) / "spec.json").read_text())
    assert selected["name"] == "Careful Agent" and initial["name"] == "Chain-of-Thought"
    assert ev["select_scores"][ev["selected_executor"]["digest"]] == 1.0 and ev["select_scores"][ev["initial_executor"]["digest"]] == 0.0
    # 记账：元 agent 的调用算改进器，agent 里的调用算执行者，型号都是角色配置的而不是上游写死的 gpt
    assert {m for r, m in fake.models if r == "executor"} == {"mock-weak"} and {m for r, m in fake.models if r == "improver"} == {"mock-strong"}
    assert sum(r == "improver" for r, _ in fake.models) == 4  # 提案 + 两次反思 + 一次 debug
    rows = [json.loads(l) for l in (ctx.settings.run_dir / "ledger.jsonl").read_text().splitlines()]
    assert sum(r["role"] == "improver" for r in rows) == 4 and ev["cost"]["evolve"]["tokens"] > 0 and ev["cost"]["select"]["tokens"] > 0


@needs_upstream
def test_adas_forward_error_is_scored_zero(world):
    ctx, bench, fake = world
    ad = get_adapter("adas")
    broken = ad._executor(ctx, "_mgsm", {"name": "x", "thought": "", "code": BROKEN_FORWARD})
    task = bench.splits()["search"][0]
    r = ad.run_task(broken, bench, task, ctx, 0)
    assert r.output.startswith(ERROR_PREFIX) and "boom" in r.output and r.score == 0.0


# --------------------------------------------------------------------------
# Gödel Agent
# --------------------------------------------------------------------------
@needs_upstream
def test_godel_self_modification_is_frozen_and_replayed(world):
    ctx, bench, fake = world
    ad = get_adapter("godel_agent", {"calls_per_step": 4, "feedback_batch": 2})
    tasks = bench.tasks("search", "evolve")
    s0 = ad.initial(ctx, bench)
    assert s0.improver.digest == ad.initial(ctx, bench).improver.digest

    # 放开自我修改的一轮：4 次 evolve，改了 solver，也改了自己的一个 action 函数
    s1 = ad.step(s0, ctx, bench, tasks, modify_improver=True)
    arch = s1.archive.load_json()
    assert arch["steps"][0]["calls"] == 4 and arch["steps"][0]["exit"] == "max_calls" and arch["steps"][0]["feedback_scores"] == [1.0]
    assert [e["search_score"] for e in arch["entries"]] == [0.0, 1.0]
    log = s1.executor.load_json()["log"]
    assert [(e["kind"], e["improver"]) for e in log] == [("action_run_code", False), ("action_adjust_logic", False), ("action_adjust_logic", True)]
    imp = s1.improver.load_json()
    assert s1.improver.digest != s0.improver.digest and [e["id"] for e in imp["entries"]] == [log[2]["id"]]
    assert "v2" in imp["sources"]["action_display_analysis"]
    assert "gpt-5" in s1.executor.load_json()["solver_source"]

    # 冻结的执行者在新进程里重放后能答题：依赖 run_code 里建的全局变量，型号仍是执行模型
    fake.models.clear()
    report = bench.tasks("report", "report")[0]
    assert bench.score(report, ad.solve(s1.executor, bench, report, ctx, 0)) == 1.0
    assert bench.score(report, ad.solve(s0.executor, bench, report, ctx, 0)) == 0.0
    assert fake.models == [("executor", "mock-weak")] * 2

    # 同起点对照：第 1 代改进器装到初始状态上，冻结着再跑一轮。改进器一个字节不变，它的改动在新进程里已生效
    fake.improver_calls = 0
    f1 = ad.step(ad.with_improver(s0, s1.improver), ctx, bench, tasks, modify_improver=False)
    assert f1.improver.digest == s1.improver.digest
    flog = Path(ctx.settings.run_dir / "work" / "godel_agent").glob(f"*-{s1.improver.digest[:6]}.log")
    assert "Analysis Received (v2)." in next(flog).read_text()
    assert [e["improver"] for e in f1.executor.load_json()["log"]] == [True, False, False]

    # 第 0 代改进器冻结着跑：第 4 次调用想改自己，被撤销并报错，其余照常
    fake.improver_calls = 0
    f0 = ad.step(s0, ctx, bench, tasks, modify_improver=False)
    assert f0.improver.digest == s0.improver.digest
    assert [(e["kind"], e["improver"]) for e in f0.executor.load_json()["log"]] == [("action_run_code", False), ("action_adjust_logic", False)]
    assert "is disabled in this run" in json.dumps(f0.archive.load_json()["history"])

    # 接着跑第二轮：从上一轮的运行时状态和对话历史继续
    s2 = ad.step(s1, ctx, bench, tasks, modify_improver=True)
    assert s2.generation == 2 and s2.improver.digest == s1.improver.digest and len(s2.archive.load_json()["entries"]) == 3
    assert ad.candidates(s2, 3) == [s1.executor, s0.executor]


@needs_upstream
def test_godel_runs_under_the_protocol_runner(world):
    ctx, bench, fake = world
    ad = get_adapter("godel_agent", {"calls_per_step": 3, "feedback_batch": 2})
    ev = evolve(ad, bench, ctx, rounds=2)
    assert ev["rounds_done"] == 2 and ev["select_scores"][ev["selected_executor"]["digest"]] == 1.0
    assert ev["selected_executor"]["digest"] != ev["initial_executor"]["digest"]
    assert ev["generations"][1]["improver"]["digest"] == ev["generations"][0]["improver"]["digest"]  # 前 3 次调用还没改自己
    assert ev["generations"][2]["improver"]["digest"] != ev["generations"][0]["improver"]["digest"]  # 第 4 次改了


# --------------------------------------------------------------------------
# 拒答、其他厂商
# --------------------------------------------------------------------------
def test_refusals_are_flagged_and_abort_a_doomed_run(settings):
    from rsi_eval.frameworks.external import UpstreamRefused
    from rsi_eval.proxy import is_refusal

    assert is_refusal(json.dumps({"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}).encode())
    assert is_refusal(json.dumps({"stop_reason": "refusal", "content": []}).encode())
    assert not is_refusal(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "5"}}]}).encode())

    ctx = make_root_ctx(settings, "mockA")
    ad = get_adapter("adas", {"refusal_min_calls": 10})
    ad._note_refusals(ctx, {"calls": {"executor": 9}, "refusals": {"executor": 9}})  # 样本还不够
    ad._note_refusals(ctx, {"calls": {"improver": 20}, "refusals": {"improver": 2}})  # 另一个模型，拒答率低
    with pytest.raises(UpstreamRefused, match="mock-weak"):
        ad._note_refusals(ctx, {"calls": {"executor": 1}, "refusals": {"executor": 0}})


def test_other_vendor_models_are_forwarded_to_their_endpoint(settings, monkeypatch):
    settings.models["mock-weak"] = {"provider": "openai_compat", "model": "other-1", "base_url": "https://llm.example/v1", "api_key_env": "OTHER_KEY"}
    monkeypatch.setenv("OTHER_KEY", "ok")
    px = MeteringProxy(make_root_ctx(settings, "mockA"), "https://api.openai.com")
    sent = []
    px._post = lambda url, body, headers, tries=5, **kw: (sent.append((url, body, headers)), (200, b"{}", "application/json"))[1]
    try:
        px.forward("/v1/chat/completions", b"", {"model": "other-1", "temperature": 0.5, "stop": None, "messages": []}, {}, "executor")
        assert "OTHER_KEY" in px.secret_env_names()
    finally:
        px.server.server_close()
    assert sent == [("https://llm.example/v1/chat/completions", {"model": "other-1", "temperature": 0.5, "messages": []},
                     {"content-type": "application/json", "authorization": "Bearer ok"})]


def test_model_spec_can_override_the_callers_token_limit(settings, monkeypatch):
    settings.models["mock-weak"] = {"provider": "openai_compat", "model": "r-1", "base_url": "https://llm.example/v1", "max_tokens": 16000, "override_max_tokens": True}
    px = MeteringProxy(make_root_ctx(settings, "mockA"), "https://api.openai.com")
    sent = []
    px._post = lambda url, body, headers, tries=5, **kw: (sent.append(body), (200, b"{}", "application/json"))[1]
    try:
        px.forward("/v1/chat/completions", b"", {"model": "r-1", "max_tokens": 4096, "messages": []}, {}, "executor")
    finally:
        px.server.server_close()
    assert sent[0]["max_tokens"] == 16000


def test_streamed_upstream_reply_is_reassembled():
    from rsi_eval.proxy import _collect_stream

    def sse(*events):
        return [("data: " + (e if isinstance(e, str) else json.dumps(e)) + "\n").encode() for e in events]

    got = _collect_stream(sse(
        {"id": "x", "model": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "hm"}}]},
        {"choices": [{"index": 0, "delta": {"content": "4"}}]},
        {"choices": [{"index": 0, "delta": {"content": "2", "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{\"a\""}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7}},
        "[DONE]",
    ))
    msg = got["choices"][0]["message"]
    assert (msg["content"], msg["reasoning_content"], got["choices"][0]["finish_reason"]) == ("42", "hm", "tool_calls")
    assert msg["tool_calls"] == [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]
    assert got["usage"] == {"prompt_tokens": 5, "completion_tokens": 7}
    with pytest.raises(ConnectionError):  # 流中途断了要当失败重试，不能当成一条空回复
        _collect_stream(sse({"choices": [{"index": 0, "delta": {"content": "4"}}]}))
