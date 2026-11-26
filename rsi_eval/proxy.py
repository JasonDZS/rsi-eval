"""计量代理：让上游框架（子进程里跑的 Gödel Agent / ADAS / HyperAgents / HGM）的模型调用也入账。

上游代码不用改记账逻辑，只要把 ANTHROPIC_BASE_URL / OPENAI_BASE_URL 指到这个代理：
- 每个请求按模型名对应到角色（executor / improver），token 和金额记到当前 Ctx 的 Meter 上；
- 预算用完后直接返回 429，并置 exhausted 标志，适配器据此结束这次改进运行；
- 真实密钥只在代理进程里，子进程拿到的是占位密钥。

支持 Anthropic Messages（/v1/messages）和 OpenAI 兼容（/v1/chat/completions）两种格式，
流式响应整段转发后再从 SSE 事件里取 usage。upstream="mock" 时用 MockClient 应答（离线测试）。

角色的两种确定方式：
- 路径前缀 /_role/<role>/v1/...：调用方明说这次调用算谁的，代理把请求里的型号强制换成该角色配置的型号。
  上游框架的驱动脚本都用这种方式，所以上游代码里写死的 gpt-4o 之类的型号不起作用，执行模型没法被“自改进”偷换。
- 没有前缀：按请求里的型号反查角色（旧行为）。

OpenAI 格式的请求发给 Claude：Gödel Agent 和 ADAS 的上游代码只会说 OpenAI Chat Completions。
角色对应的模型是 anthropic 时，代理把请求转到 Anthropic 的 OpenAI 兼容端点，并抹平三处差异：
- temperature / top_p：新模型不接受，去掉（模型定义里写 supports_temperature: true 可保留）；
- response_format={"type": "json_object"}：兼容端点不接受，去掉后在系统消息里补一句“只输出一个 JSON 对象”，
  再把回复里的代码围栏、Python 风格的单引号字典修成合法 JSON（上游拿到后直接 json.loads）；
- n > 1：兼容端点只支持 n=1，代理连发 n 次再把 choices 和 usage 合并。
"""
from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .budget import BudgetExhausted, Ledger, Meter, SpendCapExceeded, Usage
from .llm import ANTHROPIC_API_URL, DEFAULT_ANTHROPIC_KEY_ENV, anthropic_key


def parse_usage(body: bytes, content_type: str) -> Usage:
    """从响应里取 token 用量；JSON 和 SSE 都认。"""
    inp = out = 0

    def take(u: dict):
        nonlocal inp, out
        if not isinstance(u, dict):
            return
        i = (u.get("input_tokens") or u.get("prompt_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
        o = u.get("output_tokens") or u.get("completion_tokens") or 0
        inp, out = max(inp, i), max(out, o)

    text = body.decode("utf-8", "replace")
    if "event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    ev = json.loads(line[5:])
                except json.JSONDecodeError:
                    continue
                take(ev.get("usage"))
                take((ev.get("message") or {}).get("usage"))
    else:
        try:
            take(json.loads(text).get("usage"))
        except (json.JSONDecodeError, AttributeError):
            pass
    return Usage(inp, out)


def is_refusal(body: bytes) -> bool:
    """应答是不是拒答：Anthropic 格式 stop_reason=refusal，OpenAI 格式 finish_reason=content_filter。"""
    try:
        out = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(out, dict):
        return False
    return out.get("stop_reason") == "refusal" or any((c or {}).get("finish_reason") == "content_filter" for c in out.get("choices") or [])


def repair_json(text: str) -> str:
    """把模型回复修成能直接 json.loads 的对象文本；修不了就原样返回（上游自己处理解析失败）。"""
    t = (text or "").strip()
    cands = [t]
    m = re.search(r"```(?:json|JSON)?\s*(.*?)```", t, re.S)
    if m:
        cands.append(m.group(1).strip())
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        cands.append(t[a : b + 1])
    for c in cands:
        try:
            if isinstance(json.loads(c), dict):
                return c
        except (json.JSONDecodeError, RecursionError):
            pass
    for c in cands:
        try:
            obj = ast.literal_eval(c)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(obj, dict):
            try:
                return json.dumps(obj, ensure_ascii=False)
            except (TypeError, ValueError):
                continue
    return text


JSON_MODE_HINT = "Respond with a single valid JSON object and nothing else (double-quoted keys and strings, no markdown code fences)."
ROLE_PREFIX = re.compile(r"^/_role/([A-Za-z_]+)(/.*)$")
RETRY_STATUS = (429, 500, 502, 503, 529)


class MeteringProxy:
    """用法：
        with MeteringProxy(ctx, upstream="https://api.anthropic.com") as px:
            subprocess.run(cmd, env={**os.environ, **px.env()})
        if px.exhausted: raise BudgetExhausted(...)
    """

    def __init__(self, ctx, upstream: str, host: str = "127.0.0.1", port: int = 0):
        self.ctx, self.upstream, self.exhausted = ctx, upstream.rstrip("/"), False
        self.cap_exceeded = False  # 套件的真实花费硬上限到了（和单次运行的预算用完区分开）
        self.calls: dict = {}  # role -> 成功的调用数
        self.refusals: dict = {}  # role -> 其中被模型拒答的次数
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype="application/json", extra: dict | None = None):
                self.send_response(code)
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("content-type", ctype)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("content-length", 0)))
                try:
                    req = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    req = {}
                path, pinned = self.path, ROLE_PREFIX.match(self.path)
                if pinned:
                    role, path = pinned.group(1), pinned.group(2)
                    if role not in proxy.ctx.roles:
                        return self._send(400, json.dumps({"error": {"type": "invalid_request_error", "message": f"rsi-eval: unknown role '{role}'"}}).encode())
                    spec = proxy._spec(role)
                    req["model"] = spec.get("model", proxy.ctx.roles[role])  # 型号由角色决定，不由调用方决定
                    raw = json.dumps(req).encode()
                else:
                    role = proxy.role_of(req.get("model", ""))
                try:
                    proxy.ctx.meter.check(proxy.ctx.phase)
                except (BudgetExhausted, SpendCapExceeded) as e:
                    proxy.exhausted = True
                    proxy.cap_exceeded = proxy.cap_exceeded or isinstance(e, SpendCapExceeded)
                    # x-should-retry: false 让 openai / anthropic SDK 不要重试这个 429
                    return self._send(429, json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": f"rsi-eval budget: {e}"}}).encode(),
                                      extra={"x-should-retry": "false"})
                try:
                    code, body, ctype = proxy.forward(path, raw, req, dict(self.headers), role)
                except Exception as e:  # 上游连不上等：让子进程看到一个明确的 502，而不是连接被掐断
                    code, body, ctype = 502, json.dumps({"error": {"type": "api_error", "message": f"rsi-eval proxy: {type(e).__name__}: {e}"}}).encode(), "application/json"
                if code == 200:
                    usage = parse_usage(body, ctype)
                    refused = is_refusal(body)
                    proxy.calls[role] = proxy.calls.get(role, 0) + 1
                    proxy.refusals[role] = proxy.refusals.get(role, 0) + int(refused)
                    proxy.ctx.meter.charge(proxy.ctx.phase, role, proxy.ctx.roles.get(role, req.get("model", "?")), usage, proxy.ctx.price_usd(role, usage), refusal=refused)
                self._send(code, body, ctype)

        self.server = ThreadingHTTPServer((host, port), H)
        self.url = f"http://{host}:{self.server.server_address[1]}"

    def role_of(self, model: str) -> str:
        """请求里的型号对应哪个角色。两个角色同型号时（配置 B）记为 executor，按总量对齐不受影响。"""
        for role in ("executor", "improver"):
            name = self.ctx.roles.get(role)
            if name and self.ctx.settings.models[name].get("model", name) == model:
                return role
        return "improver"

    def _spec(self, role: str) -> dict:
        """角色对应的模型定义（取 api_key_env 用）；独立运行时没有角色表，返回空。"""
        return self.ctx.settings.models.get(self.ctx.roles.get(role), {}) or {}

    def forward(self, path, raw, req, headers, role):
        spec = self._spec(role)
        if self.upstream == "mock":
            return self._mock(path, req, role)
        if "/chat/completions" in path and spec.get("provider") == "anthropic":
            return self._anthropic_compat(req, spec)
        base = spec.get("base_url") or os.environ.get(spec.get("base_url_env") or "", "")
        if "/chat/completions" in path and spec.get("provider") == "openai_compat" and base:
            # 其他厂商的 OpenAI 兼容接口：按模型定义里的地址（base_url，或 base_url_env 指的环境变量）和密钥转发
            base = base.rstrip("/")
            cap = spec.get("max_tokens")
            asked = req.get("max_tokens") or req.get("max_completion_tokens")
            # 上游不设上限时用模型定义里的。override_max_tokens: true 时总是用模型定义里的：
            # 上游写死的上限（ADAS 是 4096）是按“输出全是可见内容”的模型定的，推理型模型会在出结果前把它用完。
            if cap and (not asked or spec.get("override_max_tokens")):
                req = {k: v for k, v in req.items() if k != "max_completion_tokens"}
                req["max_tokens"] = int(cap)
            url = base + (path[len("/v1"):] if base.endswith("/v1") and path.startswith("/v1/") else path)
            key = os.environ.get(spec.get("api_key_env", "OPENAI_API_KEY"), "")
            # extra_body：这个模型每次请求都要带的参数（比如关掉思考模式）。timeout：单次请求的读超时，到了就重试，
            # 用来对付偶尔一直不回的线路。
            body = {**{k: v for k, v in req.items() if v is not None}, **(spec.get("extra_body") or {})}
            return self._post(url, body, {"content-type": "application/json", "authorization": "Bearer " + key},
                              timeout=float(spec.get("timeout", 900)))
        h = {k: v for k, v in headers.items() if k.lower() in ("content-type", "anthropic-version", "anthropic-beta", "accept")}
        key = anthropic_key(self._spec(role))
        if "/messages" in path and key:
            h["x-api-key"] = key
        else:
            h["authorization"] = "Bearer " + os.environ.get("RSI_EVAL_UPSTREAM_KEY", os.environ.get("OPENAI_API_KEY", ""))
        r = urllib.request.Request(self.upstream + path, data=raw, headers=h, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=900) as resp:
                return resp.status, resp.read(), resp.headers.get("content-type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("content-type", "application/json")

    # ---- OpenAI 格式 -> Claude ----
    def _post(self, url: str, body: dict, headers: dict, tries: int = 5, timeout: float = 900):
        data = json.dumps(body).encode()
        for i in range(tries):
            r = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                # 每次重试把读超时翻倍：短超时用来甩掉挂死的连接，但确实要生成很久的请求不能永远被它打断
                with urllib.request.urlopen(r, timeout=min(900, timeout * 2 ** i)) as resp:
                    return resp.status, resp.read(), resp.headers.get("content-type", "application/json")
            except urllib.error.HTTPError as e:
                out = e.code, e.read(), e.headers.get("content-type", "application/json")
                if e.code not in RETRY_STATUS or i == tries - 1:
                    return out
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if i == tries - 1:
                    raise
            time.sleep(min(30, 2 ** i))

    def _anthropic_compat(self, req: dict, spec: dict):
        body = {k: v for k, v in req.items() if v is not None}
        body["model"] = spec.get("model", body.get("model"))
        rf = body.get("response_format") or {}
        want_json = rf.get("type") == "json_object"
        if rf.get("type") != "json_schema":
            body.pop("response_format", None)
        if not spec.get("supports_temperature", False):
            body.pop("temperature", None)
            body.pop("top_p", None)
        for k in ("frequency_penalty", "presence_penalty", "logit_bias", "seed", "logprobs", "top_logprobs", "user", "parallel_tool_calls"):
            body.pop(k, None)
        n = max(1, int(body.pop("n", 1) or 1))
        cap = int(spec.get("max_tokens", 4096))
        asked = body.pop("max_completion_tokens", None) or body.get("max_tokens")
        body["max_tokens"] = min(int(asked), cap) if asked else cap
        if want_json:
            body["messages"] = [{"role": "system", "content": JSON_MODE_HINT}, *body.get("messages", [])]
        body["stream"] = False
        url = spec.get("base_url", ANTHROPIC_API_URL).rstrip("/") + "/v1/chat/completions"
        headers = {"content-type": "application/json", "authorization": "Bearer " + (anthropic_key(spec) or "")}
        merged, usage = None, {"prompt_tokens": 0, "completion_tokens": 0}
        for i in range(n):
            code, raw, ctype = self._post(url, body, headers)
            if code != 200:
                return code, raw, ctype
            out = json.loads(raw)
            for ch in out.get("choices", []):
                msg = ch.get("message") or {}
                if want_json and isinstance(msg.get("content"), str):
                    msg["content"] = repair_json(msg["content"])
                ch["index"] = i
            u = out.get("usage") or {}
            usage["prompt_tokens"] += u.get("prompt_tokens") or 0
            usage["completion_tokens"] += u.get("completion_tokens") or 0
            if merged is None:
                merged = out
            else:
                merged["choices"].extend(out.get("choices", []))
        merged["usage"] = {**usage, "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]}
        return 200, json.dumps(merged).encode(), "application/json"

    def _mock(self, path, req, role):
        msgs = req.get("messages", [])
        flat = lambda c: c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))  # noqa: E731
        system = req.get("system") if isinstance(req.get("system"), str) else None
        prompt = "\n".join(flat(m.get("content", "")) for m in msgs if m.get("role") != "system")
        c = self.ctx.client(role).complete(prompt, system=system, sample=self.ctx.nonce())
        if "/messages" in path:
            body = {"id": "msg_mock", "type": "message", "role": "assistant", "model": req.get("model"), "content": [{"type": "text", "text": c.text}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": c.usage.input_tokens, "output_tokens": c.usage.output_tokens}}
        else:
            body = {"id": "chatcmpl-mock", "object": "chat.completion", "model": req.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": c.text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": c.usage.input_tokens, "completion_tokens": c.usage.output_tokens, "total_tokens": c.usage.total}}
        return 200, json.dumps(body).encode(), "application/json"

    def model_of(self, role: str) -> str:
        """角色对应的厂商型号（给驱动脚本用，只是展示给上游 agent 看；真正生效的是代理按角色强制的型号）。"""
        name = self.ctx.roles.get(role, "")
        return (self.ctx.settings.models.get(name) or {}).get("model", name)

    def env(self) -> dict:
        """给子进程的环境变量：两种 SDK 都指向代理，密钥是占位符。
        RSI_PROXY_URL + /_role/<role>/v1 是按角色记账的 OpenAI 兼容入口。"""
        return {"RSI_PROXY_URL": self.url, "RSI_MODEL_EXECUTOR": self.model_of("executor"), "RSI_MODEL_IMPROVER": self.model_of("improver"),
                "ANTHROPIC_BASE_URL": self.url, "ANTHROPIC_API_KEY": "rsi-eval-proxy",
                "OPENAI_BASE_URL": self.url + "/v1", "OPENAI_API_BASE": self.url + "/v1", "OPENAI_API_KEY": "rsi-eval-proxy",
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}

    def secret_env_names(self) -> set:
        """装着真实密钥的环境变量名：默认的实验密钥、上游密钥，以及各模型自己指定的 api_key_env。"""
        names = {DEFAULT_ANTHROPIC_KEY_ENV, "RSI_EVAL_UPSTREAM_KEY"}
        names |= {m["api_key_env"] for m in self.ctx.settings.models.values() if isinstance(m, dict) and m.get("api_key_env")}
        return names

    #: minimal=True 时子进程只继承这些变量。上游框架会执行模型写的代码，宿主环境里别的东西（会话令牌等）不该让它读到。
    SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ", "USER", "SHELL", "VIRTUAL_ENV",
                "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy",
                "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "PIP_CERT", "PIP_INDEX_URL", "NODE_EXTRA_CA_CERTS")

    def child_env(self, base: dict | None = None, extra: dict | None = None, minimal: bool = False) -> dict:
        """子进程的完整环境：继承 base（默认当前环境），去掉所有真实密钥，再指向代理。
        真实密钥只留在代理进程里，子进程读不到，也就没法绕过代理直接调用。
        minimal=True 时改成白名单：只留 SAFE_ENV 里的变量。"""
        secrets = self.secret_env_names()
        src = os.environ if base is None else base
        env = {k: v for k, v in src.items() if k not in secrets and (not minimal or k in self.SAFE_ENV)}
        return {**env, **self.env(), **(extra or {})}

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def serve(host: str, port: int, upstream: str, ledger_path: str, limit_tokens: int | None):
    """独立运行：只记流水和可选的总量上限，不绑定实验。用于手工调试上游框架。"""
    from pathlib import Path
    from types import SimpleNamespace

    meter = Meter(Ledger(Path(ledger_path)), tags={"source": "proxy"}, limit=limit_tokens)
    ctx = SimpleNamespace(meter=meter, phase="evolve", roles={}, settings=SimpleNamespace(models={}), price_usd=lambda role, usage: 0.0)
    px = MeteringProxy(ctx, upstream, host, port)
    print(f"metering proxy on {px.url} -> {upstream}; ledger {ledger_path}")
    try:
        px.server.serve_forever()
    except KeyboardInterrupt:
        print(json.dumps(meter.summary()["total"]))
