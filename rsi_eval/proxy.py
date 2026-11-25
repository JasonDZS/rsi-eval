"""计量代理：让上游框架（子进程里跑的 Gödel Agent / ADAS / HyperAgents / HGM）的模型调用也入账。

上游代码不用改记账逻辑，只要把 ANTHROPIC_BASE_URL / OPENAI_BASE_URL 指到这个代理：
- 每个请求按模型名对应到角色（executor / improver），token 和金额记到当前 Ctx 的 Meter 上；
- 预算用完后直接返回 429，并置 exhausted 标志，适配器据此结束这次改进运行；
- 真实密钥只在代理进程里，子进程拿到的是占位密钥。

支持 Anthropic Messages（/v1/messages）和 OpenAI 兼容（/v1/chat/completions）两种格式，
流式响应整段转发后再从 SSE 事件里取 usage。upstream="mock" 时用 MockClient 应答（离线测试）。
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .budget import BudgetExhausted, Ledger, Meter, SpendCapExceeded, Usage
from .llm import anthropic_key


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


class MeteringProxy:
    """用法：
        with MeteringProxy(ctx, upstream="https://api.anthropic.com") as px:
            subprocess.run(cmd, env={**os.environ, **px.env()})
        if px.exhausted: raise BudgetExhausted(...)
    """

    def __init__(self, ctx, upstream: str, host: str = "127.0.0.1", port: int = 0):
        self.ctx, self.upstream, self.exhausted = ctx, upstream.rstrip("/"), False
        proxy = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
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
                role = proxy.role_of(req.get("model", ""))
                try:
                    proxy.ctx.meter.check(proxy.ctx.phase)
                except (BudgetExhausted, SpendCapExceeded) as e:
                    proxy.exhausted = True
                    return self._send(429, json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": f"rsi-eval budget: {e}"}}).encode())
                code, body, ctype = proxy.forward(self.path, raw, req, dict(self.headers), role)
                if code == 200:
                    usage = parse_usage(body, ctype)
                    proxy.ctx.meter.charge(proxy.ctx.phase, role, proxy.ctx.roles.get(role, req.get("model", "?")), usage, proxy.ctx.price_usd(role, usage))
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

    def forward(self, path, raw, req, headers, role):
        if self.upstream == "mock":
            return self._mock(path, req, role)
        h = {k: v for k, v in headers.items() if k.lower() in ("content-type", "anthropic-version", "anthropic-beta", "accept")}
        if "/messages" in path and anthropic_key():
            h["x-api-key"] = anthropic_key()
        else:
            h["authorization"] = "Bearer " + os.environ.get("RSI_EVAL_UPSTREAM_KEY", os.environ.get("OPENAI_API_KEY", ""))
        r = urllib.request.Request(self.upstream + path, data=raw, headers=h, method="POST")
        try:
            with urllib.request.urlopen(r, timeout=900) as resp:
                return resp.status, resp.read(), resp.headers.get("content-type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("content-type", "application/json")

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

    def env(self) -> dict:
        """给子进程的环境变量：两种 SDK 都指向代理，密钥是占位符。"""
        return {"ANTHROPIC_BASE_URL": self.url, "ANTHROPIC_API_KEY": "rsi-eval-proxy",
                "OPENAI_BASE_URL": self.url + "/v1", "OPENAI_API_BASE": self.url + "/v1", "OPENAI_API_KEY": "rsi-eval-proxy",
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}

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
