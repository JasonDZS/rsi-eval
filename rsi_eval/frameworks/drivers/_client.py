"""驱动脚本共用：经过计量代理、按角色记账的 OpenAI 客户端替身。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


class BudgetStop(BaseException):
    """rsi-eval 的预算用完。继承 BaseException，上游代码里的 `except Exception` 和无限重试拦不住它。"""


class RoutedClient:
    """长得像 openai.OpenAI()：只实现上游用到的 client.chat.completions.create(**kw)。

    每次调用先用 role_of(kw) 决定算哪个角色的，再发到代理的 /_role/<role>/v1。
    型号由代理按角色强制，调用方传什么 model 都不起作用。
    """

    def __init__(self, role_of):
        import openai

        self._openai = openai
        base = os.environ["RSI_PROXY_URL"]
        self.models = {"executor": os.environ["RSI_MODEL_EXECUTOR"], "improver": os.environ["RSI_MODEL_IMPROVER"]}
        self._clients = {r: openai.OpenAI(base_url=f"{base}/_role/{r}/v1", api_key="rsi-eval-proxy", max_retries=2, timeout=900) for r in self.models}
        self.role_of = role_of
        self.chat = self
        self.completions = self

    def create(self, **kw):
        role = self.role_of(kw)
        kw["model"] = self.models[role]
        return self._send(role, kw)

    def _send(self, role, kw):
        try:
            return self._clients[role].chat.completions.create(**kw)
        except self._openai.RateLimitError as e:
            if "rsi-eval budget" in str(e):
                raise BudgetStop(str(e)) from None
            raise


def main(commands: dict):
    """驱动脚本的入口：读 --in，跑命令，结果写到 --out。上游的 print 全部改道到 stderr。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=sorted(commands))
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    payload = json.loads(open(a.inp, encoding="utf-8").read())
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    code = 0
    try:
        out = commands[a.command](payload)
    except BudgetStop as e:
        out, code = {"error": f"budget: {e}", "budget": True}, 75
    except BaseException:  # noqa: BLE001  包括上游的 sys.exit
        out, code = {"error": traceback.format_exc()[-4000:]}, 1
    finally:
        sys.stdout = real_stdout
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=str)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # 上游可能留下非守护线程，不等它们
