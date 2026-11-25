"""模型客户端。可插拔：mock（离线）、anthropic、openai 兼容接口。

新增厂商：继承 LLMClient，实现 complete()，用 @register_provider 注册，
然后在 configs/models.yaml 里把 provider 指过去。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .budget import Price, Usage


@dataclass
class Completion:
    text: str
    usage: Usage
    model: str
    stop_reason: str = ""


class LLMClient(ABC):
    """一次无状态的补全调用。`sample` 用来区分同一提示词的多次独立采样。"""

    def __init__(self, name: str, spec: dict):
        self.name = name  # models.yaml 里的键
        self.spec = spec
        self.model = spec.get("model", name)  # 发给厂商的型号，必须固定版本
        p = spec.get("price", {})
        self.price = Price(float(p.get("input", 0)), float(p.get("output", 0)))

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 4096, sample: int = 0) -> Completion: ...


PROVIDERS: dict[str, type[LLMClient]] = {}


def register_provider(name: str):
    def deco(cls):
        PROVIDERS[name] = cls
        return cls

    return deco


def make_client(name: str, models: dict) -> LLMClient:
    if name not in models:
        raise KeyError(f"model '{name}' not defined in models.yaml (have: {sorted(models)})")
    spec = models[name]
    provider = spec.get("provider")
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider '{provider}' for model '{name}' (have: {sorted(PROVIDERS)})")
    return PROVIDERS[provider](name, spec)


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
DEFAULT_ANTHROPIC_KEY_ENV = "RSI_EVAL_ANTHROPIC_KEY"
ANTHROPIC_API_URL = "https://api.anthropic.com"


def anthropic_key(spec: dict | None = None) -> str | None:
    """spec.api_key_env 指定的变量优先，其次 RSI_EVAL_ANTHROPIC_KEY，最后 ANTHROPIC_API_KEY。"""
    names = [(spec or {}).get("api_key_env"), DEFAULT_ANTHROPIC_KEY_ENV, "ANTHROPIC_API_KEY"]
    return next((os.environ[n] for n in names if n and os.environ.get(n)), None)


@register_provider("anthropic")
class AnthropicClient(LLMClient):
    """用官方 SDK。密钥从 RSI_EVAL_ANTHROPIC_KEY 读取（可用 spec.api_key_env 改）。

    不传 temperature（新模型会拒绝）；多次采样靠模型自身的随机性。
    不启用模型回退：实验要求同一组对比内模型版本固定。
    spec.params 里的键原样传给 messages.create（例如 output_config: {effort: low}）。
    """

    def __init__(self, name, spec):
        super().__init__(name, spec)
        import anthropic  # 延迟导入：离线跑 mock 不需要装

        self._anthropic = anthropic
        # 实验专用密钥：默认读 RSI_EVAL_ANTHROPIC_KEY（部分托管环境保留了 ANTHROPIC_API_KEY，
        # 并把 ANTHROPIC_BASE_URL 指向自己的网关，所以这里显式给出密钥和地址，不走 SDK 的环境解析）。
        key = anthropic_key(spec)
        if not key:
            raise RuntimeError(f"model '{name}': 没有找到密钥，请设置环境变量 {spec.get('api_key_env', DEFAULT_ANTHROPIC_KEY_ENV)}")
        self._client = anthropic.Anthropic(api_key=key, base_url=spec.get("base_url", ANTHROPIC_API_URL),
                                           max_retries=int(spec.get("max_retries", 4)))

    def complete(self, prompt, *, system=None, max_tokens=4096, sample=0):
        kwargs = dict(self.spec.get("params", {}))
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=int(self.spec.get("max_tokens", max_tokens)),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        # 拒答(refusal)等情况下没有 text 块，按空输出处理，评分为 0
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        inp = u.input_tokens + (getattr(u, "cache_read_input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        return Completion(text, Usage(inp, u.output_tokens), resp.model, resp.stop_reason or "")


# --------------------------------------------------------------------------
# OpenAI 兼容接口（其他厂商、本地推理服务）
# --------------------------------------------------------------------------
@register_provider("openai_compat")
class OpenAICompatClient(LLMClient):
    """spec: {model, base_url?, api_key_env?, params?}。用于非 Claude 模型。"""

    def __init__(self, name, spec):
        super().__init__(name, spec)
        import openai

        self._client = openai.OpenAI(
            base_url=spec.get("base_url"),
            api_key=os.environ.get(spec.get("api_key_env", "OPENAI_API_KEY")),
            max_retries=int(spec.get("max_retries", 4)),
        )

    def complete(self, prompt, *, system=None, max_tokens=4096, sample=0):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        params = dict(self.spec.get("params", {}))
        params.setdefault(self.spec.get("max_tokens_param", "max_completion_tokens"), int(self.spec.get("max_tokens", max_tokens)))
        resp = self._client.chat.completions.create(model=self.model, messages=msgs, **params)
        ch = resp.choices[0]
        u = resp.usage
        return Completion(ch.message.content or "", Usage(u.prompt_tokens, u.completion_tokens), resp.model, ch.finish_reason or "")


# --------------------------------------------------------------------------
# Mock：离线、确定性，用来验证协议、记账和统计流程
# --------------------------------------------------------------------------
SKILLS = [
    "work step by step",
    "double-check the arithmetic",
    "restate the question first",
    "carry digits carefully",
    "verify the final answer",
    "estimate the magnitude first",
    "write intermediate sums",
    "re-read the numbers",
]
META_SKILLS = [
    "analyze failure patterns before proposing",
    "keep instructions that raised the score",
    "change one thing at a time",
    "prefer checks that target observed errors",
    "drop instructions that did not help",
]
_FILLERS = ["be polite", "stay calm", "use a friendly tone", "be concise", "think positively"]


def _between(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1) if m else None


@register_provider("mock")
class MockClient(LLMClient):
    """一个可预测的假世界：

    - 答题：能解析 "A + B" 形式的题。答对概率 = skill + 0.07 × 提示词里出现的 SKILLS 条数。
    - 被要求产出执行者规格(<<EXECUTOR_SPEC>>)：以 0.25 + 0.15 × 提示词里 META_SKILLS 条数 的概率
      给当前规格加一条有用的 SKILL，否则加一条没用的。
    - 被要求产出改进器规格(<<IMPROVER_SPEC>>)：以 0.6 的概率给 guidance 加一条 META_SKILL。

    所以在这个假世界里“改进器被改过之后确实更会改进”，各项实验应当能测出来。
    """

    def complete(self, prompt, *, system=None, max_tokens=4096, sample=0):
        full = (system or "") + "\n" + prompt
        seed = hashlib.sha256(f"{self.name}|{self.spec.get('seed', 0)}|{sample}|{full}".encode()).hexdigest()
        rng = random.Random(seed)
        if "<<EXECUTOR_SPEC>>" in prompt:
            text = self._executor_spec(prompt, rng)
        elif "<<IMPROVER_SPEC>>" in prompt:
            text = self._improver_spec(prompt, rng)
        else:
            text = self._answer(full, rng)
        usage = Usage(max(1, len(full) // 4), max(1, len(text) // 4) + 20)
        return Completion(text, usage, self.model, "end_turn")

    def _answer(self, full, rng):
        nums = re.findall(r"What is (-?\d+) \+ (-?\d+)\?", full)
        if not nums:
            return "ANSWER: A"
        a, b = map(int, nums[-1])
        p = float(self.spec.get("skill", 0.4)) + 0.07 * sum(s in full for s in SKILLS)
        ans = a + b if rng.random() < min(p, 0.95) else a + b + rng.choice([-10, -1, 1, 10])
        return f"Let me compute.\nANSWER: {ans}"

    def _executor_spec(self, prompt, rng):
        cur = json.loads(_between(prompt, "current_executor") or "{}")
        ins = list(cur.get("instructions", []))
        guidance = _between(prompt, "guidance") or ""
        p_good = 0.25 + 0.15 * sum(m in guidance for m in META_SKILLS)
        unused = [s for s in SKILLS if s not in ins]
        if unused and rng.random() < min(p_good, 0.9):
            ins.append(rng.choice(unused))
        else:
            ins.append(rng.choice(_FILLERS))
        return "```json\n" + json.dumps({**cur, "instructions": ins}) + "\n```"

    def _improver_spec(self, prompt, rng):
        cur = json.loads(_between(prompt, "current_improver") or "{}")
        g = list(cur.get("guidance", []))
        unused = [m for m in META_SKILLS if m not in g]
        if unused and rng.random() < 0.6:
            g.append(rng.choice(unused))
        else:
            g.append(rng.choice(_FILLERS))
        return json.dumps({**cur, "guidance": g})


def extract_json(text: str) -> dict | None:
    """从模型输出里稳健地取出第一个 JSON 对象。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = [m.group(1)] if m else []
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        cands.append(text[s : e + 1])
    for c in cands:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
