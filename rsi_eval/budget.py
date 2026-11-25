"""Token / USD 记账与预算。

所有 LLM 调用都经过 Meter：
- 每次调用写一行到 ledger.jsonl（实验、阶段、角色、模型、token、金额、是否缓存重放）；
- Meter 可以嵌套，子 Meter 的消耗会向上累加；
- Meter 可以带预算上限，只对 `budget_phases` 里的阶段生效（默认只限 evolve）。

阶段(phase)约定：
- evolve  演化/改进阶段的全部消耗（改进器调用 + 搜索集上的候选评测）
- select  选择集上挑候选的消耗（计入演化总成本，但不受预算上限截断）
- report  报告集 / 未见任务上的最终计分（评测成本，单独记）
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

PHASES = ("evolve", "select", "report")


class BudgetExhausted(Exception):
    """预算用完。运行器捕获后结束当前改进运行。"""


class SpendCapExceeded(Exception):
    """整个套件的真实花费超过硬上限，立即中止。"""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass
class Price:
    """美元 / 百万 token。"""

    input: float = 0.0
    output: float = 0.0

    def usd(self, usage: Usage) -> float:
        return (usage.input_tokens * self.input + usage.output_tokens * self.output) / 1e6


class Ledger:
    """追加写的调用流水，线程安全。"""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        if not self.path:
            return
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Totals:
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    # 其中由评测缓存重放的部分（计入预算，但没有真实花钱）
    cached_tokens: int = 0
    cached_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.tokens,
            "usd": round(self.usd, 6),
            "calls": self.calls,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class Meter:
    ledger: Ledger
    tags: dict = field(default_factory=dict)
    limit: float | None = None  # 预算上限；None 表示不限
    unit: str = "tokens"  # "tokens" 或 "usd"
    budget_phases: tuple = ("evolve",)
    parent: "Meter | None" = None
    hard_cap_usd: float | None = None  # 只在根 Meter 上设：真实花费硬上限
    by_phase: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def child(self, limit: float | None = None, unit: str | None = None, **tags) -> "Meter":
        return Meter(
            ledger=self.ledger,
            tags={**self.tags, **tags},
            limit=limit,
            unit=unit or self.unit,
            budget_phases=self.budget_phases,
            parent=self,
        )

    # ---- 查询 ----
    def totals(self, phases=None) -> Totals:
        out = Totals()
        with self._lock:
            for ph, t in self.by_phase.items():
                if phases is None or ph in phases:
                    out.input_tokens += t.input_tokens
                    out.output_tokens += t.output_tokens
                    out.usd += t.usd
                    out.calls += t.calls
                    out.cached_tokens += t.cached_tokens
                    out.cached_usd += t.cached_usd
        return out

    def spent(self, phases=None) -> float:
        t = self.totals(phases if phases is not None else self.budget_phases)
        return t.usd if self.unit == "usd" else t.tokens

    def remaining(self) -> float | None:
        return None if self.limit is None else self.limit - self.spent()

    def summary(self) -> dict:
        out = {ph: self.totals((ph,)).as_dict() for ph in PHASES}
        out["total"] = self.totals().as_dict()
        return out

    # ---- 记账 ----
    def check(self, phase: str) -> None:
        """调用 LLM 之前检查预算；超了就抛 BudgetExhausted。"""
        m: Meter | None = self
        while m is not None:
            if m.limit is not None and phase in m.budget_phases and m.spent() >= m.limit:
                raise BudgetExhausted(f"budget {m.limit} {m.unit} exhausted ({m.tags})")
            if m.hard_cap_usd is not None:
                t = m.totals()
                if t.usd - t.cached_usd >= m.hard_cap_usd:
                    raise SpendCapExceeded(f"real spend reached hard cap ${m.hard_cap_usd}")
            m = m.parent

    def charge(self, phase: str, role: str, model: str, usage: Usage, usd: float, cached: bool = False, refusal: bool = False) -> None:
        m: Meter | None = self
        while m is not None:
            with m._lock:
                t = m.by_phase.setdefault(phase, Totals())
                t.input_tokens += usage.input_tokens
                t.output_tokens += usage.output_tokens
                t.usd += usd
                t.calls += 1
                if cached:
                    t.cached_tokens += usage.total
                    t.cached_usd += usd
            m = m.parent
        self.ledger.write(
            {
                "ts": round(time.time(), 3),
                **self.tags,
                "phase": phase,
                "role": role,
                "model": model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "usd": round(usd, 8),
                "cached": cached,
                **({"refusal": True} if refusal else {}),  # 模型拒答（没有内容）：照常计费，流水里标出来
            }
        )
