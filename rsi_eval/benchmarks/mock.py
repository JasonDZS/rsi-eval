"""离线假基准，配合 MockClient 验证整条流程。不需要数据文件。"""
from __future__ import annotations

import random

from .base import Task, register
from .qa import _Numeric


class _Arith(_Numeric):
    lo, hi, n, tag = 10, 99, 400, "a"

    def load(self):
        rng = random.Random(f"mock-{self.tag}")
        out = []
        for i in range(self.n):
            a, b = rng.randint(self.lo, self.hi), rng.randint(self.lo, self.hi)
            out.append(Task(f"{self.name}-{i}", f"What is {a} + {b}?", a + b))
        return out


@register
class MockArith(_Arith):
    name = "mock_arith"


@register
class MockArithUnseen(_Arith):
    """“未见任务”：同形态、数字更大。"""

    name = "mock_arith_unseen"
    lo, hi, tag = 1000, 9999, "b"
