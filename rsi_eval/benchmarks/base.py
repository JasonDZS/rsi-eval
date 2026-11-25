from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core import ProtocolViolation

SPLITS = ("search", "select", "report")


@dataclass(frozen=True)
class Task:
    id: str
    input: str
    target: Any
    meta: dict = field(default_factory=dict, hash=False, compare=False)


class Benchmark(ABC):
    name: str = ""
    kind: str = "qa"  # qa | code
    #: 追加在题目后面的输出格式要求，执行者和改进器都能看到
    instructions: str = ""
    #: True 表示评分由框架适配器调用上游评测脚本完成（如 SWE-bench 的 docker harness）
    adapter_scored: bool = False

    def __init__(self, cfg: dict, data_dir: Path):
        self.cfg = cfg  # protocol.yaml 里 benchmarks.<name> 的配置
        self.data_dir = data_dir
        self._splits: dict | None = None

    # ---- 子类实现 ----
    @abstractmethod
    def load(self) -> list[Task]: ...

    @abstractmethod
    def score(self, task: Task, output: str) -> float: ...

    def normalize(self, output: str) -> str:
        """把输出规整成可投票的答案（重复采样对照用）。"""
        return output.strip()

    def prepare(self) -> list[dict]:
        """从公开来源取数据，返回 jsonl 行。由 `rsi-eval data prepare` 调用。"""
        raise NotImplementedError(f"{self.name}: no prepare(); put data at {self.data_path} manually")

    # ---- 通用 ----
    @property
    def data_path(self) -> Path:
        return self.data_dir / f"{self.name}.jsonl"

    def load_jsonl(self) -> list[Task]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"{self.data_path} missing. Run: rsi-eval data prepare {self.name}")
        rows = [json.loads(l) for l in self.data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [Task(str(r["id"]), r["input"], r["target"], r.get("meta", {})) for r in rows]

    def splits(self) -> dict[str, list[Task]]:
        """确定性地切成 搜索 / 选择 / 报告 三份。切分只由 split_seed 和题目 id 决定。"""
        if self._splits is None:
            tasks = sorted(self.load(), key=lambda t: hashlib.sha256(f"{self.cfg.get('split_seed', 0)}|{t.id}".encode()).hexdigest())
            sizes = self.cfg.get("sizes")  # {search: n, select: n, report: n}，优先
            if sizes:
                need = sum(sizes[s] for s in SPLITS)
                if need > len(tasks):
                    raise ValueError(f"{self.name}: sizes need {need} tasks, only {len(tasks)} available")
                bounds, acc = {}, 0
                for s in SPLITS:
                    bounds[s] = (acc, acc + sizes[s])
                    acc += sizes[s]
            else:
                fr = self.cfg.get("fractions", {"search": 0.3, "select": 0.2, "report": 0.5})
                n, acc, bounds = len(tasks), 0, {}
                for s in SPLITS:
                    k = int(n * fr[s])
                    bounds[s] = (acc, acc + k)
                    acc += k
            self._splits = {s: tasks[a:b] for s, (a, b) in bounds.items()}
        return self._splits

    def tasks(self, split: str, phase: str) -> list[Task]:
        """按阶段取题。报告集只有 report 阶段能读——这是协议的硬约束。"""
        if split == "report" and phase != "report":
            raise ProtocolViolation(f"{self.name}: report split requested during phase '{phase}'")
        if split != "report" and phase == "report":
            raise ProtocolViolation(f"{self.name}: split '{split}' requested during report phase")
        return self.splits()[split]

    def vote(self, outputs: list[str]) -> str:
        """多数投票，平票取先出现的。"""
        norm = [self.normalize(o) for o in outputs]
        best = Counter(norm).most_common(1)[0][0]
        return outputs[norm.index(best)]


BENCHMARKS: dict[str, type[Benchmark]] = {}


def register(cls):
    BENCHMARKS[cls.name] = cls
    return cls


def get_benchmark(name: str, protocol: dict, data_dir: Path) -> Benchmark:
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark '{name}' (have: {sorted(BENCHMARKS)})")
    cfg = {**protocol.get("split_defaults", {}), **protocol.get("benchmarks", {}).get(name, {})}
    return BENCHMARKS[name](cfg, data_dir)


def shuffled(items: list, seed: str) -> list:
    out = list(items)
    random.Random(seed).shuffle(out)
    return out
