"""问答类基准：DROP、MGSM、MMLU、GPQA（演化用），GSM8K、GSM-Hard（未见任务）。

全部规则评分，不依赖 LLM 裁判。执行者的输出里最后一个 `ANSWER: xxx` 是答案。

prepare() 里的 Hugging Face 数据集 id 和字段名是按公开数据集写的，
还没有在这台机器上实际拉取验证过；第一次 `rsi-eval data prepare` 时如有出入在这里改。
"""
from __future__ import annotations

import re
import string
from collections import Counter

from .base import Benchmark, Task, register, shuffled

_ANS = re.compile(r"ANSWER\s*[:：]\s*(.+)", re.I)


def extract_answer(output: str) -> str:
    m = _ANS.findall(output or "")
    return (m[-1] if m else (output or "").strip().splitlines()[-1] if (output or "").strip() else "").strip()


def to_number(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def _hf(path, name=None, split="test"):
    from datasets import load_dataset  # pip install rsi-eval[data]

    return load_dataset(path, name, split=split)


class _JsonlQA(Benchmark):
    def load(self) -> list[Task]:
        return self.load_jsonl()


class _Numeric(_JsonlQA):
    instructions = "Finish with a line of the form `ANSWER: <number>`."

    def normalize(self, output):
        n = to_number(extract_answer(output))
        return "" if n is None else f"{n:g}"

    def score(self, task, output):
        n = to_number(extract_answer(output))
        return float(n is not None and abs(n - float(task.target)) < 1e-4)


class _Choice(_JsonlQA):
    instructions = "Finish with a line of the form `ANSWER: <letter>` where <letter> is one of A, B, C, D."

    def normalize(self, output):
        m = re.search(r"[A-D]", extract_answer(output).upper())
        return m.group(0) if m else ""

    def score(self, task, output):
        return float(self.normalize(output) == task.target)

    @staticmethod
    def fmt(question: str, choices: list[str]) -> str:
        return question.strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))


@register
class GSM8K(_Numeric):
    name = "gsm8k"

    def prepare(self):
        ds = _hf("openai/gsm8k", "main", "test")
        return [{"id": f"gsm8k-{i}", "input": r["question"], "target": to_number(r["answer"].split("####")[-1])} for i, r in enumerate(ds)]


@register
class GSMHard(_Numeric):
    name = "gsm_hard"

    def prepare(self):
        ds = _hf("reasoning-machines/gsm-hard", None, "train")
        return [{"id": f"gsmhard-{i}", "input": r["input"], "target": float(r["target"])} for i, r in enumerate(ds)]


@register
class MGSM(_Numeric):
    name = "mgsm"
    LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

    def prepare(self):
        rows = []
        for lang in self.cfg.get("langs", self.LANGS):
            for i, r in enumerate(_hf("juletxara/mgsm", lang, "test")):
                rows.append({"id": f"mgsm-{lang}-{i}", "input": r["question"], "target": float(r["answer_number"]), "meta": {"lang": lang}})
        return rows


@register
class MMLU(_Choice):
    name = "mmlu"

    def prepare(self):
        ds = _hf("cais/mmlu", "all", "test")
        return [
            {"id": f"mmlu-{i}", "input": self.fmt(r["question"], r["choices"]), "target": "ABCD"[r["answer"]], "meta": {"subject": r["subject"]}}
            for i, r in enumerate(ds)
        ]


@register
class GPQA(_Choice):
    name = "gpqa"

    def prepare(self):
        # 受限数据集：需要先在 Hugging Face 上同意条款并设置 HF_TOKEN
        ds = _hf("Idavidrein/gpqa", "gpqa_main", "train")
        rows = []
        for i, r in enumerate(ds):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = shuffled(list(range(4)), f"gpqa-{i}")
            rows.append({"id": f"gpqa-{i}", "input": self.fmt(r["Question"], [opts[j] for j in order]), "target": "ABCD"[order.index(0)]})
        return rows


def _norm_text(s: str) -> list[str]:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def f1(pred: str, gold: str) -> float:
    p, g = _norm_text(pred), _norm_text(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


@register
class DROP(_JsonlQA):
    name = "drop"
    instructions = "Finish with a line of the form `ANSWER: <short answer>`."

    def normalize(self, output):
        return " ".join(_norm_text(extract_answer(output)))

    def score(self, task, output):
        pred = extract_answer(output)
        golds = task.target if isinstance(task.target, list) else [task.target]
        return max(f1(pred, str(g)) for g in golds)

    def prepare(self):
        ds = _hf("ucinlp/drop", None, "validation")
        rows = []
        for i, r in enumerate(ds):
            spans = list(dict.fromkeys(r["answers_spans"]["spans"]))
            rows.append({"id": f"drop-{i}", "input": f"Passage: {r['passage']}\n\nQuestion: {r['question']}", "target": spans})
        return rows
