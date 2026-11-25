"""问答类基准：DROP、MGSM、MMLU、GPQA（演化用），GSM8K、GSM-Hard（未见任务）。

全部规则评分，不依赖 LLM 裁判。执行者的输出里最后一个 `ANSWER: xxx` 是答案。

数据来源：ADAS 上游仓库里自带的数据文件（vendor/adas/dataset/，Gödel Agent 仓库里是同一批文件），
也就是两篇论文实际用的那一份。prepare() 读取前逐个核对文件的 sha256（见 SOURCES），对不上就拒绝。
题目的文字格式沿用 ADAS 上游的写法（MGSM 的各语言指令、DROP 的单例提示、选择题的 (A)(B)(C)(D)），
这样上游框架拿到的输入和它们原本看到的一致。
GPQA 用的是上游自带的 diamond 子集（198 题）。它是受限数据集，导出的 jsonl 不进仓库，也不要公开转发。
"""
from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

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


#: 数据文件的版本：ADAS 上游固定 commit 里的路径 -> sha256。prepare() 核对后才读。
ADAS_COMMIT = "2702bee8fefda42255efc5be9f60e3bd3db96ae4"
SOURCES = {
    "dataset/drop_v0_dev.jsonl.gz": "7a58d35552dc476699d87ee8af0254f63e9f00f5a7f6b8b033e11c933157e186",
    "dataset/mmlu.csv": "15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb",
    "dataset/gpqa_diamond.csv": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
    "dataset/GSM8k_test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "dataset/gsmhardv2.jsonl": "2032b0188a49ac0ec38cbe1e695e7d62d6dddc44bbb64f920728aca710e9ddef",
    "dataset/mgsm/mgsm_bn.tsv": "6b00bc7cc635547e866989284afa924d789b5affa2eb8de623c385dd943ad977",
    "dataset/mgsm/mgsm_de.tsv": "4dfea30fede44b813e2e496f5f0534049e83c7cc8aed6e00600b30ec62053626",
    "dataset/mgsm/mgsm_en.tsv": "50021d0f28cc957edcb44e7806425b1c7fbd648ddcb9e0a8ec689d10e57d40fa",
    "dataset/mgsm/mgsm_es.tsv": "5bd27ebdf00140cec845c5298dc17715f5cd57d8edda6df1976db40bf3f0750a",
    "dataset/mgsm/mgsm_fr.tsv": "36207c1c03fd7cd3ea491441eb755408199f94e4a647851df531bbaedc99d606",
    "dataset/mgsm/mgsm_ja.tsv": "59a2b50debe77981fd784cb3b2bef1505e3abf2a37116dc9d7a366ab029b4637",
    "dataset/mgsm/mgsm_ru.tsv": "6bd30fd2e80c5bac23f566fb4ae0e6a55a19578401fbf9a01fb21beb3645fef8",
    "dataset/mgsm/mgsm_sw.tsv": "2bac828d77229e65d7c7197b1ad4a2cb5b1fe99b163f2cbdd66501de6a2115c4",
    "dataset/mgsm/mgsm_te.tsv": "dd6b1452c244bb2e4ba254c01ea84137f80c1cefc57d69c23b0ed88b9c0f36b7",
    "dataset/mgsm/mgsm_th.tsv": "f3932dc5ad8e9d0ea82b017adc1e1461dd647af861e7166d6741986602a0cfd6",
    "dataset/mgsm/mgsm_zh.tsv": "b2fa63151022370a0de1f4211c8c284eae74b0f5a3b003b1d5982c0d4a73f661",
}


def _upstream_module(repo: Path, rel: str):
    """按路径加载 ADAS 上游的一个工具模块（只用其中的题目格式化函数，不执行别的）。"""
    spec = importlib.util.spec_from_file_location("adas_" + re.sub(r"\W", "_", rel), repo / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _JsonlQA(Benchmark):
    #: 这个基准用到的上游数据文件
    files: tuple = ()

    def load(self) -> list[Task]:
        return self.load_jsonl()

    @property
    def repo(self) -> Path:
        return Path(self.cfg.get("adas_dir", self.data_dir.parent / "vendor" / "adas"))

    def source(self, rel: str) -> Path:
        """上游数据文件的路径；sha256 和 SOURCES 里固定的不一致就拒绝。"""
        p = self.repo / rel
        if not p.exists():
            raise FileNotFoundError(f"{self.name}: 找不到 {p}。先运行 `rsi-eval vendor fetch adas`。")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != SOURCES[rel]:
            raise ValueError(f"{self.name}: {rel} 的 sha256 是 {got}，固定的版本是 {SOURCES[rel]}")
        return p

    def provenance(self) -> dict:
        return {"repo": "https://github.com/ShengranHu/ADAS", "commit": ADAS_COMMIT, "files": {f: SOURCES[f] for f in self.files}}


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
        ans = extract_answer(output)
        # 先找单独成词的大写选项字母（"B"、"(B)"、"B) Paris"、"The answer is C"），找不到再退回到任意位置的 A-D
        m = re.search(r"(?<![A-Za-z])[A-D](?![A-Za-z])", ans) or re.search(r"[A-D]", ans.upper())
        return m.group(0) if m else ""

    def score(self, task, output):
        return float(self.normalize(output) == task.target)


@register
class GSM8K(_Numeric):
    name = "gsm8k"
    files = ("dataset/GSM8k_test.jsonl",)

    def prepare(self):
        lines = self.source(self.files[0]).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(l) for l in lines if l.strip()]
        return [{"id": f"gsm8k-{i}", "input": "Solve this math problem:\n" + r["question"], "target": to_number(r["answer"].split("####")[-1])}
                for i, r in enumerate(rows)]


@register
class GSMHard(_Numeric):
    name = "gsm_hard"
    files = ("dataset/gsmhardv2.jsonl",)

    def prepare(self):
        lines = self.source(self.files[0]).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(l) for l in lines if l.strip()]
        return [{"id": f"gsmhard-{i}", "input": "Solve this math problem:\n" + r["input"], "target": float(r["target"])} for i, r in enumerate(rows)]


@register
class MGSM(_Numeric):
    name = "mgsm"
    LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
    files = tuple(f"dataset/mgsm/mgsm_{l}.tsv" for l in LANGS)

    def prepare(self):
        for f in self.files:
            self.source(f)
        utils = _upstream_module(self.repo, "_mgsm/utils.py")
        rows = []
        with _cwd(self.repo):  # 上游用相对路径读数据
            for lang in self.cfg.get("langs", self.LANGS):
                for i, ex in enumerate(utils.get_lang_examples(lang)):
                    rows.append({"id": f"mgsm-{lang}-{i}", "input": ex["inputs"], "target": float(ex["targets"].replace(",", "")), "meta": {"lang": lang}})
        return rows


@register
class MMLU(_Choice):
    name = "mmlu"
    files = ("dataset/mmlu.csv",)

    def prepare(self):
        fmt = _upstream_module(self.repo, "_mmlu/utils.py").format_multichoice_question
        with open(self.source(self.files[0]), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [{"id": f"mmlu-{i}", "input": fmt(r), "target": r["Answer"], "meta": {"subject": r["Subject"]}} for i, r in enumerate(rows)]


@register
class GPQA(_Choice):
    name = "gpqa"
    files = ("dataset/gpqa_diamond.csv",)

    def prepare(self):
        with open(self.source(self.files[0]), newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        rows = []
        for i, r in enumerate(data):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = shuffled(list(range(4)), f"gpqa-{i}")  # 选项顺序由题号决定，每次导出都一样
            a, b, c, d = (opts[j].strip() for j in order)
            rows.append({"id": f"gpqa-{i}", "target": "ABCD"[order.index(0)], "meta": {"domain": r.get("High-level domain", "")},
                         "input": f"What is the correct answer to this question: {r['Question'].strip()}\n\nChoices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}"})
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
    files = ("dataset/drop_v0_dev.jsonl.gz",)
    instructions = "Finish with a line of the form `ANSWER: <short answer>`."

    def normalize(self, output):
        return " ".join(_norm_text(extract_answer(output)))

    def score(self, task, output):
        pred = extract_answer(output)
        golds = task.target if isinstance(task.target, list) else [task.target]
        return max(f1(pred, str(g)) for g in golds)

    def prepare(self):
        path = self.source(self.files[0])
        with gzip.open(path, "rb") as f:
            n = sum(1 for _ in f)
        examples = _upstream_module(self.repo, "_drop/utils.py").load_drop(str(path))
        # 上游把第一题和最后一题留作示例，不参与评测
        return [{"id": f"drop-{i}", "input": ex["inputs"], "target": [t for t in ex["targets"] if t.strip()]} for i, ex in enumerate(examples) if 0 < i < n - 1]
