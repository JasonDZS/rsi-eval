"""代码类基准：Polyglot（演化用），SWE-bench Verified 子集、LiveCodeBench 新时间窗（未见任务）。

评分都靠可执行测试。Polyglot 和 SWE-bench 的评测需要各语言工具链 / 官方 docker 镜像，
HyperAgents 和 HGM 上游仓库各自带了评测脚本，所以这两个基准标记为 adapter_scored：
由框架适配器调用上游脚本得到每题通过与否，本框架只负责切分、记账和统计。

LiveCodeBench 的标准输入输出题在这里直接评分（子进程 + 超时；设置
RSI_EVAL_SANDBOX=docker 时放进无网络的容器里跑）。

数据集 id 和字段名还没有在这台机器上实际拉取验证过。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from .base import Benchmark, Task, register


def extract_code(output: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", output or "", re.S)
    return blocks[-1] if blocks else (output or "")


def run_python(code: str, stdin: str, timeout: float = 10.0) -> tuple[int, str]:
    """运行模型生成的代码。默认子进程；RSI_EVAL_SANDBOX=docker 时用无网络容器。"""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        if os.environ.get("RSI_EVAL_SANDBOX") == "docker":
            cmd = ["docker", "run", "--rm", "-i", "--network", "none", "--memory", "1g", "-v", f"{d}:/w:ro",
                   os.environ.get("RSI_EVAL_SANDBOX_IMAGE", "python:3.12-slim"), "python", "/w/sol.py"]
        else:
            cmd = [sys.executable, "-I", path]
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout, cwd=d)
            return p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            return -1, ""


@register
class LiveCodeBench(Benchmark):
    name = "livecodebench"
    kind = "code"
    instructions = "Write a complete Python 3 program that reads from standard input and writes to standard output. Put it in one ```python code block."

    def load(self):
        return self.load_jsonl()

    def normalize(self, output):
        return extract_code(output).strip()

    def score(self, task: Task, output: str) -> float:
        code = extract_code(output)
        for t in task.target:  # [{"input": ..., "output": ...}]
            rc, out = run_python(code, t["input"], float(self.cfg.get("timeout", 10)))
            if rc != 0 or out.split() != t["output"].split():
                return 0.0
        return 1.0

    def prepare(self):
        from datasets import load_dataset

        ds = load_dataset("livecodebench/code_generation_lite", version_tag=self.cfg.get("version_tag", "release_latest"), split="test", trust_remote_code=True)
        since = self.cfg.get("since", "2025-01-01")  # 只取较新时间窗的题，降低数据污染
        rows = []
        for r in ds:
            if str(r["contest_date"])[:10] < since:
                continue
            tests = [t for t in json.loads(r["public_test_cases"]) if t.get("testtype") == "stdin"]
            if not tests:
                continue
            rows.append({
                "id": f"lcb-{r['question_id']}",
                "input": r["question_content"],
                "target": [{"input": t["input"], "output": t["output"]} for t in tests],
                "meta": {"date": str(r["contest_date"])[:10], "difficulty": r.get("difficulty")},
            })
        return rows


class _AdapterScored(Benchmark):
    kind = "code"
    adapter_scored = True

    def load(self):
        return self.load_jsonl()

    def score(self, task, output):
        raise RuntimeError(f"{self.name} is scored by the framework adapter's upstream harness, not by Benchmark.score()")


@register
class Polyglot(_AdapterScored):
    """Aider Polyglot（Exercism 多语言题）。data/polyglot.jsonl 每行 {id, input, target: null, meta: {language, exercise}}。"""

    name = "polyglot"


@register
class SWEBenchVerified(_AdapterScored):
    """SWE-bench Verified 子集。子集的 instance_id 列表在 protocol.yaml 里固定。"""

    name = "swebench_verified"

    def prepare(self):
        from datasets import load_dataset

        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        keep = set(self.cfg.get("instance_ids") or [])
        return [
            {"id": r["instance_id"], "input": r["problem_statement"], "target": None, "meta": {"repo": r["repo"], "base_commit": r["base_commit"]}}
            for r in ds
            if not keep or r["instance_id"] in keep
        ]
