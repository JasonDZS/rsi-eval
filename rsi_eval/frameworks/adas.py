"""ADAS（Automated Design of Agentic Systems）适配器。L1：元 agent 的提示词固定，只累积存档。

上游：https://github.com/ShengranHu/ADAS ，固定在 ADAS.commit。上游按领域分目录，每个目录一个 search.py。

映射
- 改进器 = 上游该领域的元 agent 提示词文件（<领域>_prompt.py）加搜索参数。内容固定，永远不变。
- 执行者 = 一个 agent：{name, thought, code}，code 是上游约定的 `def forward(self, taskInfo)`。
  第 0 代执行者是上游种子库里的第一个（Chain-of-Thought）。
- 存档 = 上游的 archive：种子库 7 个 agent 加上每一轮新生成的，各带搜索集分数。

一轮 step = 上游 search() 循环体的一次迭代：
  首轮先把种子库在搜索集上评一遍（上游也是这么做的）→ 元 agent 提案 + 两次反思
  → 新 agent 在搜索集上评测 → 报错或全错就把错误贴回去让元 agent 改，最多 debug_max 次 → 入档。

和上游不一样的地方（都是协议要求的）
- 评测题目是协议传进来的搜索集，不是上游自己切的验证集；评分用本仓库的规则评分，不用上游的。
- 模型是协议配置的角色模型，经计量代理入账；上游代码里的 gpt 型号不起作用。
- 上游一轮失败时不计数、重来；这里失败也算一轮，保证预算内一定会停。
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from ..core import Artifact
from ..evaluation import evaluate
from .base import SystemState, register
from .external import ERROR_PREFIX, DriverAdapter

#: 协议里的基准 -> 上游的领域目录。未见任务（GSM8K、GSM-Hard）用数学领域的提示词，上游的迁移实验也是这么做的。
DOMAINS = {"drop": "_drop", "mgsm": "_mgsm", "mmlu": "_mmlu", "gpqa": "_gpqa", "gsm8k": "_mgsm", "gsm_hard": "_mgsm"}


@register
class ADAS(DriverAdapter):
    name = "adas"
    level = "L1"
    repo_url = "https://github.com/ShengranHu/ADAS"
    commit = "2702bee8fefda42255efc5be9f60e3bd3db96ae4"  # 2025-01-27 "fix role description typo"
    upstream_api = "https://api.openai.com"
    driver = "rsi_eval.frameworks.drivers.adas_driver"

    # ---- 上游对象 ----
    def domain(self, bench) -> str:
        d = self.cfg.get("domains", {}).get(bench.name) or DOMAINS.get(bench.name) or self.cfg.get("default_domain")
        if not d:
            raise KeyError(f"adas: 基准 {bench.name} 没有对应的上游领域目录，在套件的 frameworks_cfg.adas.domains 里指定")
        return d

    def _seeds(self, ctx, domain: str) -> list[dict]:
        """上游种子库。提示词文件只有常量和纯函数，直接在本进程里读（不执行模型生成的代码）。"""
        path = self.ensure_repo(ctx) / domain / f"{domain.lstrip('_')}_prompt.py"
        ns: dict = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        return [{"name": s["name"], "thought": s["thought"], "code": s["code"]} for s in ns["get_init_archive"]()]

    def _executor(self, ctx, domain: str, sol: dict) -> Artifact:
        return ctx.store.put_json("executor", {"framework": self.name, "domain": domain, "name": str(sol.get("name", "")),
                                               "thought": str(sol.get("thought", "")), "code": str(sol.get("code", ""))})

    def _drive(self, command: str, payload: dict, ctx, domain: str, timeout: float) -> dict:
        with tempfile.TemporaryDirectory(prefix="rsi-adas-") as cwd:  # 模型写的代码在一次性目录里跑
            return self.run_driver(self.driver, command, {"upstream": str(self.ensure_repo(ctx)), "domain": domain, **payload}, ctx, Path(cwd), timeout=timeout)

    # ---- 状态 ----
    def initial(self, ctx, bench):
        domain = self.domain(bench)
        prompt = self.ensure_repo(ctx) / domain / f"{domain.lstrip('_')}_prompt.py"
        improver = ctx.store.put_json("improver", {
            "framework": self.name, "commit": self.commit, "domain": domain,
            "meta_prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
            "debug_max": int(self.cfg.get("debug_max", 3)),
        })
        seeds = self._seeds(ctx, domain)
        archive = ctx.store.put_json("archive", {"domain": domain, "entries": [], "history": []})
        return SystemState(improver, self._executor(ctx, domain, seeds[0]), archive)

    # ---- 执行者 ----
    def solve(self, executor, bench, task, ctx, sample):
        spec = executor.load_json()
        res = self._drive("solve", {"code": spec["code"], "input": task.input}, ctx, spec["domain"], float(self.cfg.get("solve_timeout", 900)))
        return self.as_output(res)

    # ---- 一轮演化 ----
    def step(self, state, ctx, bench, tasks, *, modify_improver):
        imp = state.improver.load_json()
        arch = state.archive.load_json()
        domain = imp["domain"]
        batch = tasks[: int(self.cfg.get("eval_batch", len(tasks)))]

        def run_eval(sol: dict):
            ex = self._executor(ctx, domain, sol)
            r = evaluate(self, ex, bench, batch, ctx)
            errors = [o[len(ERROR_PREFIX):] for o in r.outputs.values() if o.startswith(ERROR_PREFIX)]
            return ex, r, errors

        def entry(sol: dict, ex: Artifact, r, generation) -> dict:
            return {"executor": ex.to_dict(), "search_score": r.mean, "generation": generation,
                    "solution": {**sol, "generation": generation, "acc_list": [r.scores[t.id] for t in batch]}}

        # 上游：先把种子库里还没分数的都评一遍
        if not arch["entries"]:
            for sol in self._seeds(ctx, domain):
                ex, r, errors = run_eval(sol)
                if errors:  # 上游：种子出错就跳过它
                    continue
                arch["entries"].append(entry(sol, ex, r, "initial"))

        gen = state.generation + 1
        hist = {"generation": gen, "added": False, "debug_rounds": 0, "error": None}
        hist["meta_refusals"] = 0
        res = self._drive("propose", {
            "archive": [dict(e["solution"]) for e in arch["entries"]], "generation": state.generation,
            "score_scale": 100.0 if domain == "_drop" else 1.0,
        }, ctx, domain, float(self.cfg.get("propose_timeout", 1800)))
        hist["meta_refusals"] += res["_llm"]["refusals"].get("improver", 0)
        for e, f in zip(arch["entries"], res.get("fitness", [])):  # 分数字符串算一次就记下，之后各轮给元 agent 看的是同一份
            e["solution"]["fitness"] = f
            e["solution"].pop("acc_list", None)

        sol, msgs = res.get("solution"), res.get("msgs")
        if not isinstance(sol, dict) or "code" not in sol:
            hist["error"] = str(res.get("error") or "meta agent returned no code")[-500:]
        else:
            for attempt in range(int(imp.get("debug_max", 3))):
                ex, r, errors = run_eval(sol)
                problem = errors[0] if errors else ("All 0 accuracy" if r.mean < 0.01 else None)
                if problem is None:
                    clean = {k: v for k, v in sol.items() if k not in ("debug_thought", "reflection")}
                    arch["entries"].append(entry(clean, ex, r, gen))
                    hist.update(added=True, name=str(sol.get("name", "")), search_score=r.mean)
                    break
                hist.update(debug_rounds=attempt + 1, error=problem[-500:])
                if attempt == int(imp.get("debug_max", 3)) - 1:
                    break
                fix = self._drive("debug", {"msgs": msgs, "solution": sol, "error": problem}, ctx, domain, float(self.cfg.get("propose_timeout", 1800)))
                hist["meta_refusals"] += fix["_llm"]["refusals"].get("improver", 0)
                msgs = fix.get("msgs", msgs)
                if isinstance(fix.get("solution"), dict) and "code" in fix["solution"]:
                    sol = fix["solution"]
        arch["history"].append(hist)

        best = max(arch["entries"], key=lambda e: e["search_score"], default=None)
        executor = Artifact.from_dict(best["executor"]) if best else state.executor
        return SystemState(state.improver, executor, ctx.store.put_json("archive", arch), gen)
