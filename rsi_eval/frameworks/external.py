"""上游框架的接入层：公共部分，以及还没接入的 HyperAgents、HGM 的骨架。

已接入：ADAS（adas.py）、Gödel Agent（godel.py）。两者都固定了上游 commit，
用 `rsi-eval vendor fetch` 拉到 vendor/<名字>/，运行时核对 HEAD 和固定的 commit 一致。

公共部分：拉取并固定上游版本、在计量代理后面运行子进程（run / run_driver）、把目录冻结成产物。
HyperAgents、HGM 目前会抛 NotWired 并说明要接什么，仓库地址是按论文公开信息填的，还没有实际拉取核对过。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from ..budget import BudgetExhausted, SpendCapExceeded
from ..core import Artifact, Ctx
from ..proxy import MeteringProxy
from .base import FrameworkAdapter, SystemState, TaskResult, register


class NotWired(NotImplementedError):
    """这个上游框架的适配器还没接好，或者上游代码没有就位。"""


class UpstreamRefused(RuntimeError):
    """模型大量拒答上游框架的提示词，继续跑没有意义。"""


def repo_head(d: Path) -> str:
    return subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def fetch_repo(url: str, commit: str, dst: Path) -> str:
    """把上游仓库拉到 dst 并检出固定的 commit。已经在那个 commit 上就什么都不做。"""
    if dst.exists() and repo_head(dst) == commit:
        return "already pinned"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(dst)], check=True)
    else:
        subprocess.run(["git", "-C", str(dst), "fetch", "--quiet", "origin"], check=True)
    subprocess.run(["git", "-C", str(dst), "-c", "advice.detachedHead=false", "checkout", "--quiet", commit], check=True)
    return "fetched"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class ExternalRepoAdapter(FrameworkAdapter):
    repo_url: str = ""
    #: 接入时把上游 commit 固定在这里，保证各次运行用的是同一份上游代码
    commit: str = ""
    #: 上游用哪家接口
    upstream_api: str = "https://api.anthropic.com"

    # ---- 公共能力 ----
    _verified: set = set()
    _verify_lock = threading.Lock()

    def repo_dir(self, ctx: Ctx | None = None) -> Path:
        if self.cfg.get("repo_dir"):
            return Path(self.cfg["repo_dir"]).resolve()
        root = ctx.settings.root if ctx is not None else Path(".")
        return (Path(root) / "vendor" / self.name).resolve()

    def ensure_repo(self, ctx: Ctx | None = None) -> Path:
        """上游代码必须在 vendor/<名字>/，且 HEAD 就是固定的 commit（cfg.allow_unpinned 可跳过核对，结果里会记下实际 commit）。"""
        d = self.repo_dir(ctx)
        if not d.exists():
            raise NotWired(f"{self.name}: 上游代码不在 {d}。先运行 `rsi-eval vendor fetch {self.name}`。")
        key = (str(d), self.commit)
        with self._verify_lock:
            if self.commit and key not in self._verified and not self.cfg.get("allow_unpinned"):
                head = repo_head(d)
                if head != self.commit:
                    raise NotWired(f"{self.name}: {d} 的 HEAD 是 {head}，固定的 commit 是 {self.commit}。运行 `rsi-eval vendor fetch {self.name}`。")
                dirty = subprocess.run(["git", "-C", str(d), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
                if dirty:
                    raise NotWired(f"{self.name}: {d} 里的上游文件被改过：\n{dirty}")
                self._verified.add(key)
        return d

    def workspace(self, ctx: Ctx, tag: str) -> Path:
        """给一次运行准备一份干净的上游代码副本，避免各次运行互相污染。"""
        dst = ctx.settings.run_dir / "work" / self.name / tag
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.ensure_repo(ctx), dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        return dst

    def _upstream(self, ctx: Ctx) -> str:
        """代理把请求转给谁。两个角色都是假模型时走 mock（离线测试）。"""
        provs = {ctx.settings.models[ctx.roles[r]].get("provider") for r in ("executor", "improver") if r in ctx.roles}
        return "mock" if provs == {"mock"} else self.cfg.get("upstream_api", self.upstream_api)

    def run(self, cmd: list[str], ctx: Ctx, cwd: Path, env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
        """在计量代理后面跑上游命令。上游的所有模型调用都会记到 ctx 的 Meter 上；预算用完抛 BudgetExhausted。"""
        with MeteringProxy(ctx, self._upstream(ctx)) as px:
            p = subprocess.run(cmd, cwd=cwd, env=px.child_env(extra=env), capture_output=True, text=True, timeout=timeout)
            if px.cap_exceeded:
                raise SpendCapExceeded(f"{self.name}: suite hard cap reached inside upstream process")
            if px.exhausted:
                raise BudgetExhausted(f"{self.name}: budget exhausted inside upstream process")
        if p.returncode != 0:
            raise RuntimeError(f"{self.name}: `{' '.join(cmd)}` failed ({p.returncode}):\n{p.stderr[-2000:]}")
        return p

    def run_driver(self, module: str, command: str, payload: dict, ctx: Ctx, cwd: Path, *, timeout: float, env: dict | None = None,
                   log_to: Path | None = None) -> dict:
        """在计量代理后面跑一个驱动脚本（rsi_eval/frameworks/drivers/ 下的模块），JSON 进、JSON 出。

        驱动脚本在子进程里导入上游代码并调用它，模型生成的代码也只在这个子进程里执行。
        子进程的环境是白名单，拿不到真实密钥；它的所有模型调用都经过代理按角色入账。
        返回驱动写出的结果；子进程超时或崩溃时返回 {"error": ...}，由适配器决定怎么处理。
        预算用完抛 BudgetExhausted，套件硬上限到了抛 SpendCapExceeded。
        """
        pkg_root = str(Path(__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory(prefix="rsi-driver-") as io_dir:
            fin, fout = Path(io_dir) / "in.json", Path(io_dir) / "out.json"
            fin.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cmd = [sys.executable, "-m", module, command, "--in", str(fin), "--out", str(fout)]
            with MeteringProxy(ctx, self._upstream(ctx)) as px:
                child_env = px.child_env(extra={"PYTHONPATH": pkg_root, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}, minimal=True)
                err = None
                try:
                    p = subprocess.run(cmd, cwd=cwd, env=child_env, capture_output=True, text=True, errors="replace", timeout=timeout)
                    stdout, stderr = p.stdout, p.stderr
                except subprocess.TimeoutExpired as e:
                    p = None
                    dec = lambda b: b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")  # noqa: E731
                    stdout, stderr, err = dec(e.stdout), dec(e.stderr), f"timeout after {timeout}s"
                if log_to is not None:
                    log_to.parent.mkdir(parents=True, exist_ok=True)
                    log_to.write_text(f"$ {' '.join(cmd)}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n", encoding="utf-8")
                if px.cap_exceeded:
                    raise SpendCapExceeded(f"{self.name}: suite hard cap reached inside upstream process")
                if px.exhausted:
                    raise BudgetExhausted(f"{self.name}: budget exhausted inside upstream process")
                stats = {"calls": dict(px.calls), "refusals": dict(px.refusals)}
            self._note_refusals(ctx, stats)
            out = None
            if fout.exists():
                try:
                    out = json.loads(fout.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    err = err or "driver wrote unreadable output"
            if not isinstance(out, dict):
                out = {"error": err or f"driver exited with {p.returncode} and no output: {stderr[-1500:]}"}
            out["_llm"] = stats
            return out

    # ---- 拒答 ----
    _refusal_lock = threading.Lock()

    def _note_refusals(self, ctx: Ctx, stats: dict) -> None:
        """上游代码通常把空回复吞掉当成答错，拒答多了分数全是 0 却看不出原因。
        这里按 模型 累计：调用满 min_calls 次且拒答率超过 abort_refusal_rate 就中止，免得把预算花在没有意义的运行上。"""
        limit = float(self.cfg.get("abort_refusal_rate", 0.5))
        with self._refusal_lock:
            seen = self.__dict__.setdefault("_refusal_seen", {})
            for role, n in stats["calls"].items():
                model = ctx.roles.get(role, role)
                c = seen.setdefault(model, [0, 0])
                c[0] += n
                c[1] += stats["refusals"].get(role, 0)
                if c[0] >= int(self.cfg.get("refusal_min_calls", 30)) and c[1] / c[0] > limit:
                    raise UpstreamRefused(
                        f"{self.name}: 模型 {model} 对上游框架的提示词拒答了 {c[1]}/{c[0]} 次（流水里 refusal=true 的行）。"
                        f"分数会全是 0，已中止。换一个不拒答的模型，或把 frameworks_cfg.{self.name}.abort_refusal_rate 调到 1 强行继续。")

    def freeze(self, ctx: Ctx, kind: str, path: Path) -> Artifact:
        return ctx.store.put_dir(kind, path)

    # ---- 待接入 ----
    todo = ""

    def _not_wired(self, what: str):
        raise NotWired(f"{self.name}.{what} 还没接入上游代码。{self.todo} 见 docs/adapters.md。")

    def initial(self, ctx, bench) -> SystemState:
        self._not_wired("initial")

    def step(self, state, ctx, bench, tasks, *, modify_improver) -> SystemState:
        self._not_wired("step")

    def solve(self, executor, bench, task, ctx, sample) -> str:
        self._not_wired("solve")

    def candidates(self, state, k):
        self._not_wired("candidates")


#: 上游子进程出错时 solve() 返回的输出以它开头，评分为 0；演化时适配器据此把错误反馈给改进器。
ERROR_PREFIX = "[upstream-error] "


class DriverAdapter(ExternalRepoAdapter):
    """用驱动脚本接入的上游框架的公共部分：答题、评分、取候选。"""

    driver = ""

    def candidates(self, state, k):
        entries = sorted(state.archive.load_json()["entries"], key=lambda e: -e["search_score"])
        seen, out = set(), []
        for e in entries:
            if e["executor"]["digest"] not in seen:
                seen.add(e["executor"]["digest"])
                out.append(Artifact.from_dict(e["executor"]))
        return out[:k]

    def run_task(self, executor, bench, task, ctx, sample) -> TaskResult:
        out = self.solve(executor, bench, task, ctx, sample)
        return TaskResult(out, 0.0 if out.startswith(ERROR_PREFIX) else bench.score(task, out))

    @staticmethod
    def as_output(res: dict) -> str:
        if "answer" in res:
            return f"ANSWER: {res['answer']}"
        return ERROR_PREFIX + str(res.get("error", "no answer"))[-2000:]


@register
class HyperAgents(ExternalRepoAdapter):
    name = "hyperagents"
    level = "L2"
    repo_url = "https://github.com/facebookresearch/HyperAgents"
    todo = ("改进器 = 元 agent 代码，执行者 = 任务 agent 代码，两者在同一程序里，按文件路径拆开冻结；"
            "modify_improver=False 用上游自带的“去掉自改进”消融；Polyglot / SWE-bench 的评分调用上游的 docker 评测脚本（重写 run_task）；"
            "improve_improver 对应让元 agent 只改元 agent 的一轮。")


@register
class HGM(ExternalRepoAdapter):
    name = "hgm"
    level = "L2"
    repo_url = "https://github.com/metauto-ai/HGM"
    todo = ("编码 agent 改自己，改进器和执行者是同一份代码：改进器 = 第 k 代的代码快照，执行者 = 用于答题的那份快照；"
            "同起点对照时用第 k 代快照去改第 0 代快照；modify_improver=False 时每轮结束把改进器目录还原。")
