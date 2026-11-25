"""上游框架的接入层：Gödel Agent、ADAS、HyperAgents、HGM。

现状：四个适配器只有骨架。公共部分（拉取并固定上游版本、在计量代理后面运行子进程、
把目录冻结成产物）已经实现并测试过；每个框架“怎么取出改进器 / 执行者 / 存档、怎么跑一轮”
这几处需要对着上游代码写，目前会抛 NotWired 并说明要接什么。接入步骤见 docs/adapters.md。

下面的仓库地址是按论文公开信息填的，还没有实际拉取核对过。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..budget import BudgetExhausted
from ..core import Artifact, Ctx
from ..proxy import MeteringProxy
from .base import FrameworkAdapter, SystemState, register


class NotWired(NotImplementedError):
    """这个上游框架的适配器还没接好。"""


class ExternalRepoAdapter(FrameworkAdapter):
    repo_url: str = ""
    #: 接入时把上游 commit 固定在这里，保证各次运行用的是同一份上游代码
    commit: str = ""
    #: 上游用哪家接口
    upstream_api: str = "https://api.anthropic.com"

    # ---- 公共能力 ----
    @property
    def repo_dir(self) -> Path:
        return Path(self.cfg.get("repo_dir", Path("vendor") / self.name))

    def ensure_repo(self) -> Path:
        d = self.repo_dir
        if not d.exists():
            raise NotWired(f"{self.name}: 上游代码不在 {d}。先 `git clone {self.repo_url} {d}` 并固定 commit。")
        return d

    def workspace(self, ctx: Ctx, tag: str) -> Path:
        """给一次运行准备一份干净的上游代码副本，避免各次运行互相污染。"""
        dst = ctx.settings.run_dir / "work" / self.name / tag
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.ensure_repo(), dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        return dst

    def run(self, cmd: list[str], ctx: Ctx, cwd: Path, env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
        """在计量代理后面跑上游命令。上游的所有模型调用都会记到 ctx 的 Meter 上；预算用完抛 BudgetExhausted。"""
        with MeteringProxy(ctx, self.cfg.get("upstream_api", self.upstream_api)) as px:
            p = subprocess.run(cmd, cwd=cwd, env=px.child_env(extra=env), capture_output=True, text=True, timeout=timeout)
            if px.exhausted:
                raise BudgetExhausted(f"{self.name}: budget exhausted inside upstream process")
        if p.returncode != 0:
            raise RuntimeError(f"{self.name}: `{' '.join(cmd)}` failed ({p.returncode}):\n{p.stderr[-2000:]}")
        return p

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


@register
class GodelAgent(ExternalRepoAdapter):
    name = "godel_agent"
    level = "L2"
    repo_url = "https://github.com/Arvid-pku/Godel_Agent"
    upstream_api = "https://api.openai.com"
    todo = ("改进器 = agent 运行时可改写的自身逻辑（需要把运行时状态落盘成目录才能冻结，这是导出难度“中”的原因）；"
            "执行者 = 当前的任务求解函数；modify_improver=False 时禁止它改写自改进相关的函数。")


@register
class ADAS(ExternalRepoAdapter):
    name = "adas"
    level = "L1"
    repo_url = "https://github.com/ShengranHu/ADAS"
    upstream_api = "https://api.openai.com"
    todo = ("改进器 = 固定的元 agent 提示词（不变）；执行者 = 存档里的 forward 函数代码；"
            "存档 = 上游的 archive json。一轮 step = 元 agent 生成一个新 agent 并在搜索集上评测入档。")


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
