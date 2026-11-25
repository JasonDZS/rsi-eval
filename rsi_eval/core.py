"""运行上下文、产物存档、评测缓存、配置加载。"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .budget import Ledger, Meter, Usage
from .llm import Completion, LLMClient, make_client


class ProtocolViolation(Exception):
    """违反评测协议（例如在演化阶段读取报告集）。"""


# --------------------------------------------------------------------------
# 产物：改进器、执行者、存档，全部按内容寻址、冻结后不可变
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Artifact:
    kind: str  # improver | executor | archive
    digest: str
    path: str  # 目录的绝对路径

    def to_dict(self) -> dict:
        return {"kind": self.kind, "digest": self.digest, "path": self.path}

    @staticmethod
    def from_dict(d: dict) -> "Artifact":
        return Artifact(d["kind"], d["digest"], d["path"])

    def load_json(self) -> Any:
        return json.loads((Path(self.path) / "spec.json").read_text(encoding="utf-8"))


def _hash_dir(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


class ArtifactStore:
    """runs/<name>/artifacts/<kind>/<digest>/。同样内容只存一份。"""

    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()

    def put_json(self, kind: str, obj: Any) -> Artifact:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:16]
        d = self.root / kind / digest
        with self._lock:
            if not d.exists():
                d.mkdir(parents=True)
                (d / "spec.json").write_text(blob, encoding="utf-8")
        return Artifact(kind, digest, str(d))

    def put_dir(self, kind: str, src: Path) -> Artifact:
        """冻结一个目录（上游框架的代码快照、checkpoint 等）。"""
        digest = _hash_dir(src)
        d = self.root / kind / digest
        with self._lock:
            if not d.exists():
                shutil.copytree(src, d)
        return Artifact(kind, digest, str(d))


# --------------------------------------------------------------------------
# 评测缓存：同一执行者 × 模型 × 题 × 采样号 只真实调用一次。
# 命中时仍按原始消耗记账（标记 cached），保证各条件的预算口径一致。
# --------------------------------------------------------------------------
class EvalCache:
    def __init__(self, root: Path | None):
        self.root = root
        self._lock = threading.Lock()

    def _path(self, key: tuple) -> Path:
        h = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        return self.root / h[:2] / f"{h}.json"

    def get(self, key: tuple) -> dict | None:
        if not self.root:
            return None
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def put(self, key: tuple, value: dict) -> None:
        if not self.root:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


@dataclass
class Settings:
    models: dict  # models.yaml 的 models
    model_configs: dict  # A / B：role -> 模型名
    protocol: dict  # protocol.yaml
    suite: dict  # suites/*.yaml
    root: Path  # 仓库根目录（找 data/、vendor/）
    run_dir: Path

    @staticmethod
    def load(suite_path: str | Path, out_root: str | Path = "runs", config_dir: str | Path | None = None) -> "Settings":
        suite_path = Path(suite_path)
        cfg_dir = Path(config_dir) if config_dir else suite_path.resolve().parent.parent
        m = load_yaml(cfg_dir / "models.yaml")
        suite = load_yaml(suite_path)
        protocol = load_yaml(cfg_dir / "protocol.yaml")
        # 彩排用：套件可以缩小题量。改过之后协议哈希就变了，所以 require_frozen_protocol 的套件用不了这个口子。
        for name, over in suite.get("benchmark_overrides", {}).items():
            protocol["benchmarks"][name] = {**protocol["benchmarks"].get(name, {}), **over}
        return Settings(
            models=m["models"],
            model_configs=m["model_configs"],
            protocol=protocol,
            suite=suite,
            root=cfg_dir.parent,
            run_dir=Path(out_root) / suite["name"],
        )

    def protocol_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.protocol, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# 上下文：适配器和运行器只通过它拿模型、记账、存产物
# --------------------------------------------------------------------------
class MeteredClient:
    """包一层记账。适配器拿到的永远是这个，不会漏记。"""

    def __init__(self, inner: LLMClient, role: str, ctx: "Ctx"):
        self.inner, self.role, self.ctx = inner, role, ctx
        self.model = inner.model

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 4096, sample: int | None = None) -> Completion:
        ctx = self.ctx
        ctx.meter.check(ctx.phase)
        c = self.inner.complete(prompt, system=system, max_tokens=max_tokens, sample=ctx.nonce() if sample is None else sample)
        ctx.meter.charge(ctx.phase, self.role, self.inner.name, c.usage, self.inner.price.usd(c.usage))
        return c


@dataclass
class Ctx:
    settings: Settings
    roles: dict  # role -> 模型名（executor / improver / transfer）
    store: ArtifactStore
    cache: EvalCache
    meter: Meter
    phase: str = "evolve"
    seed: int = 0
    _clients: dict = field(default_factory=dict, repr=False)
    _counter: list = field(default_factory=lambda: [0], repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def with_(self, **kw) -> "Ctx":
        if "seed" in kw:  # 新的 seed 对应新的独立运行：采样计数从头开始，保证可复现
            kw.setdefault("_counter", [0])
        return replace(self, **kw)

    def client(self, role: str) -> LLMClient:
        name = self.roles[role]
        with self._lock:
            if name not in self._clients:
                self._clients[name] = make_client(name, self.settings.models)
            return self._clients[name]

    def llm(self, role: str) -> MeteredClient:
        return MeteredClient(self.client(role), role, self)

    def nonce(self) -> int:
        """每次调用不同、但对同一 seed 可复现的采样号。"""
        with self._lock:
            self._counter[0] += 1
            return self.seed * 1_000_003 + self._counter[0]

    def price_usd(self, role: str, usage: Usage) -> float:
        return self.client(role).price.usd(usage)

    @property
    def protocol(self) -> dict:
        return self.settings.protocol


def make_root_ctx(settings: Settings, model_config: str) -> Ctx:
    run_dir = settings.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir / "ledger.jsonl")
    meter = Meter(ledger, tags={"suite": settings.suite["name"], "model_config": model_config},
                  unit=settings.protocol.get("budget_unit", "tokens"),
                  hard_cap_usd=settings.suite.get("hard_cap_usd"))
    return Ctx(
        settings=settings,
        roles=dict(settings.model_configs[model_config]),
        store=ArtifactStore(run_dir / "artifacts"),
        cache=EvalCache(run_dir / "eval_cache"),
        meter=meter,
    )
