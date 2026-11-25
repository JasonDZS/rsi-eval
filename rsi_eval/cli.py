"""命令行入口。

rsi-eval plan    --suite configs/suites/pilot.yaml      # 列出要跑的格子数，不调用模型
rsi-eval run     --suite configs/suites/pilot.yaml      # 跑（可中断续跑）
rsi-eval report  --suite configs/suites/pilot.yaml      # 汇总三层结论
rsi-eval vendor fetch                                   # 拉取上游框架（ADAS、Gödel Agent）到 vendor/，检出固定的 commit
rsi-eval data prepare gsm8k mmlu ...                    # 把基准数据导出成固定的 jsonl，并写下带 sha256 的清单
rsi-eval protocol freeze                                # 写死判定标准
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .core import Settings, load_yaml


def _settings(a) -> Settings:
    return Settings.load(a.suite, a.out)


def cmd_plan(a):
    from .experiments import plan

    s = _settings(a)
    tier = s.suite.get("tier", "minimal")
    p = {**s.protocol["tiers"][tier], **s.suite.get("params", {})}
    rep, gens = int(p["repeats"]), len(p.get("generations", ["first", "mid", "last"]))
    rows, total_imp = [], 0
    for exp, cfg, fw, bench, unseen in plan(s):
        n_imp = {"E6": rep * (gens + 1), "E8": rep, "E9": rep * (int(p["rounds"]) + 1), "E10": rep * gens}.get(exp, 0)
        n_evo = {"E2": rep, "E7": rep, "E11": rep}.get(exp, 0)
        total_imp += n_imp
        rows.append((exp, fw, bench, cfg, n_evo, n_imp))
    print(f"套件 {s.suite['name']}（{tier} 档）：每个条件 {rep} 次独立运行，演化 {p['rounds']} 轮，对比 {gens} 代改进器")
    print(f"{'实验':<5}{'框架':<14}{'演化基准':<14}{'配置':<5}{'演化运行':>8}{'改进运行(上限)':>14}")
    for r in rows:
        print(f"{r[0]:<6}{r[1]:<16}{r[2]:<18}{r[3]:<6}{r[4]:>8}{r[5]:>14}")
    print(f"改进运行合计上限 {total_imp} 次（E6/E8/E9 之间重复的格子只跑一次）。")


def cmd_run(a):
    from .experiments import run_suite
    from .report import write

    s = _settings(a)
    only = [x.strip().upper() for x in a.only.split(",")] if a.only else None
    paid = sorted({m for cfg in s.model_configs.values() for m in cfg.values() if s.models[m]["provider"] != "mock"}
                  & {m for m in s.model_configs[s.suite.get("model_config", "A")].values()})
    if paid and not a.yes:
        sys.exit(f"这个套件会调用付费模型 {paid}。确认预算后加 --yes 再跑；建议先在套件里设 hard_cap_usd。")
    run_suite(s, only)
    print(f"报告：{write(s)}")


def cmd_report(a):
    from .report import write

    p = write(_settings(a))
    print(p.read_text(encoding="utf-8"))


def cmd_data(a):
    from .benchmarks import get_benchmark

    root = Path(a.root)
    protocol = load_yaml(root / "configs" / "protocol.yaml")
    for name in a.names:
        b = get_benchmark(name, protocol, (root / "data").resolve())
        rows = b.prepare()
        b.data_path.parent.mkdir(parents=True, exist_ok=True)
        blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
        b.data_path.write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        if b.manifest_path.exists() and json.loads(b.manifest_path.read_text(encoding="utf-8"))["jsonl_sha256"] != digest and not a.update_manifest:
            sys.exit(f"{name}: 导出的数据和仓库里的清单 {b.manifest_path} 不一致。确认是有意更换数据版本后加 --update-manifest。")
        manifest = {"benchmark": name, "rows": len(rows), "jsonl_sha256": digest, "source": b.provenance()}
        b.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        b._splits = None
        print(f"{name}: {len(rows)} 题 -> {b.data_path}；切分 " + ", ".join(f"{k}={len(v)}" for k, v in b.splits().items()))


def cmd_vendor(a):
    from .frameworks import ADAPTERS
    from .frameworks.external import ExternalRepoAdapter, fetch_repo, repo_head

    root = Path(a.root)
    pinned = {n: c for n, c in ADAPTERS.items() if issubclass(c, ExternalRepoAdapter) and c.commit}
    for name in a.names or sorted(pinned):
        if name not in pinned:
            sys.exit(f"{name}: 没有固定 commit 的上游框架（有：{sorted(pinned)}）")
        cls, dst = pinned[name], root / "vendor" / name
        if a.action == "fetch":
            print(f"{name}: {fetch_repo(cls.repo_url, cls.commit, dst)} -> {dst} @ {cls.commit}")
        else:
            head = repo_head(dst) if dst.exists() else "(没有拉取)"
            print(f"{name}: 固定 {cls.commit}，本地 {head} {'✓' if head == cls.commit else '✗'}")


def cmd_protocol(a):
    root = Path(a.root)
    h = hashlib.sha256(json.dumps(load_yaml(root / "configs" / "protocol.yaml"), sort_keys=True).encode()).hexdigest()[:16]
    (root / "configs" / "protocol.lock").write_text(h + "\n")
    print(f"protocol.lock = {h}。把它提交进仓库；之后改 protocol.yaml 会被 require_frozen_protocol 的套件拒绝。")


def cmd_proxy(a):
    from .proxy import serve

    serve(a.host, a.port, a.upstream, a.ledger, a.limit_tokens)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rsi-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("run", cmd_run), ("report", cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("--suite", required=True)
        p.add_argument("--out", default="runs")
        if name == "run":
            p.add_argument("--only", help="只跑这些实验，例如 E1,E2")
            p.add_argument("--yes", action="store_true", help="确认可以调用付费模型")
        p.set_defaults(fn=fn)
    p = sub.add_parser("data")
    p.add_argument("action", choices=["prepare"])
    p.add_argument("names", nargs="+")
    p.add_argument("--root", default=".")
    p.add_argument("--update-manifest", action="store_true", help="有意更换数据版本时，覆盖仓库里的清单")
    p.set_defaults(fn=cmd_data)
    p = sub.add_parser("vendor", help="拉取上游框架代码到 vendor/ 并检出固定的 commit")
    p.add_argument("action", choices=["fetch", "status"])
    p.add_argument("names", nargs="*")
    p.add_argument("--root", default=".")
    p.set_defaults(fn=cmd_vendor)
    p = sub.add_parser("protocol")
    p.add_argument("action", choices=["freeze"])
    p.add_argument("--root", default=".")
    p.set_defaults(fn=cmd_protocol)
    p = sub.add_parser("proxy", help="给上游框架子进程用的计量代理")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--upstream", required=True, help="真实接口地址，例如 https://api.anthropic.com")
    p.add_argument("--ledger", default="runs/proxy-ledger.jsonl")
    p.add_argument("--limit-tokens", type=int)
    p.set_defaults(fn=cmd_proxy)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
