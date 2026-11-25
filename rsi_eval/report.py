"""把格子结果汇总成三层结论。判定标准来自 protocol.yaml（开跑前写死）。"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .core import Settings
from .experiments import Run
from .stats import mean, required_n, slope, summarize


def load_cells(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "cells").rglob("*.json"))]


def _cfg_of(c):  # key = EXP/cfg/fw/bench/...
    return c["key"].split("/")[1]


def _ret(cs):
    """增益保留比例 = 迁移后的平均增益 / 原任务上的平均增益（原增益不为正时无定义）。"""
    tg = mean([c["target_gain"] for c in cs])
    return mean([c["gain"] for c in cs]) / tg if tg > 0 else None


def _fmt(x, pct=True):
    if x is None:
        return "—"
    return f"{100 * x:.1f}" if pct else f"{x:.3g}"


def _line(name, s):
    if not s or s["n"] == 0:
        return f"| {name} | — | — | — | — | 未跑 |"
    ci = f"[{_fmt(s['ci'][0])}, {_fmt(s['ci'][1])}]"
    return f"| {name} | {s['n']} | {_fmt(s['mean'])} | {ci} | {_fmt(s['p'], False)} | {'通过' if s['pass'] else '未通过'} |"


def build(settings: Settings) -> dict:
    run = Run(settings, log=lambda m: None)
    cells = load_cells(settings.run_dir)
    method = settings.protocol.get("ci_method", "t")
    S = lambda xs: summarize(xs, method)  # noqa: E731
    by = defaultdict(list)
    for c in cells:
        by[(c["key"].split("/")[0], _cfg_of(c), c.get("framework"), c.get("benchmark"))].append(c)

    groups = sorted({(cfg, fw, b) for (exp, cfg, fw, b) in by if exp == "E2"})
    out = {"suite": settings.suite["name"], "protocol_hash": settings.protocol_hash(), "groups": [], "costs": {}}

    for cfg, fw, bench in groups:
        e2 = {c["seed"]: c for c in by[("E2", cfg, fw, bench)]}
        seeds = sorted(e2)
        g = {"model_config": cfg, "framework": fw, "benchmark": bench, "n_runs": len(seeds)}
        e1 = [c["score"] for c in by[("E1", cfg, fw, bench)]]
        g["baseline"] = {"mean": mean(e1) if e1 else None, "n": len(e1)}
        g["evolved"] = {"score": mean([e2[s]["score"] for s in seeds]), "gain": S([e2[s]["gain"] for s in seeds]),
                        "evolve_tokens": mean([e2[s]["evolve_tokens"] for s in seeds]), "usd": mean([e2[s]["total_usd"] for s in seeds])}

        # ---- 第一层 ----
        e3 = {c["seed"]: c for c in by[("E3", cfg, fw, bench)]}
        l1 = S([e2[s]["score"] - e3[s]["score"] for s in seeds if s in e3])
        g["L1"] = {"vs_repeated_sampling": l1, "pass": l1["pass"], "control_score": mean([e3[s]["score"] for s in e3]) if e3 else None,
                   "control_mode": next(iter(e3.values()))["control_mode"] if e3 else None}

        # ---- 第二层 ----
        l2 = {"unseen": {}, "model": None}
        for c in by[("E4", cfg, fw, bench)]:
            l2["unseen"].setdefault(c["unseen"], []).append(c)
        l2["unseen"] = {ub: {"gain": S([c["gain"] for c in cs]), "retention": _ret(cs)}
                        for ub, cs in l2["unseen"].items()}
        e5 = by[("E5", cfg, fw, bench)]
        if e5:
            l2["model"] = {"transfer_model": e5[0]["transfer_model"], "gain": S([c["gain"] for c in e5]),
                           "retention": _ret(e5)}
        l2["complete"] = bool(l2["unseen"]) and l2["model"] is not None
        l2["pass"] = l2["complete"] and all(v["gain"]["pass"] for v in l2["unseen"].values()) and l2["model"]["gain"]["pass"]
        g["L2"] = l2

        # ---- 第三层 ----
        imp = defaultdict(dict)  # (seed, unseen) -> {(gen, budget): cell}
        for c in by[("IMP", cfg, fw, bench)]:
            imp[(c["seed"], c["unseen"])][(c["gen"], int(c["budget"]))] = c
        ctl = {(c["seed"], c["unseen"]): c for c in by[("E6ctl", cfg, fw, bench)]}
        e7 = {c["seed"]: c for c in by[("E7", cfg, fw, bench)]}
        l3 = {}
        for ub in sorted({ub for (_, ub) in imp}):
            d_first, d_ctl, d_cost, by_label, slopes, adj = [], [], [], defaultdict(list), [], []
            for s in seeds:
                runs = imp.get((s, ub), {})
                B = int(run.improve_budget(e2[s]))
                idx = run.gen_indices(e2[s])
                gains = {lab: runs[(i, B)]["gain"] for lab, i in idx.items() if (i, B) in runs}
                for lab, v in gains.items():
                    by_label[lab].append(v)
                if "last" in gains and "first" in gains:
                    d_first.append(gains["last"] - gains["first"])
                if "last" in gains and (s, ub) in ctl:
                    d_ctl.append(gains["last"] - ctl[(s, ub)]["gain"])
                extra = int(B + e2[s]["generations"][idx.get("last", len(e2[s]["generations"]) - 1)]["tokens_cum"])
                if "last" in gains and (0, extra) in runs and extra != B:
                    d_cost.append(gains["last"] - runs[(0, extra)]["gain"])
                # 相邻代（E9）
                allg = run.gen_indices(e2[s], "all")
                series = [runs[(i, B)]["gain"] for i in allg.values() if (i, B) in runs]
                if len(series) == len(allg) and len(series) >= 3:
                    slopes.append(slope(series))
                    adj.append(mean([float(b > a) for a, b in zip(series, series[1:])]))
            d_exp = [e2[s]["score"] - e7[s]["score"] for s in seeds if s in e7]
            checks = {"final_vs_gen0": S(d_first), "vs_repeated_sampling": S(d_ctl), "vs_experience_only": S(d_exp), "vs_cost_accounted": S(d_cost)}
            complete = all(c["n"] > 0 for c in checks.values())
            trend = S(slopes)
            l3[ub] = {"gain_by_generation": {lab: mean(v) for lab, v in by_label.items()}, "checks": checks, "complete": complete,
                      "pass": complete and all(c["pass"] for c in checks.values()),
                      "adjacent": {"slope": trend, "frac_adjacent_up": mean(adj) if adj else None}}
            l3[ub]["recursion_pass"] = l3[ub]["pass"] and trend["pass"]
            if d_first and len(d_first) >= 2:
                l3[ub]["suggested_repeats"] = required_n(summarize(d_first)["sd"], abs(mean(d_first)))
        g["L3"] = l3

        # ---- E10 ----
        e10 = defaultdict(list)
        for c in by[("E10", cfg, fw, bench)]:
            e10[c["gen"]].append(c)
        if e10:
            g["E10"] = {f"gen{k}": {"child_gain": mean([c["gain"] for c in v]), "changed": mean([float(c["changed"]) for c in v]), "n": len(v)}
                        for k, v in sorted(e10.items())}
        out["groups"].append(g)

    # ---- 成本 ----
    cost = defaultdict(lambda: defaultdict(float))
    for c in cells:
        exp = c["key"].split("/")[0]
        for ph in ("evolve", "select", "report"):
            cost[exp][ph] += c["_cost"][ph]["tokens"]
        cost[exp]["usd"] += c["_cost"]["total"]["usd"]
        cost[exp]["cells"] += 1
    out["costs"] = {k: dict(v) for k, v in sorted(cost.items())}
    return out


def to_markdown(r: dict) -> str:
    L = [f"# RSI 三层评测报告：{r['suite']}", "", f"协议哈希 `{r['protocol_hash']}`。分数和差值单位是百分点，置信区间为 95%，p 值为配对符号翻转检验（单侧）。", ""]
    hdr = "| 比较 | n | 均值差 | 95% CI | p | 判定 |\n|---|---|---|---|---|---|"
    for g in r["groups"]:
        L += [f"## {g['framework']} / {g['benchmark']}（模型配置 {g['model_config']}，{g['n_runs']} 次独立演化）", ""]
        ev = g["evolved"]
        L += [f"初始系统报告集分数 {_fmt(g['baseline']['mean'])}，自改进后 {_fmt(ev['score'])}，"
              f"每次演化平均 {ev['evolve_tokens']:.0f} token（${ev['usd']:.2f}）。", ""]
        L += ["### 第一层：目标任务性能", "", hdr, _line(f"自改进 − 同预算重复采样（{g['L1']['control_mode'] or '—'}）", g["L1"]["vs_repeated_sampling"]),
              _line("自改进 − 初始系统（参考）", ev["gain"]), ""]
        L += ["### 第二层：未见任务泛化", "", hdr]
        for ub, v in g["L2"]["unseen"].items():
            L.append(_line(f"未见基准 {ub} 上的增益（保留 {_fmt(v['retention'])}%）", v["gain"]))
        if g["L2"]["model"]:
            m = g["L2"]["model"]
            L.append(_line(f"换执行模型 {m['transfer_model']} 后的增益（保留 {_fmt(m['retention'])}%）", m["gain"]))
        L += ["", f"第二层判定：{'通过' if g['L2']['pass'] else ('未通过' if g['L2']['complete'] else '未跑全')}", ""]
        for ub, v in g["L3"].items():
            L += [f"### 第三层：改进器自身进化（改进运行在 {ub} 上）", "",
                  "各代改进器从同一起点、同预算改一轮带来的增益：" + "，".join(f"{k} {_fmt(x)}" for k, x in v["gain_by_generation"].items()), "", hdr,
                  _line("最终代 − 第 0 代", v["checks"]["final_vs_gen0"]),
                  _line("最终代 − 第 0 代同预算多次重启取最好", v["checks"]["vs_repeated_sampling"]),
                  _line("自改进 − 只累积经验（同预算，报告集分数）", v["checks"]["vs_experience_only"]),
                  _line("最终代 − 第 0 代加上演化成本", v["checks"]["vs_cost_accounted"]),
                  _line("相邻代增益斜率（每代）", v["adjacent"]["slope"]), "",
                  f"第三层判定：{'通过' if v['pass'] else ('未通过' if v['complete'] else '未跑全')}；"
                  f"递归成立：{'是' if v['recursion_pass'] else '否'}"
                  + (f"；按实测方差估计需要约 {v['suggested_repeats']} 次重复才能稳定检出最终代与第 0 代的差距" if v.get("suggested_repeats") else ""), ""]
        if g.get("E10"):
            L += ["### 改进器改改进器（E10）", "", "| 父代 | 子代改进器带来的增益 | 子代确实被改动的比例 | n |", "|---|---|---|---|"]
            L += [f"| {k} | {_fmt(v['child_gain'])} | {_fmt(v['changed'])}% | {v['n']} |" for k, v in g["E10"].items()] + [""]
    L += ["## 成本", "", "| 实验 | 格子数 | 演化 token | 选择 token | 评测 token | 美元 |", "|---|---|---|---|---|---|"]
    for k, v in r["costs"].items():
        L.append(f"| {k} | {int(v.get('cells', 0))} | {int(v.get('evolve', 0))} | {int(v.get('select', 0))} | {int(v.get('report', 0))} | {v.get('usd', 0):.2f} |")
    L += ["", "IMP 是 E6、E8、E9 共用的改进运行。token 数包含评测缓存重放的部分（口径上各条件一致），美元同理；真实花费见 ledger.jsonl 中 cached=false 的行。", ""]
    return "\n".join(L)


def write(settings: Settings) -> Path:
    r = build(settings)
    (settings.run_dir / "summary.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    p = settings.run_dir / "report.md"
    p.write_text(to_markdown(r), encoding="utf-8")
    return p
