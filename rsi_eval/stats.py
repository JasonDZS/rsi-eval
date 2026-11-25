"""统计：均值置信区间、配对差值、符号翻转置换检验、趋势斜率。纯标准库。"""
from __future__ import annotations

import itertools
import math
import random

# 双侧 95% 的 t 临界值，df=1..30
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def t_ci(xs, level: float = 0.95):
    """t 分布置信区间。小样本（3–5 次运行）下比自助法更保守可靠。目前只支持 95%。"""
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    t = _T95[n - 2] if n - 1 <= 30 else 1.96
    h = t * sd(xs) / math.sqrt(n)
    return (mean(xs) - h, mean(xs) + h)


def bootstrap_ci(xs, level: float = 0.95, n_boot: int = 10000, seed: int = 0):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(mean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    a = (1 - level) / 2
    return (ms[int(a * n_boot)], ms[min(n_boot - 1, int((1 - a) * n_boot))])


def signflip_p(diffs) -> float:
    """配对符号翻转置换检验，单侧（H1: 均值 > 0）。n ≤ 16 时精确枚举。"""
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = sum(diffs)
    if n <= 16:
        signs = itertools.product((1, -1), repeat=n)
        total, ge = 2**n, 0
        for s in signs:
            if sum(d * x for d, x in zip(diffs, s)) >= obs - 1e-12:
                ge += 1
        return ge / total
    rng = random.Random(0)
    ge = sum(sum(d * rng.choice((1, -1)) for d in diffs) >= obs - 1e-12 for _ in range(20000))
    return ge / 20000


def summarize(xs, method: str = "t") -> dict:
    """一组数（通常是配对差值）的摘要。pass = 置信区间下界 > 0。"""
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0, "mean": None, "ci": [None, None], "p": None, "pass": False}
    lo, hi = t_ci(xs) if method == "t" else bootstrap_ci(xs)
    ok = len(xs) >= 2 and not math.isnan(lo) and lo > 0
    return {"n": len(xs), "mean": mean(xs), "sd": None if len(xs) < 2 else sd(xs),
            "ci": [None if math.isnan(lo) else lo, None if math.isnan(hi) else hi],
            "p": signflip_p(xs), "pass": bool(ok)}


def slope(ys) -> float:
    """对 x = 0..n-1 的最小二乘斜率。"""
    n = len(ys)
    if n < 2:
        return float("nan")
    xm, ym = (n - 1) / 2, mean(ys)
    den = sum((i - xm) ** 2 for i in range(n))
    return sum((i - xm) * (y - ym) for i, y in enumerate(ys)) / den


def required_n(sd_diff: float, effect: float, power_z: float = 0.84, alpha_z: float = 1.96) -> int:
    """按试点实测的差值标准差估计需要多少次重复（正态近似，默认 80% 功效）。"""
    if effect <= 0 or math.isnan(sd_diff):
        return 0
    return max(2, math.ceil(((alpha_z + power_z) * sd_diff / effect) ** 2))
