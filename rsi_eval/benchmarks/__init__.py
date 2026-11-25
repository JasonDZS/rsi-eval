"""Benchmark 注册表。

新增一个基准：继承 Benchmark，实现 load() 和 score()，用 @register 注册。
数据统一放在 data/<name>.jsonl（每行 {id, input, target, meta}），由
`rsi-eval data prepare <name>` 从公开来源导出，保证每次运行用的是同一份固定数据。
"""
from .base import BENCHMARKS, Benchmark, Task, get_benchmark, register  # noqa: F401
from . import qa, code, mock  # noqa: F401,E402  触发注册
