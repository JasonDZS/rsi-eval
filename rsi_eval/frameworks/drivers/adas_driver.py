"""ADAS 驱动：在上游某个领域目录（_drop / _mgsm / _mmlu / _gpqa）的 search.py 上做三件事。

propose  元 agent 看存档提出一个新 agent（一次提案 + 两次反思，和上游 search() 的一轮相同）
debug    新 agent 在搜索集上报错或全错时，把错误贴回同一段对话让元 agent 改（上游的 debug 循环）
solve    用一个 agent 的 forward 代码答一道题（上游 evaluate_forward_fn 里对单题做的事）

用到的上游对象：<领域>_prompt.get_prompt / get_reflexion_prompt、search.get_json_response_from_gpt_reflect、
search.LLMAgentBase / Info / AgentSystem、utils.bootstrap_confidence_interval。
唯一的替换：search.client 换成按角色记账的客户端（元 agent 算改进器，agent 里的调用算执行者）。
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys

from rsi_eval.frameworks.drivers._client import RoutedClient, main


def load(payload: dict, role: str):
    d = os.path.join(payload["upstream"], payload["domain"])
    sys.path.insert(0, d)
    os.environ.setdefault("OPENAI_API_KEY", "rsi-eval-proxy")
    search = importlib.import_module("search")
    search.client = RoutedClient(lambda kw: role)
    search.SEARCHING_MODE = True
    return search


def _fitness(search, payload: dict) -> list:
    """给还没有 fitness 字符串的存档条目补上，用上游的自助法区间（上游拿它当给元 agent 看的分数）。"""
    import numpy as np

    utils = importlib.import_module("utils")
    scale = float(payload.get("score_scale", 1.0))  # _drop 的上游分数是 0-100 的 F1，其余是 0/1
    out = []
    for sol in payload["archive"]:
        if "fitness" not in sol:
            acc = [float(x) * scale for x in sol.pop("acc_list")]
            np.random.seed(int(hashlib.sha256(repr(acc).encode()).hexdigest()[:8], 16))
            sol["fitness"] = utils.bootstrap_confidence_interval(acc)
        sol.pop("acc_list", None)
        out.append(sol["fitness"])
    return out


def cmd_propose(payload: dict) -> dict:
    search = load(payload, "improver")
    prompts = importlib.import_module(payload["domain"].lstrip("_") + "_prompt")
    fitness = _fitness(search, payload)
    archive = payload["archive"]
    model = os.environ["RSI_MODEL_IMPROVER"]
    system_prompt, prompt = prompts.get_prompt(archive)
    msg_list = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    try:
        sol = search.get_json_response_from_gpt_reflect(msg_list, model)
        r1, r2 = prompts.get_reflexion_prompt(archive[-1] if payload.get("generation", 0) > 0 else None)
        msg_list.append({"role": "assistant", "content": str(sol)})
        msg_list.append({"role": "user", "content": r1})
        sol = search.get_json_response_from_gpt_reflect(msg_list, model)
        msg_list.append({"role": "assistant", "content": str(sol)})
        msg_list.append({"role": "user", "content": r2})
        sol = search.get_json_response_from_gpt_reflect(msg_list, model)
    except Exception as e:  # 上游：打印后放弃这一轮
        return {"fitness": fitness, "error": f"During LLM generate new solution: {e!r}"}
    return {"fitness": fitness, "solution": sol, "msgs": msg_list}


def cmd_debug(payload: dict) -> dict:
    search = load(payload, "improver")
    msg_list = payload["msgs"]
    msg_list.append({"role": "assistant", "content": str(payload["solution"])})
    msg_list.append({"role": "user", "content": f"Error during evaluation:\n{payload['error']}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
    try:
        sol = search.get_json_response_from_gpt_reflect(msg_list, os.environ["RSI_MODEL_IMPROVER"])
    except Exception as e:
        return {"error": f"During LLM generate new solution: {e!r}", "msgs": msg_list}
    return {"solution": sol, "msgs": msg_list}


def _answer(search, res) -> str:
    """上游各领域对 forward 返回值的取法合在一起：Info 取 content，列表里优先找名叫 answer 的 Info。"""
    if isinstance(res, search.Info):
        return str(res.content)
    if isinstance(res, (list, tuple)):
        infos = [x for x in res if isinstance(x, search.Info)]
        named = [x for x in infos if x.name == "answer"]
        if named or infos:
            return str((named or infos)[-1].content)
    return str(res)


def cmd_solve(payload: dict) -> dict:
    search = load(payload, "executor")
    namespace = {}
    try:
        exec(payload["code"], vars(search), namespace)  # 和上游 evaluate_forward_fn 一样：在 search 模块的全局里定义 forward
        names = list(namespace.keys())
        if len(names) != 1:
            raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
        func = namespace[names[0]]
        if not callable(func):
            raise AssertionError(f"{func} is not callable")
        setattr(search.AgentSystem, "forward", func)
        res = search.AgentSystem().forward(search.Info("task", "User", payload["input"], -1))
        return {"answer": _answer(search, res)}
    except Exception as e:
        return {"error": repr(e)[:2000]}


if __name__ == "__main__":
    main({"propose": cmd_propose, "debug": cmd_debug, "solve": cmd_solve})
