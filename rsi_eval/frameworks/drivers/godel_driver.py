"""Gödel Agent 驱动：把上游那个“在一个 Python 进程里不停改自己”的 agent 变成可以冻结、恢复、分轮运行的东西。

上游（src/agent_module.py）的运行方式：导入模块时创建 self_evolving_agent，然后 evolve() 和 execute_action()
互相递归。每次 evolve() 让模型选动作（读代码、改代码 action_adjust_logic、跑代码 action_run_code、评测…），
改动直接打在进程内存里（monkey patching），进程一退出状态就没了。

这里的做法
- 状态 = 动作日志。每一次 action_adjust_logic / action_run_code 调用（参数、期间发生的模型调用的应答）都记下来。
  恢复状态 = 起一个新进程、导入上游模块、按顺序重放日志；重放时模型调用用记下的应答，不花钱、结果一致。
- 改进器 = agent 自身的逻辑：Agent 类的方法、action_* 函数、goal_prompt、动作列表。一次动作前后这些东西变了，
  这条日志就标记为“改了改进器”。modify_improver=False 时这类改动会被撤销并向 agent 报错。
- 执行者 = 完整的动作日志（solver 会调用 agent 的方法，所以执行者必须带着它被写出来时的整个运行时状态）。
- 一轮 = 至多 max_calls 次 evolve()（改进侧模型调用）。到数后在下一次调用前停下，此时所有动作结果都已写回历史。

对上游内存对象做的替换（上游文件不改）
- agent.client：换成经过计量代理的客户端。带 tools 的调用（evolve 自己的那次）算改进器，其余全部算执行者，
  型号由角色决定。这样 agent 没法靠把 solver 里的 model 改成更强的型号来“进步”。
- solver：换成上游 task_<基准>.py 里对应基准的初始 solver（上游是手工改 agent_module.py 来换基准的）。
- agent.goal_task / action_evaluate_on_task：评测题目换成协议传进来的搜索集，评分用本仓库的规则评分；
  反馈文本的格式照上游。上游“验证集过线就去测测试集”的逻辑去掉，演化期间不碰留出集。
- 发请求前丢掉找不到对应 tool_calls 的 tool 消息：上游只保留最近 10 条历史，截断处可能留下这种孤儿消息。
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import json
import os
import random
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rsi_eval.frameworks.drivers._client import BudgetStop, RoutedClient, main

#: 协议里的基准 -> 上游的任务模块（取其中的初始 solver 和反馈格式）
FAMILIES = {"drop": "task_drop", "mgsm": "task_mgsm", "mmlu": "task_mmlu", "gpqa": "task_gpqa", "gsm8k": "task_mgsm", "gsm_hard": "task_mgsm"}
ACTIONS = ("action_adjust_logic", "action_run_code")
FROZEN_MSG = ("Self-modification of the agent's own logic (the `Agent` class, `action_*` functions, goal prompt and action list) "
              "is disabled in this run and the change was reverted. You can still modify `solver` and create helper functions or objects.")


class StepDone(BaseException):
    """这一轮的改进侧调用次数用完了。"""


# --------------------------------------------------------------------------
# 模型客户端：分角色、限次数、录制与重放
# --------------------------------------------------------------------------
def plain_messages(messages) -> list[dict]:
    """历史里混着 dict 和 SDK 的消息对象，统一成只含有效字段的 dict，并丢掉孤儿 tool 消息。"""
    out, known = [], set()
    for m in messages:
        d = m if isinstance(m, dict) else m.model_dump(mode="json")
        d = {k: d[k] for k in ("role", "content", "name", "tool_calls", "tool_call_id") if d.get(k) is not None}
        if d.get("role") == "assistant" and d.get("tool_calls"):
            known |= {tc.get("id") for tc in d["tool_calls"]}
            if d.get("content") in ("None", None):  # 上游把 content 一律 str()，空内容变成了 "None"
                d["content"] = ""
        if d.get("role") == "tool" and d.get("tool_call_id") not in known:
            continue
        out.append(d)
    return out


class GodelClient(RoutedClient):
    def __init__(self, mode: str, max_calls: int):
        super().__init__(self._role)
        self.mode, self.max_calls, self.improver_calls = mode, max_calls, 0
        self.recording: list | None = None  # 正在执行一条会被记录的动作：模型应答记到这里
        self.replaying: dict | None = None  # 正在重放一条动作：请求指纹 -> 待用的应答
        self._lock = threading.Lock()

    def _role(self, kw) -> str:
        return "improver" if (self.mode == "step" and kw.get("tools")) else "executor"

    def create(self, **kw):
        from openai.types.chat import ChatCompletion

        role = self._role(kw)
        kw["model"] = self.models[role]
        kw["messages"] = plain_messages(kw.get("messages", []))
        key = hashlib.sha256(json.dumps({k: kw.get(k) for k in ("messages", "tools", "tool_choice", "response_format", "n")},
                                        sort_keys=True, default=str).encode()).hexdigest()[:24]
        with self._lock:
            if self.replaying is not None:
                if self.replaying.get(key):
                    return ChatCompletion.model_validate(self.replaying[key].pop(0))
            elif role == "improver" and self.recording is None:
                if self.improver_calls >= self.max_calls:
                    raise StepDone()
                self.improver_calls += 1
        resp = self._send(role, kw)
        with self._lock:
            if self.recording is not None:
                self.recording.append({"key": key, "response": resp.model_dump(mode="json")})
        return resp


# --------------------------------------------------------------------------
# 动作日志：记录、区分是否改了改进器、重放
# --------------------------------------------------------------------------
def _jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, ValueError):
        return False


class Recorder:
    def __init__(self, am, client: GodelClient):
        self.am, self.client = am, client
        self.log: list[dict] = []
        self.depth = 0
        self.frozen = False

    # ---- 改进器由哪些对象组成 ----
    def snapshot(self):
        am, agent = self.am, self.am.self_evolving_agent
        mod = {k: v for k, v in vars(am).items() if (k.startswith("action_") and callable(v)) or k in ("Agent", "AgentBase", "self_evolving_agent")}
        cls = {k: (dict(v) if k == "__source__" else v) for k, v in vars(am.Agent).items() if not k.startswith("__") or k == "__source__"}
        inst = {"goal_prompt": agent.goal_prompt, "action_functions": copy.deepcopy(agent.action_functions)}
        return mod, cls, inst

    @staticmethod
    def _same(a: dict, b: dict) -> bool:
        return a.keys() == b.keys() and all(a[k] == b[k] if isinstance(a[k], (dict, str)) else a[k] is b[k] for k in a)

    def changed(self, before) -> bool:
        now = self.snapshot()
        return not (self._same(now[0], before[0]) and self._same(now[1], before[1]) and now[2] == before[2])

    def restore(self, before) -> None:
        am = self.am
        mod, cls, inst = before
        for k in [k for k in self.snapshot()[0] if k not in mod]:
            delattr(am, k)
        for k, v in mod.items():
            setattr(am, k, v)
        for k in [k for k in self.snapshot()[1] if k not in cls]:
            delattr(am.Agent, k)
        for k, v in cls.items():
            setattr(am.Agent, k, dict(v) if k == "__source__" else v)
        agent = am.self_evolving_agent
        agent.goal_prompt, agent.action_functions = inst["goal_prompt"], copy.deepcopy(inst["action_functions"])

    # ---- 包住两个会改状态的动作 ----
    def install(self) -> None:
        """agent 可能把这两个动作函数本身也改掉；每次动作之后重新包一层，记录不会断。"""
        for name in ACTIONS:
            cur = getattr(self.am, name, None)
            if callable(cur) and not getattr(cur, "_rsi_recorder", False):
                setattr(self.am, name, self._wrap(name, cur))

    def _wrap(self, name, fn):
        @functools.wraps(fn)  # inspect.getsource 会顺着 __wrapped__ 找到原函数，agent 读到的还是它自己的代码
        def wrapper(*args, **kwargs):
            return self._call(name, fn, args, kwargs)

        wrapper._rsi_recorder = True
        return wrapper

    def _protected_target(self, name, args, kwargs) -> bool:
        if name != "action_adjust_logic":
            return False
        module = kwargs.get("module_name", args[0] if args else "")
        target = str(kwargs.get("target_name", args[1] if len(args) > 1 else ""))
        root = target.split(".")[0]
        return module == "agent_module" and (root in ("Agent", "AgentBase", "self_evolving_agent") or root.startswith("action_"))

    def _call(self, name, fn, args, kwargs):
        if self.depth > 0:  # 动作里又调了动作（比如 run_code 里调 adjust_logic）：算在外层那条里
            return fn(*args, **kwargs)
        if self.frozen and self._protected_target(name, args, kwargs):
            raise ValueError(FROZEN_MSG)
        entry = {"id": uuid.uuid4().hex[:12], "kind": name, "args": list(args), "kwargs": dict(kwargs), "improver": False, "llm": []}
        if not _jsonable([entry["args"], entry["kwargs"]]):
            entry.update(args=[], kwargs={}, replayable=False, note=repr((args, kwargs))[:500])
        before = self.snapshot()
        self.depth += 1
        self.client.recording = entry["llm"]
        exc = None
        try:
            result = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001  失败的动作也可能留下一半的改动，同样记下来重放
            exc = e
        finally:
            self.depth -= 1
            self.client.recording = None
        changed = self.changed(before)
        if changed and self.frozen:
            self.restore(before)
        else:
            entry["improver"] = changed
        entry["ok"] = exc is None
        self.log.append(entry)
        self.install()
        if changed and self.frozen:
            raise ValueError(FROZEN_MSG)
        if exc is not None:
            raise exc
        return result

    def replay(self, log: list[dict]) -> None:
        for e in log:
            self.log.append(e)
            if not e.get("replayable", True):
                continue
            fn = getattr(self.am, e["kind"], None)
            fn = getattr(fn, "__wrapped__", fn) if getattr(fn, "_rsi_recorder", False) else fn
            before = None if e["improver"] else self.snapshot()
            cassette: dict = {}
            for r in e.get("llm", []):
                cassette.setdefault(r["key"], []).append(r["response"])
            self.depth += 1
            self.client.replaying = cassette
            try:
                fn(*e["args"], **e["kwargs"])
            except Exception as ex:  # noqa: BLE001
                print(f"[rsi-eval] replay of {e['kind']} {e['id']} raised {ex!r}", file=sys.stderr)
            finally:
                self.depth -= 1
                self.client.replaying = None
            if before is not None and self.changed(before):  # 不属于改进器的动作，重放后改进器必须原样
                self.restore(before)
            self.install()


# --------------------------------------------------------------------------
# 评测：协议传进来的搜索集，反馈格式照上游
# --------------------------------------------------------------------------
class RsiTask:
    def __init__(self, bench_name: str, tasks: list[dict], batch: int, seed: str, workers: int):
        from rsi_eval.benchmarks import BENCHMARKS, Task

        self.bench = BENCHMARKS[bench_name]({}, Path("."))
        self.tasks = [Task(str(t["id"]), t["input"], t["target"], t.get("meta", {})) for t in tasks]
        self.batch, self.seed, self.workers = batch, seed, workers
        self.include_info = FAMILIES[bench_name] != "task_mgsm"  # 上游 MGSM 的反馈只有准确率，其余基准附带逐题信息
        self.evals: list[float] = []

    def evaluate(self, solver):
        sample = random.Random(f"{self.seed}|{len(self.evals)}").sample(self.tasks, min(self.batch, len(self.tasks)))
        with ThreadPoolExecutor(max_workers=max(1, min(len(sample), self.workers))) as pool:
            results = list(pool.map(solver, [t.input for t in sample]))
        acc, info = [], []
        for i, (t, res) in enumerate(zip(sample, results)):
            try:
                extracted = str(res["answer"])
                score = self.bench.score(t, f"ANSWER: {extracted}")
            except Exception as e:  # noqa: BLE001
                info.append(f"Valid Sample {i}:\n{repr(e)}\nModel Output: {res}\n")
                acc.append(0.0)
                continue
            acc.append(score)
            info.append(f"Valid Sample {i}:\n{t.input}\nModel Output: {res}\nModel Answer: {extracted}\nCorrect Answer: {t.target}\nIs Correct: {score}\n")
        valid = sum(acc) / len(acc) if acc else 0.0
        self.evals.append(valid)
        feedback = f"Valid Accuracy: {valid}\n" + ("Evaluation Info:\n" + "\n".join(info) if self.include_info else "")
        return feedback, 0.0


EVALUATE_SOURCE = '''def action_evaluate_on_task(task, solver):
    """
    Evaluate the current solver on the goal task samples and return the evaluation feedback.

    Returns:
        feedback (str): Evaluation feedback including valid set accuracy, valid sample inputs, model outputs and valid sample answer.
    """
    feedback, acc = task.evaluate(solver)
    return feedback
'''


def initial_solver_source(bench: str, model: str) -> str:
    """上游 task_<基准>.py 里的初始 solver，原样取出，只把写死的型号换成执行模型的型号。"""
    text = Path("src", FAMILIES[bench] + ".py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "solver")
    src = re.sub(r'model\s*=\s*"[^"]*"', f'model="{model}"', ast.get_source_segment(text, node))
    if bench == "gsm_hard":  # GSM-Hard 的答案不一定是整数；上游没有这个基准的 solver，借用 MGSM 的并放宽这一句
        src = src.replace("The answer MUST be an integer.", "The answer MUST be a number.")
    return src + "\n"


def boot(p: dict, mode: str):
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    os.environ["OPENAI_API_KEY"] = "rsi-eval-proxy"
    import agent_module as am  # 上游：导入时就创建 self_evolving_agent

    agent = am.self_evolving_agent
    client = GodelClient(mode, int(p.get("max_calls", 10)))
    agent.client = client
    model = client.models["executor"]

    am.action_adjust_logic("agent_module", "solver", initial_solver_source(p["bench"], model), target_type="function", operation="modify")
    task = RsiTask(p["bench"], p.get("tasks", []), int(p.get("feedback_batch", 20)), str(p.get("seed", 0)), int(p.get("workers", 8)))
    agent.goal_task = task
    ns: dict = {}
    exec(compile(EVALUATE_SOURCE, "running.agent_module.action_evaluate_on_task", "exec"), vars(am), ns)
    ns["action_evaluate_on_task"].__source__ = EVALUATE_SOURCE
    am.action_evaluate_on_task = ns["action_evaluate_on_task"]

    for f in agent.action_functions:  # 告诉 agent 实情：solver 用的模型是固定的
        if f["function"]["name"] == "action_call_json_format_llm":
            f["function"]["parameters"]["properties"]["model"]["enum"] = [model]
    agent.goal_prompt += (f"\n\n### **Runtime Note**:\n\n+ Every LLM call made through `action_call_json_format_llm` or `action_call_llm` "
                          f"(including the calls inside `solver`) is served by `{model}`, whatever `model` argument is passed. Changing the `model` argument has no effect.\n")

    rec = Recorder(am, client)
    rec.install()
    rec.replay(p.get("log", []))
    return am, client, rec, task


# --------------------------------------------------------------------------
# 命令
# --------------------------------------------------------------------------
def _sources(am) -> dict:
    import logic

    out = {}
    targets = {k: v for k, v in vars(am).items() if k.startswith("action_") and callable(v)}
    targets["Agent"] = am.Agent
    for name, obj in sorted(targets.items()):
        try:
            out[name] = logic.get_source_code(obj, name)
        except Exception as e:  # noqa: BLE001
            out[name] = f"<unavailable: {e!r}>"
    out["goal_prompt"] = am.self_evolving_agent.goal_prompt
    return out


def _solver_source(am) -> str:
    try:
        return am.action_read_logic("agent_module", "solver")
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e!r}>"


def cmd_step(p: dict) -> dict:
    from openai.types.chat import ChatCompletionMessage

    am, client, rec, task = boot(p, "step")
    rec.frozen = not p.get("modify_improver", True)
    agent = am.self_evolving_agent
    if p.get("history"):
        agent.optimize_history = [ChatCompletionMessage.model_validate(m["__msg__"]) if "__msg__" in m else m for m in p["history"]]
    else:
        agent.reinit()
    reason = "max_calls"
    try:
        agent.evolve()
        reason = "returned"
    except StepDone:
        pass
    except BudgetStop:
        raise
    except SystemExit as e:  # 上游在模型调用失败或 evolve 满 100 次时 sys.exit
        reason = f"exit({e.code})"
    except Exception as e:  # noqa: BLE001
        reason = f"error: {e!r}"[:500]
        import traceback
        traceback.print_exc()  # 调用栈进这一轮的日志，方便分清是上游代码还是 agent 改出来的代码抛的
    agent = am.self_evolving_agent
    history = [m if isinstance(m, dict) else {"__msg__": m.model_dump(mode="json")} for m in agent.optimize_history]
    return {"log": rec.log, "history": json.loads(json.dumps(history, default=str)), "solver_source": _solver_source(am),
            "improver_sources": _sources(am), "calls": client.improver_calls, "exit": reason, "evals": task.evals}


def cmd_solve(p: dict) -> dict:
    am, client, rec, task = boot(p, "solve")
    try:
        res = am.solver(am.self_evolving_agent, p["input"])
        return {"answer": str(res["answer"])}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:2000]}


if __name__ == "__main__":
    main({"step": cmd_step, "solve": cmd_solve})
