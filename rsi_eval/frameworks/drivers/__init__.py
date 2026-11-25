"""驱动脚本：在子进程里导入上游框架的代码并调用它。

每个脚本用 `python -m rsi_eval.frameworks.drivers.<名字> <命令> --in in.json --out out.json` 运行，
由 ExternalRepoAdapter.run_driver() 在计量代理后面启动。上游代码本身一个字节都不改，
需要的适配（换模型客户端、换评测题目、限制一轮的调用次数）都在导入之后对内存里的对象做。
"""
