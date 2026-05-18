import json
from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import before_model, dynamic_prompt, ModelRequest, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_crisis_prompts, load_report_prompts, load_system_prompts

_CRISIS_RISK_LEVELS = frozenset({"高", "危机"})


def _apply_risk_context(runtime: Runtime, tool_content: str) -> None:
    try:
        data = json.loads(tool_content)
    except json.JSONDecodeError:
        logger.warning("[tool monitor]assess_risk_level 返回内容非 JSON，跳过危机提示词切换")
        return

    risk_level = data.get("风险等级")
    if risk_level in _CRISIS_RISK_LEVELS:
        runtime.context["crisis"] = True
        runtime.context["risk_level"] = risk_level
        logger.info(f"[tool monitor]风险等级为{risk_level}，已切换危机应对提示词")


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call["name"] == "fill_context_for_report":
            request.runtime.context["report"] = True

        if request.tool_call["name"] == "assess_risk_level" and isinstance(result, ToolMessage):
            _apply_risk_context(request.runtime, result.content)

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
):
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    last_message = state["messages"][-1]
    content = getattr(last_message, "content", "")
    if isinstance(content, str):
        preview = content.strip()
    else:
        preview = str(content)
    logger.debug(f"[log_before_model]{type(last_message).__name__} | {preview}")
    return None


@dynamic_prompt
def prompt_switch(request: ModelRequest):
    """优先级：危机 > 报告 > 主提示词。不可返回 None，否则会触发 SystemMessage 校验错误。"""
    ctx = request.runtime.context
    if ctx.get("crisis"):
        return load_crisis_prompts()
    if ctx.get("report"):
        return load_report_prompts()
    return load_system_prompts()


ALL_MIDDLEWARE = [
    monitor_tool,
    log_before_model,
    prompt_switch,
]
