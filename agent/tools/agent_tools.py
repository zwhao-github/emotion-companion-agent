import csv
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

_rag_service: RagSummarizeService | None = None
_emotion_records_cache: dict[str, dict[str, dict[str, str]]] | None = None

_CRISIS_KEYWORDS = (
    "不想活", "不想活了", "想死", "自杀", "自尽", "了结", "结束生命",
    "跳楼", "割腕", "上吊", "服药自杀", "安眠药", "农药",
    "具体计划", "今晚就", "准备好了", "遗书", "告别", "永别",
    "无法保证安全", "保证不了安全",
)
_HIGH_RISK_KEYWORDS = (
    "自伤", "伤害自己", "想消失", "活着没意思", "活不下去",
    "撑不下去", "结束一切", "一了百了", "想离开这个世界",
)
_MEDIUM_RISK_KEYWORDS = (
    "绝望", "没有意义", "没意义", "看不到希望", "好想消失",
    "控制不住", "崩溃", "熬不住",
)

_STYLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "安静陪伴": ("只想有人陪", "别给建议", "不要建议", "听我说", "陪我就好", "不用说太多"),
    "情绪梳理": ("帮我梳理", "理清楚", "说不清楚", "情绪很乱", "怎么回事"),
    "问题解决": ("怎么办", "帮我想办法", "有什么办法", "该怎么做", "给点建议"),
    "转移注意": ("转移注意", "分散注意", "别想了", "做点别的", "分心"),
    "危机支持": ("不想活", "自杀", "自伤", "想死", "撑不住"),
}

_EMOTION_RECORD_PROMPT = PromptTemplate.from_template(
    """你是情绪记录整理助手。请根据对话内容输出结构化情绪记录（中文），不要诊断疾病。
必须包含以下字段，每字段单独一行：
主情绪:
强度(0-10):
触发因素:
身体感受:
应对方式:
风险提示:
下一步小目标:

用户ID: {user_id}
对话内容:
{conversation}
"""
)


def _get_rag_service() -> RagSummarizeService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RagSummarizeService()
    return _rag_service


def _load_emotion_records() -> dict[str, dict[str, dict[str, str]]]:
    global _emotion_records_cache
    if _emotion_records_cache is not None:
        return _emotion_records_cache

    data_path = Path(get_abs_path(agent_conf["external_data_path"]))
    if not data_path.is_file():
        raise FileNotFoundError(f"情绪记录数据文件不存在: {data_path}")

    records: dict[str, dict[str, dict[str, str]]] = {}
    with data_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            user_id = row["用户ID"].strip()
            month = row["时间"].strip()
            records.setdefault(user_id, {})[month] = {
                "用户画像": row["用户画像"].strip(),
                "情绪状态": row["情绪状态"].strip(),
                "触发与行为": row["触发与行为"].strip(),
                "应对与支持": row["应对与支持"].strip(),
                "风险提示": row["风险提示"].strip(),
                "时间": month,
            }

    _emotion_records_cache = records
    return records


def _collect_signals(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [kw for kw in keywords if kw in text]


def _infer_response_style(user_text: str) -> str:
    for style, keywords in _STYLE_KEYWORDS.items():
        if any(kw in user_text for kw in keywords):
            return style
    return "未知"


def _assess_risk_from_text(text: str) -> tuple[str, bool, list[str]]:
    crisis_signals = _collect_signals(text, _CRISIS_KEYWORDS)
    if crisis_signals:
        return "危机", True, crisis_signals

    high_signals = _collect_signals(text, _HIGH_RISK_KEYWORDS)
    if high_signals:
        return "高", True, high_signals

    medium_signals = _collect_signals(text, _MEDIUM_RISK_KEYWORDS)
    if medium_signals:
        return "中", False, medium_signals

    return "低", False, []


@tool(
    description="根据用户输入，从抑郁情绪陪伴知识库中检索并总结相关的心理教育、支持性回应、行为激活、睡眠作息、人际支持和安全边界资料。用于回答需要专业资料支撑但不涉及诊断的问题。",
)
def rag_summarize(query: str) -> str:
    return _get_rag_service().rag_summarize(query)


@tool(
    description="根据用户文本初步识别自伤/自杀/危机风险等级，并给出是否需要进入危机应对流程。该工具不做诊断，只做安全分流。",
)
def assess_risk_level(user_text: str, recent_context: str = "") -> str:
    combined = "\n".join(part for part in (user_text, recent_context) if part).strip()
    level, need_crisis_flow, signals = _assess_risk_from_text(combined)

    recommendations = {
        "危机": "立即进入危机应对：简短确认安全、鼓励联系身边可信任的人或当地紧急/危机热线，不争辩、不做长篇分析。",
        "高": "优先危机分流：温和确认是否有具体计划、工具、时间；鼓励移开危险物并联系现实支持。",
        "中": "加强关注：表达重视，建议联系可信任的人或专业支持，可继续陪伴但避免轻描淡写。",
        "低": "按常规陪伴流程回应；若持续低落或功能受损，可建议专业评估与持续记录。",
    }

    result = {
        "风险等级": level,
        "是否需要危机流程": need_crisis_flow,
        "检测到的信号": signals,
        "建议": recommendations[level],
        "说明": "本工具仅做安全分流，不构成医学诊断。",
    }
    return json.dumps(result, ensure_ascii=False)


@tool(
    description="把一轮或多轮对话整理成结构化情绪记录，用于后续报告、趋势分析或RAG增强。输出包含主情绪、强度、触发因素、身体感受、应对方式、风险提示和下一步小目标。",
)
def generate_emotion_record(conversation: str, user_id: str = "") -> str:
    prompt = _EMOTION_RECORD_PROMPT.format(
        conversation=conversation.strip(),
        user_id=user_id or "未提供",
    )
    try:
        response = chat_model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"[generate_emotion_record]生成失败: {e}")
        return (
            "主情绪: 未识别\n"
            "强度(0-10): 未知\n"
            "触发因素: 待补充\n"
            "身体感受: 待补充\n"
            "应对方式: 待补充\n"
            "风险提示: 建议人工复核\n"
            "下一步小目标: 与用户确认一个可执行的小步骤"
        )


@tool(
    description="按用户ID和月份查询模拟情绪记录，用于生成阶段性情绪陪伴报告。",
)
def fetch_user_emotion_records(user_id: str, month: str) -> str:
    records = _load_emotion_records()
    user_records = records.get(user_id.strip())
    if not user_records:
        logger.warning(f"[fetch_user_emotion_records]未找到用户: {user_id}")
        return ""

    record = user_records.get(month.strip())
    if not record:
        logger.warning(f"[fetch_user_emotion_records]未找到用户{user_id}在{month}的记录")
        return ""

    parts = [f"用户ID: {user_id}", f"月份: {month}"]
    for key, value in record.items():
        if key != "时间":
            parts.append(f"{key}:\n{value}")
    return "\n\n".join(parts)


@tool(
    description="获取当前月份字符串，格式YYYY-MM。当用户要求生成本月报告但没有指定月份时使用。",
)
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


@tool(
    description="当且仅当用户明确要求生成/查询个人情绪记录或月度报告时调用，用于触发报告场景的提示词切换和上下文注入。",
)
def fill_context_for_report() -> str:
    return "fill_context_for_report已调用"


@tool(
    description="根据用户当前偏好选择回复风格：安静陪伴、情绪梳理、问题解决、转移注意、危机支持。用于控制主Agent回复语气和长度。",
)
def grounding_response_style(user_text: str, preferred_style: str = "未知") -> str:
    style = preferred_style if preferred_style != "未知" else _infer_response_style(user_text)

    risk_level, need_crisis, _ = _assess_risk_from_text(user_text)
    if need_crisis or risk_level in ("危机", "高"):
        style = "危机支持"

    return json.dumps(
        {
            "推荐回复风格": style,
            "风险等级参考": risk_level,
        },
        ensure_ascii=False,
    )


ALL_TOOLS = [
    rag_summarize,
    assess_risk_level,
    generate_emotion_record,
    fetch_user_emotion_records,
    get_current_month,
    fill_context_for_report,
    grounding_response_style,
]
