# emotion-companion-agent

基于 RAG 与 LangChain Agent 的**抑郁情绪陪伴 AI** 原型：支持性对话、知识库检索、情绪记录整理、月度报告生成，以及自伤/自杀风险的分流与危机提示词切换。

> **声明**：本文件夹下面提供的数据均为模拟数据，**不能用于医学诊断或治疗**，不能替代心理咨询师、精神科医生或紧急救援服务。

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **RAG 知识库** | 从 `data/` 加载 txt/pdf，写入 Chroma 向量库，按问题检索并总结参考资料 |
| **ReAct Agent** | 通义千问 + 工具调用，完成陪伴对话与任务编排 |
| **安全分流** | `assess_risk_level` 识别风险等级；「高 / 危机」时自动切换危机应对提示词 |
| **报告场景** | 调用 `fill_context_for_report` 后切换报告专用提示词，结合 CSV 情绪记录生成月度总结 |
| **Web 界面** | Streamlit 聊天页（`app.py`） |

---

## 项目结构

```
emotion-companion-agent/
├── app.py                      # Streamlit 前端
├── agent/
│   ├── react_agent.py          # Agent 入口（create_agent + 流式输出）
│   └── tools/
│       ├── agent_tools.py      # 全部工具实现
│       └── middleware.py       # 工具监控、日志、动态提示词切换
├── rag/
│   ├── vector_store.py         # 文档加载、分片、入库 Chroma
│   └── rag_service.py          # 检索 + RAG 总结链
├── model/
│   └── factory.py              # 通义对话模型与 Embedding
├── prompts/                    # 主 Agent / RAG / 报告 / 危机 提示词
├── config/                     # YAML 配置
├── data/                       # 知识库原文 + 模拟情绪记录 CSV
└── utils/                      # 配置、日志、路径、文件处理、提示词加载
```

---

## Agent 工具

工具定义与说明见 `depression_rag_agent_demo/agent_tools_schema.json`（仓库内为参考 schema；实际实现以 `agent/tools/agent_tools.py` 为准）。

| 工具名 | 作用 |
|--------|------|
| `rag_summarize` | 从知识库检索并总结心理教育、支持策略等内容 |
| `assess_risk_level` | 自伤/自杀风险初筛（规则 + JSON 输出），触发危机提示词切换 |
| `generate_emotion_record` | 将对话整理为结构化情绪记录 |
| `fetch_user_emotion_records` | 按用户 ID、月份查询 `data/external/emotion_records.csv` |
| `get_current_month` | 返回当前月份 `YYYY-MM` |
| `fill_context_for_report` | 标记进入报告场景，切换报告提示词 |
| `grounding_response_style` | 返回推荐回复风格（安静陪伴 / 情绪梳理等） |

---

## 提示词与中间件

提示词路径在 `config/prompts.yml` 中配置：

| 场景 | 文件 | 切换方式 |
|------|------|----------|
| 默认陪伴 | `prompts/main_prompt_depression_companion.txt` | 默认 |
| 危机应对 | `prompts/crisis_prompt_depression.txt` | `assess_risk_level` 返回「高」或「危机」 |
| 月度报告 | `prompts/report_prompt_depression.txt` | 调用 `fill_context_for_report` |
| RAG 总结 | `prompts/rag_summarize_depression.txt` | 仅 `rag_summarize` 工具链使用 |

中间件（`agent/tools/middleware.py`）通过单个 `prompt_switch` 动态切换提示词，优先级：**危机 > 报告 > 主提示词**（必须始终返回有效字符串，不能返回 `None`）。

---

## 环境要求

- Python 3.10+
- [阿里云百炼 / DashScope](https://help.aliyun.com/zh/model-studio/) API Key（通义对话与 Embedding）

主要依赖（请自行安装，可按环境整理为 `requirements.txt`）：

- `langchain`、`langchain-core`、`langchain-community`、`langchain-chroma`、`langgraph`
- `dashscope`（通义）
- `chromadb`
- `pyyaml`
- `streamlit`
- `pypdf`（加载 PDF 知识库）

---

## 配置

1. 设置 DashScope 密钥，例如：

   ```bash
   DASHSCOPE_API_KEY=你的密钥
   ```

2. 按需修改配置文件：

   | 文件 | 内容 |
   |------|------|
   | `config/rag.yml` | 对话模型名、Embedding 模型名 |
   | `config/chroma.yml` | 向量库路径、分片参数、`data/` 路径、允许的文件类型 |
   | `config/prompts.yml` | 各场景提示词路径 |
   | `config/agent.yml` | 情绪记录 CSV 路径 |

3. 将知识库文件放入 `data/`（支持 `.txt`、`.pdf`）。

---

## 快速开始

在**项目根目录**执行（确保环境配置正确且已配置 API Key）。

### 1. 构建向量库（首次或更新知识库后）

```bash
python -m rag.vector_store
```

会读取 `data/` 下允许类型的文件，按 MD5 去重后写入 `chroma_db/`（目录默认被 git 忽略）。

### 2. 测试 RAG 总结

```bash
python -m rag.rag_service
```

### 3. 命令行测试 Agent

```bash
python -m agent.react_agent
```

### 4. 启动 Web 界面

```bash
streamlit run app.py
```

> 请始终在项目根目录使用 `python -m 包.模块` 方式运行，避免 `python rag/xxx.py` 导致导入路径错误。

---

## 数据说明

| 路径 | 说明 |
|------|------|
| `data/depression_companion_knowledge_200.txt` 等 | RAG 知识库模拟资料 |
| `data/external/emotion_records.csv` | 模拟用户月度情绪记录（报告工具使用） |
| `md5.text` | 已入库文件的 MD5 记录，用于去重 |

---

## 日志

运行日志写入 `logs/`（默认按日期命名）。

---

## 注意事项

1. **非诊疗系统**：禁止将输出当作诊断或用药依据；高风险用户应引导至现实支持与紧急服务。
2. **当前 Web 单轮调用**：`app.py` 每次仅将用户当前输入传给 Agent，未自动携带 Streamlit 中的完整聊天历史；多轮记忆需自行扩展 `ReactAgent.execute_stream`。
3. **会话 context**：每次流式调用默认 `report=False`、`crisis=False`；跨轮保持报告/危机模式需在应用层维护 `context` 并传入 `agent.stream()`。
4. **原型数据**：CSV 与知识库内容为模拟数据，仅供演示。

---

## License

见仓库约定；若未单独声明，以项目维护者为准。
