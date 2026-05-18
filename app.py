import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("情绪陪伴AI")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_chunks: list[str] = []

    def capture(generator, cache_list):
        """stream_mode=values 时每次 chunk 常为累积全文，只向前端输出增量避免重复。"""
        seen = ""
        for chunk in generator:
            cache_list.append(chunk)
            if chunk.startswith(seen):
                delta = chunk[len(seen):]
                seen = chunk
            else:
                delta = chunk
                seen += chunk
            for char in delta:
                yield char

    with st.spinner("正在思考..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)
        st.chat_message("assistant").write_stream(capture(res_stream, response_chunks))

    # execute_stream 在 values 模式下通常每次产出累积全文，取最后一块即可
    assistant_text = (
        response_chunks[-1].strip()
        if response_chunks
        else "抱歉，我暂时无法生成回复，请稍后再试。"
    )

    st.session_state["message"].append({"role": "assistant", "content": assistant_text})
    st.rerun()
