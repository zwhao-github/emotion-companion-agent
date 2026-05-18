from langchain.agents import create_agent

from agent.tools import ALL_MIDDLEWARE, ALL_TOOLS
from model.factory import chat_model


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            tools=ALL_TOOLS,
            middleware=ALL_MIDDLEWARE,
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # context 供中间件切换提示词：report / crisis
        for chunk in self.agent.stream(
            input_dict,
            stream_mode="values",
            context={"report": False, "crisis": False},
        ):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == "__main__":
    agent = ReactAgent()

    for chunk in agent.execute_stream("感觉自己干啥啥不行应该怎么办"):
        print(chunk, end="", flush=True)
