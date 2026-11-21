from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from examples.support_agent.tools import SUPPORT_TOOLS
from trajectest.model_client import ModelConfig, create_chat_model

SYSTEM_PROMPT = (
    "You are the customer support agent of an online shop. "
    "Always look up an order with get_order before acting on it. "
    "Refund at most the order price. Cancel only orders that are not delivered yet. "
    "Escalate to a human for anything your tools cannot handle. "
    "Never promise anything your tools did not confirm."
)


def build_graph(checkpointer: BaseCheckpointSaver | None = None, chat_model: BaseChatModel | None = None):
    if chat_model is None:
        model_config = ModelConfig.from_env(prefix="SUPPORT_AGENT", default_model="gpt-4o-mini")
        chat_model = create_chat_model(model_config)
    model_with_tools = chat_model.bind_tools(SUPPORT_TOOLS)

    def assistant(state: MessagesState):
        reply = model_with_tools.invoke([("system", SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(SUPPORT_TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=checkpointer)
