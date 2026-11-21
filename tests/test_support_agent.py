from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from examples.support_agent.agent import build_graph
from tests.scripted_model import scripted_model, tool_call


def test_refund_over_limit_pauses_for_approval_then_completes():
    model = scripted_model(
        tool_call("get_order", "call_1", order_id="1042"),
        tool_call("issue_refund", "call_2", order_id="1042", amount_euros=249),
        AIMessage(content="Your refund of €249 is done."),
    )
    graph = build_graph(checkpointer=InMemorySaver(), chat_model=model)
    thread = {"configurable": {"thread_id": "refund-over-limit"}}

    paused = graph.invoke({"messages": [("user", "Refund order 1042, it arrived broken")]}, thread)
    assert paused["__interrupt__"][0].value["type"] == "refund_approval"

    finished = graph.invoke(Command(resume={"approved": True}), thread)
    tool_results = [message.content for message in finished["messages"] if message.type == "tool"]
    assert tool_results[-1] == "Refunded €249 for order 1042"
    assert finished["messages"][-1].content == "Your refund of €249 is done."


def test_small_refund_needs_no_approval():
    model = scripted_model(
        tool_call("issue_refund", "call_1", order_id="1041", amount_euros=59),
        AIMessage(content="Refunded."),
    )
    graph = build_graph(checkpointer=InMemorySaver(), chat_model=model)

    result = graph.invoke({"messages": [("user", "Refund order 1041")]}, {"configurable": {"thread_id": "small-refund"}})
    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "Refunded."
