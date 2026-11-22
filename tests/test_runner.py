import pytest
from langchain_core.messages import AIMessage

from trajectest.runner import AgentRun, Interrupt, ToolCall, load_graph
from tests.scripted_model import scripted_model, tool_call

SUPPORT_AGENT = "examples.support_agent.agent:build_graph"


def test_records_tool_calls_interrupt_and_final_answer_across_a_resume():
    model = scripted_model(
        tool_call("get_order", "call_1", order_id="1042"),
        tool_call("issue_refund", "call_2", order_id="1042", amount_euros=249),
        AIMessage(content="Your refund of €249 is done."),
    )
    run = AgentRun(load_graph(SUPPORT_AGENT, chat_model=model))

    paused = run.send("Refund order 1042, it arrived broken")
    assert [call.name for call in paused.tool_calls] == ["get_order", "issue_refund"]
    assert paused.pending_interrupt.value["type"] == "refund_approval"
    assert paused.final_answer is None

    finished = run.resume({"approved": True})
    assert finished.pending_interrupt is None
    assert [type(event) for event in finished.events] == [ToolCall, ToolCall, Interrupt]
    assert finished.tool_calls[0].arguments == {"order_id": "1042"}
    assert finished.tool_calls[1].result == "Refunded €249 for order 1042"
    assert finished.final_answer == "Your refund of €249 is done."


def test_resume_without_a_pending_interrupt_is_an_error():
    run = AgentRun(load_graph(SUPPORT_AGENT, chat_model=scripted_model(AIMessage(content="Hello"))))
    run.send("Hi")
    with pytest.raises(RuntimeError, match="nothing to resume"):
        run.resume({"approved": True})


def test_agent_reference_must_name_a_factory():
    with pytest.raises(ValueError, match="package.module:build_graph"):
        load_graph("examples.support_agent.agent")
