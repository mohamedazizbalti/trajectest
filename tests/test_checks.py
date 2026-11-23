from trajectest.checks import check_forbidden_tools, check_max_tool_calls, check_tool_calls
from trajectest.runner import ToolCall
from trajectest.test_case import ExpectedToolCall

AGENT_CALLS = [
    ToolCall("get_order", {"order_id": "1042"}, "call_1"),
    ToolCall("escalate_to_human", {"reason": "customer is angry"}, "call_2"),
    ToolCall("issue_refund", {"order_id": "1042", "amount_euros": 249.0}, "call_3"),
]


def expected(tool: str, **args) -> ExpectedToolCall:
    return ExpectedToolCall(tool=tool, args=args)


def test_in_order_allows_other_calls_in_between_but_not_reordering():
    in_order = [expected("get_order", order_id="1042"), expected("issue_refund", amount_euros=249)]
    assert check_tool_calls(in_order, AGENT_CALLS, "in_order").passed
    assert not check_tool_calls(list(reversed(in_order)), AGENT_CALLS, "in_order").passed


def test_exact_requires_the_full_sequence():
    assert not check_tool_calls([expected("get_order"), expected("issue_refund")], AGENT_CALLS, "exact").passed
    full = [expected("get_order"), expected("escalate_to_human"), expected("issue_refund")]
    assert check_tool_calls(full, AGENT_CALLS, "exact").passed


def test_any_order_ignores_sequence():
    assert check_tool_calls([expected("issue_refund"), expected("get_order")], AGENT_CALLS, "any_order").passed


def test_arguments_match_partially_or_by_regex():
    assert check_tool_calls([expected("escalate_to_human", reason={"regex": "angry"})], AGENT_CALLS, "in_order").passed
    result = check_tool_calls([expected("get_order", order_id="9999")], AGENT_CALLS, "in_order")
    assert not result.passed
    assert "get_order(order_id='9999')" in result.detail and "get_order(order_id='1042')" in result.detail


def test_forbidden_tools_and_max_tool_calls():
    assert not check_forbidden_tools(["issue_refund", "cancel_order"], AGENT_CALLS).passed
    assert check_forbidden_tools(["cancel_order"], AGENT_CALLS).passed
    assert not check_max_tool_calls(2, AGENT_CALLS).passed
    assert check_max_tool_calls(3, AGENT_CALLS).passed
