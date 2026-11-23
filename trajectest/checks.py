import re
from dataclasses import dataclass
from typing import Any

from trajectest.runner import ToolCall
from trajectest.test_case import ExpectedToolCall, ToolOrder


@dataclass
class CheckResult:
    check: str
    passed: bool
    detail: str = ""


def check_tool_calls(expected: list[ExpectedToolCall], actual: list[ToolCall], order: ToolOrder) -> CheckResult:
    check = f"tool calls ({order})"
    if order == "exact":
        matched = len(expected) == len(actual) and all(map(call_matches, expected, actual))
    elif order == "in_order":
        matched = is_in_order(expected, actual)
    else:
        matched = all_matched_in_any_order(expected, actual)

    if matched:
        return CheckResult(check, True)
    wanted = ", ".join(describe(call.tool, call.args) for call in expected) or "no tools"
    called = ", ".join(describe(call.name, call.arguments) for call in actual) or "no tools"
    return CheckResult(check, False, f"expected {wanted}; the agent called {called}")


def check_forbidden_tools(forbidden: list[str], actual: list[ToolCall]) -> CheckResult:
    used = sorted({call.name for call in actual if call.name in forbidden})
    if used:
        return CheckResult("forbidden tools", False, f"the agent called forbidden tools: {', '.join(used)}")
    return CheckResult("forbidden tools", True)


def check_max_tool_calls(limit: int, actual: list[ToolCall]) -> CheckResult:
    if len(actual) > limit:
        return CheckResult("max tool calls", False, f"{len(actual)} tool calls, the limit is {limit}")
    return CheckResult("max tool calls", True)


def call_matches(expected: ExpectedToolCall, actual: ToolCall) -> bool:
    if expected.tool != actual.name:
        return False
    return all(
        name in actual.arguments and value_matches(wanted, actual.arguments[name])
        for name, wanted in expected.args.items()
    )


def value_matches(wanted: Any, actual: Any) -> bool:
    if isinstance(wanted, dict) and set(wanted) == {"regex"}:
        return re.search(wanted["regex"], str(actual)) is not None
    return wanted == actual


def is_in_order(expected: list[ExpectedToolCall], actual: list[ToolCall]) -> bool:
    remaining_calls = iter(actual)
    return all(any(call_matches(wanted, call) for call in remaining_calls) for wanted in expected)


def all_matched_in_any_order(expected: list[ExpectedToolCall], actual: list[ToolCall]) -> bool:
    # ponytail: greedy matching, can miss a valid assignment when patterns overlap; use bipartite matching if that bites
    unused_calls = list(actual)
    for wanted in expected:
        match = next((call for call in unused_calls if call_matches(wanted, call)), None)
        if match is None:
            return False
        unused_calls.remove(match)
    return True


def describe(tool_name: str, arguments: dict) -> str:
    rendered_arguments = ", ".join(f"{name}={value!r}" for name, value in arguments.items())
    return f"{tool_name}({rendered_arguments})"
