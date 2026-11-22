import importlib
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str
    result: str | None = None


@dataclass
class Interrupt:
    value: Any


@dataclass
class Trajectory:
    events: list[ToolCall | Interrupt] = field(default_factory=list)
    final_answer: str | None = None
    pending_interrupt: Interrupt | None = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [event for event in self.events if isinstance(event, ToolCall)]

    @property
    def interrupts(self) -> list[Interrupt]:
        return [event for event in self.events if isinstance(event, Interrupt)]


def load_graph(agent_reference: str, checkpointer: BaseCheckpointSaver | None = None, **build_options):
    module_name, _, factory_name = agent_reference.partition(":")
    if not module_name or not factory_name:
        raise ValueError(f"Agent reference '{agent_reference}' must look like 'package.module:build_graph'")
    build_graph = getattr(importlib.import_module(module_name), factory_name)
    return build_graph(checkpointer=checkpointer or InMemorySaver(), **build_options)


class AgentRun:
    def __init__(self, graph):
        self.graph = graph
        self.thread = {"configurable": {"thread_id": f"trajectest-{uuid.uuid4()}"}}
        self.trajectory = Trajectory()
        self.messages_already_recorded = 0
        self.tool_calls_by_id: dict[str, ToolCall] = {}

    def send(self, user_message: str) -> Trajectory:
        return self.advance({"messages": [("user", user_message)]})

    def resume(self, human_answer: Any) -> Trajectory:
        if self.trajectory.pending_interrupt is None:
            raise RuntimeError("The agent is not waiting for a human answer, so there is nothing to resume")
        return self.advance(Command(resume=human_answer))

    def advance(self, graph_input) -> Trajectory:
        state = self.graph.invoke(graph_input, self.thread)
        self.record_new_messages(state["messages"])

        # ponytail: only the first interrupt is resumable; parallel interrupts need resume-by-id
        interrupts = [Interrupt(value=interrupt.value) for interrupt in state.get("__interrupt__", [])]
        self.trajectory.events.extend(interrupts)
        self.trajectory.pending_interrupt = interrupts[0] if interrupts else None
        return self.trajectory

    def record_new_messages(self, messages: list) -> None:
        for message in messages[self.messages_already_recorded:]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    tool_call = ToolCall(name=call["name"], arguments=call["args"], call_id=call["id"])
                    self.tool_calls_by_id[call["id"]] = tool_call
                    self.trajectory.events.append(tool_call)
                if not message.tool_calls and message.text:
                    self.trajectory.final_answer = message.text
            elif isinstance(message, ToolMessage) and message.tool_call_id in self.tool_calls_by_id:
                self.tool_calls_by_id[message.tool_call_id].result = message.text
        self.messages_already_recorded = len(messages)
