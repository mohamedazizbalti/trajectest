from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ToolOrder = Literal["exact", "in_order", "any_order"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedToolCall(StrictModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ExpectedInterrupt(StrictModel):
    contains: str | None = None


class Step(StrictModel):
    expect_tools: list[ExpectedToolCall] = Field(default_factory=list)
    tool_order: ToolOrder = "in_order"
    expect_interrupt: ExpectedInterrupt | None = None
    resume: Any = None


class AgentTestCase(StrictModel):
    name: str
    agent: str
    input: str
    steps: list[Step] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    judge: str | None = None
    source_file: Path | None = Field(default=None, exclude=True)


def load_test_cases(path: str | Path) -> list[AgentTestCase]:
    path = Path(path)
    files = sorted([*path.glob("**/*.yaml"), *path.glob("**/*.yml")]) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .yaml test cases found in {path}")
    return [load_test_case(file) for file in files]


def load_test_case(file: Path) -> AgentTestCase:
    content = yaml.safe_load(file.read_text())
    if not isinstance(content, dict):
        raise ValueError(f"{file} must contain one test case as a YAML mapping")
    return AgentTestCase(**content, source_file=file)
