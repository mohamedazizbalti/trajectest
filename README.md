# trajectest

Test what your LangGraph agent **does**, not just what it says.

trajectest runs your agent against test cases written in YAML and checks:

- **Tool calls**: the right tools, with the right arguments, in the right order
- **Human approval steps**: the agent pauses at an interrupt, gets a human answer, and resumes correctly
- **Forbidden actions**: tools the agent must never call
- **The final answer**: scored by an LLM judge

Any model works: OpenAI-compatible or Anthropic Messages-compatible APIs, called directly or through a gateway.

> Status: in development (v0.1).

## Repository layout

| Path | What it is |
|---|---|
| `trajectest/` | The library |
| `examples/support_agent/` | A demo customer-support agent built with LangGraph, with its test cases |
| `tests/` | Tests for trajectest itself |
| `docs/` | Reference for writing test cases |

## License

MIT
