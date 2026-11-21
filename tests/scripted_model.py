from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, call_id: str, **arguments) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": call_id}])


def scripted_model(*replies: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(replies))
