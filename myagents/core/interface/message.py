from typing import Protocol, runtime_checkable, Union, Any


@runtime_checkable
class ToolCallRequest(Protocol):
    """ToolCallRequest is a protocol for the tool call request.
    
    Attributes:
        name (str):
            The name of the tool.
        args (dict):
            The arguments of the tool.
    """
    name: str
    args: dict


@runtime_checkable
class Message(Protocol):
    """Message is a protocol for the message.
    
    Attributes:
        role (str):
            The role of the message.
        content (Union[str, Any]):
            The content of the message. 
        tool_calls (list[ToolCallRequest]):
            The tool calls of the message.
    """
    role: str
    content: Union[str, Any]
    tool_calls: list[ToolCallRequest]
