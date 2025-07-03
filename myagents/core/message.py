from enum import Enum
from uuid import uuid4
from typing import Optional

from pydantic import BaseModel, Field


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class StopReason(Enum):
    """The stop reason of the message.
    
    Attributes:
        STOP (str):
            The stop reason of the message.
        TOOL_CALL (str):
            The tool call reason of the message.
        LENGTH (str):
            The length reason of the message.
        CONTENT_FILTER (str):
            The content filter reason of the message.
        NONE (str):
            The none reason of the message.
    """
    STOP = "stop"
    TOOL_CALL = "tool_call"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    NONE = "none"
    

class CompletionUsage(BaseModel):
    """Usage of the completion.
    
    Attributes:
        prompt_tokens (int, defaults to -100):
            The number of prompt tokens. If the value is -100, it means the usage is not available.
        completion_tokens (int, defaults to -100):
            The number of completion tokens. If the value is -100, it means the usage is not available.
        total_tokens (int, defaults to -100):
            The total number of tokens. If the value is -100, it means the usage is not available.
    """
    prompt_tokens: int = Field(description="The number of prompt tokens.", default=-100)
    completion_tokens: int = Field(description="The number of completion tokens.", default=-100)
    total_tokens: int = Field(description="The total number of tokens.", default=-100)
    

class CompletionMessage(BaseModel):
    """Message from user, tool call result, assistant, system.
    
    Attributes:
        id (str, defaults to uuid4().hex): 
            The unique id of the message.
        role (MessageRole): 
            The role of the message. 
        content (str, optional, defaults to None):
            The content of the message. 
        tool_calls (list[ToolCallRequest], optional, defaults to None):
            The tool calls of the message. 
        stop_reason (StopReason, defaults to StopReason.NONE):
            The stop reason of the message.
        usage (CompletionUsage, optional, defaults to None):
            The usage of the message.
    """
    # The unique id of the message
    id: str = Field(default_factory=lambda: uuid4().hex)
    # The role of the message
    role: MessageRole = Field(description="The role of the message.")
    # The content of the message
    content: Optional[str] = Field(description="The content of the message.", default="")
    # The tool calls of the message
    tool_calls: Optional[list['ToolCallRequest']] = Field(description="The tool calls of the message.", default="")
    # The stop reason of the message
    stop_reason: StopReason = Field(description="The stop reason of the message.", default=StopReason.NONE)
    # Usage of the message
    usage: Optional[CompletionUsage] = Field(description="The usage of the message.", default=None)


class ToolCallRequest(BaseModel):
    """ToolCallRequest send by the llm to mcp server.
    
    Attributes:
        id (str):
            The unique id of the tool call request.
        name (str):
            The name of the tool call request.
        args (dict):
            The arguments of the tool call request.
    """
    # The unique id of the tool call request
    id: str = Field(description="The unique id of the tool call request.")
    # The name of the tool call request
    name: str = Field(description="The name of the tool call request.")
    # The arguments of the tool call request
    args: dict = Field(description="The arguments of the tool call request.")


class ToolCallResult(BaseModel):
    """ToolCallResult from the mcp server.
    
    Attributes:
        role (MessageRole):
            The role of the tool call result. Do not specify this field. 
        tool_call_id (str):
            The id of the tool call request.
        content (str):
            The content of the tool call result. 
        is_error (bool, defaults to False):
            Whether the tool call result is an error. 
    """
    # The role of the tool call result
    role: MessageRole = Field(
        description="The role of the tool call result. Do not specify this field.", 
        default_factory=lambda: MessageRole.TOOL
    )
    # The id of the tool call request
    tool_call_id: str = Field(description="The id of the tool call request.")
    # The content of the tool call result
    content: str = Field(description="The content of the tool call result.")
    # The error of the tool call result
    is_error: bool = Field(description="Whether the tool call result is an error.", default=False)
