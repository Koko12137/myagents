import os
import getpass
import json
from typing import Optional, Union

from loguru import logger
from openai import AsyncOpenAI

from myagents.src.interface import LLM, Logger
from myagents.src.message import CompletionMessage, MessageRole, StopReason, ToolCallRequest, ToolCallResult, CompletionUsage
from myagents.src.utils.tools import Provider


def to_openai_dict(
    messages: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]
) -> list[dict[str, Union[str, dict]]]:
    """Convert the message to the OpenAI compatible messages dictionaries.
    
    Args:
        messages (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]): 
            The messages to convert.
            
    Returns:
        list[dict[str, Union[str, dict]]]: 
            The OpenAI compatible messages dictionaries.
    """
    # Create the generation history
    history = []
    for message in messages: 
        message_dict = {
            "role": message.role.value,
            "content": message.content
        }
        
        # This is only for OpenAI. 
        if isinstance(message, ToolCallResult):
            if message_dict['role'] == "tool":
                message_dict['tool_call_id'] = message.tool_call_id
        
        elif isinstance(message, CompletionMessage):
            # If the message is a tool call, add the tool call to the history
            if message.tool_calls != [] and message.tool_calls is not None:
                message_dict["tool_calls"] = []
                
                for tool_call in message.tool_calls:
                    message_dict["tool_calls"].append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name, 
                            "arguments": json.dumps(tool_call.args, ensure_ascii=False)
                        }
                    })
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
            
        history.append(message_dict)
        
    return history


class OpenAiLLM(LLM):
    provider: Provider
    model: str
    base_url: str
    temperature: float
    custom_logger: Logger
    debug: bool
    
    def __init__(
        self, 
        model: str, 
        base_url: str, 
        temperature: float = 1.0, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
        **kwargs
    ) -> None:
        """Initialize the OpenAiLLM.
        
        Args:
            model (str): 
                The model of the LLM.
            base_url (str): 
                The base URL of the LLM.
            temperature (float, defaults to 1.0): 
                The temperature of the LLM.
            custom_logger (Logger, defaults to logger): 
                The custom logger. If not provided, the default loguru logger will be used.
            debug (bool, defaults to False): 
                The debug flag.
            **kwargs: 
                The additional keyword arguments.
        """
        self.provider = Provider.OPENAI
        
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.custom_logger = custom_logger
        self.debug = debug
        if self.debug:
            self.custom_logger.enable("myagents.llms.openai")
        
        # Initialize the client
        api_key_field = kwargs.get("api_key_field", "OPENAI_KEY")
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=os.getenv(api_key_field) or getpass.getpass("Enter your OpenAI API key: "), 
        )
        
        # Check extra body for requests
        self.extra_body = kwargs.get("extra_body", {})
    
    async def completion(
        self, 
        messages: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]], 
        available_tools: Optional[list[dict[str, str]]] = None, 
    ) -> CompletionMessage:
        """Completion the messages.

        Args:
            messages (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]): 
                The messages to complete.
            available_tools (Optional[list[dict[str, str]]], optional): 
                The available tools. Defaults to None.

        Raises:
            ValueError: 
                The value error raised by the unsupported message type.

        Returns:
            CompletionMessage: 
                The completed message.
        """
        # Check tools are provided
        if available_tools is not None and len(available_tools) == 0:
            available_tools = None
        
        # Create the generation history
        history = to_openai_dict(messages)
            
        # Call for the completion
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=history,
            temperature=self.temperature,
            tools=available_tools, 
            extra_body=self.extra_body, 
        )
        content = response.choices[0].message.content
        self.custom_logger.info(f"\n{content}")
        # Get the usage
        usage = response.usage
        # Create the usage
        usage = CompletionUsage(
            prompt_tokens=usage.prompt_tokens or -100,
            completion_tokens=usage.completion_tokens or -100,
            total_tokens=usage.total_tokens or -100
        )
        # Log the usage
        self.custom_logger.info(f"Usage: {usage}")
        
        # Extract tool calls from response
        tool_calls = []
        if response.choices[0].message.tool_calls is not None:
            # Traverse all the tool calls and log the tool call
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                # Log the tool call
                self.custom_logger.info(f"Tool call {i + 1}: {tool_call}")
                
                # Create the tool call request
                tool_calls.append(ToolCallRequest(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    type="function", 
                    args=json.loads(tool_call.function.arguments)
                ))
        
        # Extract Finish reason
        if response.choices[0].finish_reason == "tool_calls":
            stop_reason = StopReason.TOOL_CALL
        elif response.choices[0].finish_reason == "stop":
            stop_reason = StopReason.STOP
        else:
            stop_reason = StopReason.NONE
        
        # Return the response
        return CompletionMessage(
            role=MessageRole.ASSISTANT, 
            content=content, 
            tool_calls=tool_calls, 
            stop_reason=stop_reason, 
            usage=usage, 
        )
