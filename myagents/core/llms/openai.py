import os
import getpass
import json
from asyncio import Queue
from typing import Optional, Union

from loguru import logger
from openai import AsyncOpenAI
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import LLM, Provider
from myagents.core.messages import ToolCallRequest, ToolCallResult, AssistantMessage, StopReason, CompletionUsage, MessageRole, UserMessage, SystemMessage
from myagents.core.messages.openai_adapter import to_openai_dict


class OpenAiLLM(LLM):
    provider: Provider
    model: str
    base_url: str
    temperature: float
    
    def __init__(
        self, 
        model: str, 
        base_url: str, 
        temperature: float = 1.0, 
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
        
        # Initialize the client
        api_key_field = kwargs.get("api_key_field", "OPENAI_KEY")
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=os.getenv(api_key_field) or getpass.getpass(f"Enter your {api_key_field}: "), 
        )
        
        # Check extra keyword arguments for requests
        self.kwargs = {}
        for key, value in kwargs.items():
            if key == "extra_body":
                self.kwargs["extra_body"] = {key: value}
    
    async def completion(
        self, 
        messages: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]], 
        available_tools: Optional[list[dict[str, str]]] = None, 
        tool_choice: Union[str, FastMcpTool] = "auto", 
        format_json: bool = False, 
        stream: bool = False, 
        queue: Optional[Queue] = None, 
        **kwargs, 
    ) -> AssistantMessage:
        """Completion the messages.

        Args:
            messages (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]): 
                The messages to complete.
            available_tools (Optional[list[dict[str, str]]], optional): 
                The available tools. Defaults to None.
            tool_choice (str, defaults to "auto"):
                The tool choice to use for the agent. This is used to control the tool calling. 
                - "auto": The agent will automatically choose the tool to use. 
            format_json (bool, defaults to False):
                Whether to format the json in the assistant message.
            stream (bool, defaults to False):
                Whether to stream the completion.
            queue (Optional[Queue], optional):
                The queue to use for the completion.
            **kwargs:
                The additional keyword arguments.
                
        Raises:
            ValueError: 
                The value error raised by the unsupported message type.

        Returns:
            AssistantMessage: 
                The completed message.
        """
        kwargs = self.__prepare_kwargs(
            available_tools=available_tools, 
            tool_choice=tool_choice, 
            format_json=format_json, 
            **kwargs,
        )
        
        # Create the generation history
        history = to_openai_dict(messages)
            
        # Call for the completion
        response = await self.client.chat.completions.create(
            messages=history, 
            **kwargs, 
        )
        content = response.choices[0].message.content
        logger.debug(f"\n{content}")
        # Get the usage
        usage = response.usage
        # Create the usage
        usage = CompletionUsage(
            prompt_tokens=usage.prompt_tokens or -100,
            completion_tokens=usage.completion_tokens or -100,
            total_tokens=usage.total_tokens or -100
        )
        # Log the usage
        logger.warning(f"Usage: {usage}")
        
        # Extract tool calls from response
        tool_calls = []
        if response.choices[0].message.tool_calls is not None:
            # Traverse all the tool calls and log the tool call
            for i, tool_call in enumerate(response.choices[0].message.tool_calls):
                # Log the tool call
                logger.debug(f"Tool call {i + 1}: {tool_call}")
                
                # Create the tool call request
                tool_calls.append(ToolCallRequest(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    type="function", 
                    args=json.loads(tool_call.function.arguments)
                ))
        
        # Extract Finish reason
        if response.choices[0].finish_reason == "length":
            stop_reason = StopReason.LENGTH
        elif response.choices[0].finish_reason == "content_filter":
            stop_reason = StopReason.CONTENT_FILTER
        elif response.choices[0].finish_reason != "stop" or len(tool_calls) > 0:
            stop_reason = StopReason.TOOL_CALL
        elif response.choices[0].finish_reason == "stop" and len(tool_calls) == 0:
            stop_reason = StopReason.STOP
        else:
            stop_reason = StopReason.NONE
        
        # Return the response
        return AssistantMessage(
            role=MessageRole.ASSISTANT, 
            content=content, 
            tool_calls=tool_calls, 
            stop_reason=stop_reason, 
            usage=usage, 
        )

    def __prepare_kwargs(self, **kwargs) -> dict:
        """Prepare the kwargs for the completion.
        
        Args:
            **kwargs:
                The additional keyword arguments.
                
        Returns:
            dict:
                The prepared kwargs.
        """
        arguments = {
            "model": self.model, 
            "temperature": kwargs.get("temperature", self.temperature), 
            "parallel_tool_calls": kwargs.get("parallel_tool_calls", True), 
        }
            
        # Check the output token limit
        max_tokens = kwargs.get("max_tokens", None)
        if max_tokens is not None:
            arguments["max_tokens"] = max_tokens
            # Log the max tokens
            logger.info(f"Max tokens is set to {max_tokens}")
            
        # Check the format json
        if kwargs.get("format_json", False):
            arguments["response_format"] = {
                "type": "json_object",
            }
            # Log the response format
            logger.warning(f"Response format is set to {arguments['response_format']}")
            
            # Return the arguments, JSON format does not support tools and tool choice
            return arguments
        
        # Check tools are provided
        available_tools: list[dict[str, str]] = kwargs.get("available_tools", None)
        if available_tools is not None and len(available_tools) == 0:
            available_tools = None
        else:
            # Check the tool choice if the available tools are provided
            tool_choice = kwargs.get("tool_choice", None)
            if tool_choice is not None:
                if isinstance(tool_choice, str):
                    # Remove unexpectable tools
                    available_tools = [tool for tool in available_tools if tool["function"]["name"] == tool_choice]
                    if len(available_tools) == 0:
                        logger.critical(f"Tool choice {tool_choice} is not in the available tools.")
                        raise ValueError(f"Tool choice {tool_choice} is not in the available tools.")

                    # Get the tool from the available tools
                    tool_choice = available_tools[0]
                    # Set the tool choice
                    arguments["tool_choice"] = tool_choice
                    # Log the tool choice
                    logger.warning(f"Tool choice: {arguments['tool_choice']}")
                else:
                    logger.critical(f"Tool choice expected: str, the name of the tool, got: {type(tool_choice)}")
                    # Raise an error for the unsupported tool choice type
                    raise ValueError(f"Tool choice expected: str, the name of the tool, got: {type(tool_choice)}")
            
            arguments["tools"] = available_tools
        
        return arguments
