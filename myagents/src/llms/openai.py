import os
import getpass
import json

from loguru import logger
from openai import AsyncOpenAI

from myagents.src.interface import LLM, Logger
from myagents.src.message import CompletionMessage, MessageRole, StopReason, ToolCallRequest, ToolCallResult
from myagents.src.utils.tools import Provider


class OpenAiLLM(LLM):
    provider: Provider
    model: str
    base_url: str
    temperature: float
    custom_logger: Logger
    debug: bool
    
    def __init__(
        self, 
        provider: str, 
        model: str, 
        base_url: str, 
        temperature: float = 1.0, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
        **kwargs
    ) -> None:
        """Initialize the OpenAiLLM.
        
        Args:
            provider (str): 
                The provider of the LLM.
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
        self.provider = Provider(provider)
        # Assert the provider is OpenAI
        assert self.provider == Provider.OPENAI, "OpenAI is the only supported provider for OpenAiLLM."
        
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.custom_logger = custom_logger
        self.debug = debug
        if self.debug:
            self.custom_logger.enable("myagents.llms.base")
        
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
        messages: list[CompletionMessage], 
        available_tools: list[dict[str, str]] | None = None, 
    ) -> CompletionMessage:
        """Completion the messages.

        Args:
            messages (list[CompletionMessage]): 
                The messages to complete.
            available_tools (list[dict[str, str]] | None, optional): 
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
                            "tool_call_id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name, 
                                "arguments": json.dumps(tool_call.args, ensure_ascii=False)
                            }
                        })
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
                
            history.append(message_dict)
            
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
        
        # Extract tool calls from response
        tool_calls = []
        if response.choices[0].message.tool_calls is not None:
            for tool_call in response.choices[0].message.tool_calls:
                tool_calls.append(ToolCallRequest(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=json.loads(tool_call.function.arguments)
                ))
                # Log the tool call
                self.custom_logger.info(f"Tool call: {tool_call}")
        
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
        )
