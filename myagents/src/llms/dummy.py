import os
import getpass
import json

from loguru import logger
from openai import AsyncOpenAI

from myagents.src.interface import LLM, Logger
from myagents.src.message import CompletionMessage, MessageRole, StopReason, ToolCallRequest, ToolCallResult
from myagents.src.utils.tools import Provider


class DummyLLM(LLM):
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
        """Initialize the DummyLLM.
        
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
        self.provider = Provider.DUMMY
        
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.custom_logger = custom_logger
        self.debug = debug
        if self.debug:
            self.custom_logger.enable("myagents.llms.dummy")
    
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
        stop_reason = StopReason.NONE
        content = "Dummy agent is thinking..."
        tool_calls = []
        
        # Return the response
        return CompletionMessage(
            role=MessageRole.ASSISTANT, 
            content=content, 
            tool_calls=tool_calls, 
            stop_reason=stop_reason, 
        )
