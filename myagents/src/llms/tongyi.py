import os
import getpass
import json

import dashscope
from loguru import logger

from myagents.src.interface import LLM, Logger
from myagents.src.message import CompletionMessage, MessageRole, StopReason, ToolCallRequest, ToolCallResult
from myagents.src.utils.tools import Provider


class TongyiLLM(LLM):
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
        """Initialize the TongyiLLM.
        
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
        # Assert the provider is Tongyi
        assert self.provider == Provider.TONGYI, "Tongyi is the only supported provider for TongyiLLM."
        
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.custom_logger = custom_logger
        self.debug = debug
        if self.debug:
            self.custom_logger.enable("myagents.llms.base")
        
        # Initialize the client
        api_key_field: str = kwargs.get("api_key_field", "TONGYI_KEY")
        self.api_key = os.getenv(api_key_field) or getpass.getpass("Enter your Tongyi API key: ")
    
    async def completion(
        self, 
        messages: list[CompletionMessage], 
        available_tools: list[dict[str, str]] | None = None
    ) -> CompletionMessage:
        raise NotImplementedError("Tongyi does not support completion.")  
