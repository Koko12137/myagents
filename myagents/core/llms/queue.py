import json
import asyncio
from typing import Union, Optional

from loguru import logger

from myagents.core.interface import LLM, Logger
from myagents.core.message import CompletionMessage, ToolCallRequest, ToolCallResult
from myagents.core.utils.tools import Provider


class QueueLLM(LLM):
    """QueueLLM is a LLM that uses a queue to communicate with the LLM.

    Attributes:
        provider (Provider):
            The provider of the LLM.
        model (str):
            The model of the LLM.
        base_url (str):
            The base URL of the LLM.
        temperature (float):
            The temperature of the LLM.
        custom_logger (Logger):
            The custom logger.
        debug (bool):
            The debug flag.
        request_queue (asyncio.Queue):
            The queue for sending the request to the LLM.
        response_queue (asyncio.Queue):
            The queue for receiving the response from the LLM.
    """
    provider: Provider
    model: str
    base_url: str
    temperature: float
    custom_logger: Logger
    debug: bool
    
    # Additional attributes
    request_queue: asyncio.Queue
    response_queue: asyncio.Queue
    
    def __init__(
        self, 
        model: str, 
        base_url: str, 
        temperature: float = 1.0, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
        **kwargs
    ) -> None:
        """Initialize the QueueLLM.
        
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
        self.provider = Provider.QUEUE
        
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.custom_logger = custom_logger
        self.debug = debug
        
        # Create a queue for transferring the messages
        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        
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
            available_tools (list[dict[str, str]], optional): 
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
            
        # Put the history to the request queue
        await self.request_queue.put(history)
        # Get the response from the response queue
        response: CompletionMessage = await self.response_queue.get()
        # Log the response
        self.custom_logger.info(f"\n{response.content}")
        
        # Return the response
        return response
