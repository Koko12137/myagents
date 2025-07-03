from abc import abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable, Any, Optional, Union, AsyncGenerator

from myagents.core.message import CompletionMessage, ToolCallRequest, ToolCallResult
from myagents.core.interface.logger import Logger


@runtime_checkable
class Queue(Protocol):
    """Queue is a protocol for the queue.
    
    Attributes:
        queue (Queue):
            The queue.
    """
    @abstractmethod
    async def put(self, *args, **kwargs) -> None:
        """Put an item into the queue.
        
        Args:
            *args:
                The additional arguments to pass to the put method.
            **kwargs:
                The additional keyword arguments to pass to the put method.
        """
        pass
    
    @abstractmethod
    async def get(self, *args, **kwargs) -> Any:
        """Get an item from the queue.
        
        Args:
            *args:
                The additional arguments to pass to the get method.
            **kwargs:
                The additional keyword arguments to pass to the get method.
        """
        pass


class Provider(Enum):
    """The provider of the LLM.
    
    - OPENAI (str): The OpenAI provider.
    """
    OPENAI = "openai"


@runtime_checkable
class LLM(Protocol):
    """LLM is a protocol for Language Model.
    
    Attributes:
        provider (Provider) :
            The provider of the LLM.
        model (str) :
            The model of the LLM. 
        base_url (str) :
            The base URL of the LLM. 
        custom_logger (Logger):
            The custom logger. If not provided, the default loguru logger will be used. 
        debug (bool):
            The debug flag. 
    """
    provider: Provider
    model: str
    base_url: str
    
    @abstractmethod
    async def completion(
        self, 
        messages: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]], 
        available_tools: Optional[list[dict[str, str]]] = None, 
        **kwargs: dict, 
    ) -> CompletionMessage:
        """Completion the messages.
        
        Args:
            messages (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]) :
                The messages to complete. 
            available_tools (list[dict[str, str]], optional) :
                The available tools.
            **kwargs (dict) :
                The additional keyword arguments.

        Returns:
            CompletionMessage:
                The completed message.
        """
        pass

    
class StreamLLM(LLM):
    """StreamLLM is a LLM that is used to stream the LLM calls.
    
    Attributes:
        provider (Provider) :
            The provider of the LLM.
        model (str) :
            The model of the LLM. 
        base_url (str) :
            The base URL of the LLM. 
        custom_logger (Logger):
            The custom logger. If not provided, the default loguru logger will be used. 
        debug (bool):
            The debug flag. 
    """
    provider: Provider
    model: str
    base_url: str
    
    # Stream queue
    queue: Queue
    
    @abstractmethod
    async def stream(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        """Stream the LLM calls.
        
        Args:
            *args:
                The additional arguments to pass to the stream method.
            **kwargs:
                The additional keyword arguments to pass to the stream method.
        """
        pass
