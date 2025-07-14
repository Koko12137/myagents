from abc import abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable, Any, Optional, Union, AsyncGenerator

from myagents.core.messages import ToolCallResult, AssistantMessage, UserMessage, SystemMessage


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
    """
    provider: Provider
    model: str
    base_url: str
    
    @abstractmethod
    async def completion(
        self, 
        messages: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]], 
        available_tools: Optional[list[dict[str, str]]] = None, 
        tool_choice: Optional[str] = None, 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None, 
        top_p: Optional[float] = None, 
        stop: Optional[str] = None, 
        stream: Optional[bool] = None, 
        stream_options: Optional[dict] = None, 
        response_format: Optional[dict] = None, 
        **kwargs: dict, 
    ) -> AssistantMessage:
        """Completion the messages.
        
        Args:
            messages (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]) :
                The messages to complete. 
            available_tools (list[dict[str, str]], optional) :
                The available tools.
            tool_choice (str, optional) :
                The tool choice.
            max_tokens (int, optional) :
                The max tokens.
            temperature (float, optional) :
                The temperature.
            top_p (float, optional) :
                The top p.
            stop (str, optional) :
                The stop.
            stream (bool, optional) :
                The stream.
            stream_options (dict, optional) :
                The stream options.
            response_format (dict, optional) :
                The response format.
            **kwargs (dict) :
                The additional keyword arguments.

        Returns:
            AssistantMessage:
                The completed message named AssistantMessage from the LLM.
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
    """
    provider: Provider
    model: str
    base_url: str
    
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
