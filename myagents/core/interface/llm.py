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
class CompletionConfig(Protocol):
    """CompletionConfig is a protocol for the LLM config.
    
    Attributes:
        provider (Provider):
            The provider of the LLM.
    """

    @abstractmethod
    def to_dict(self, provider: Provider) -> dict:
        """Convert the completion config to a dictionary.
        
        Args:
            provider (Provider):
                The provider of the LLM.
                
        Returns:
            dict:
                The completion config as a dictionary.
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update the completion config.
        
        Args:
            **kwargs:
                The additional keyword arguments to update the completion config.
        """
        pass


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
        completion_config: CompletionConfig, 
    ) -> AssistantMessage:
        """Completion the messages.
        
        Args:
            messages (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]) :
                The messages to complete. 
            completion_config (CompletionConfig) :
                The completion config. 

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
