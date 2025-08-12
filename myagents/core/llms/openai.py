import os
import getpass
import json
from typing import Union

from loguru import logger
from openai import AsyncOpenAI

from myagents.core.messages import ToolCallRequest, ToolCallResult, AssistantMessage, StopReason, CompletionUsage, MessageRole, UserMessage, SystemMessage
from myagents.core.interface import LLM, Provider
from myagents.core.interface.llm import CompletionConfig
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
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Initialize the provider
        self.provider = Provider.OPENAI
        
        self.model = model
        self.base_url = base_url
        
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
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> AssistantMessage:
        """Completion the messages.

        Args:
            messages (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]): 
                The messages to complete.
            completion_config (CompletionConfig):
                The completion configuration.
            **kwargs:
                The additional keyword arguments.
                
        Raises:
            ValueError: 
                The value error raised by the unsupported message type.

        Returns:
            AssistantMessage: 
                The completed message.
        """
        kwargs = completion_config.to_dict(provider=self.provider)
        
        # Create the generation history
        history = to_openai_dict(messages)
        
        # Check streaming
        if completion_config.stream:
            # Get thequeue
            queue = completion_config.stream_queue
        
        # Call for the completion
        response = await self.client.chat.completions.create(
            model=self.model, 
            messages=history, 
            **kwargs, 
        )
        # Check streaming
        # if completion_config.stream:
        #     async for chunk in response:
        #         if chunk.choices[0].delta.content is not None:
        #             await queue.put(chunk.choices[0].delta.content)
            
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

    async def embed(
        self, 
        text: str, 
        dimensions: int = 1024, 
        **kwargs,
    ) -> list[float]:
        """Embedding the text.
        
        Args:
            text (str): 
                The text to embed. 
            dimensions (int, defaults to 1024):
                The dimensions of the embedding.
            **kwargs:
                The additional keyword arguments.
                
        Returns:
            list[float]: 
                The embedding of the text.
        """
        response = await self.client.embeddings.create(
            model=self.model, 
            input=text, 
        )
        return response.data[0].embedding[:dimensions]
