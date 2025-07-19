from typing import Any

from pydantic import BaseModel, Field, ConfigDict
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface.llm import Queue
from myagents.core.utils.tools import tool_schema, Provider


class BaseCompletionConfig(BaseModel):
    """CompletionConfig is the configuration for the completion.
    
    Attributes:
        tools (list[FastMcpTool]):
            The tools to use for the agent.
        tool_choice (str):
            The tool choice to use for the agent.
        exclude_tools (list[str]):
            The tools to exclude from the tool choice.
        
        top_p (float):
            The top p to use for the agent.
        max_tokens (int):
            The max tokens to use for the agent.
        frequency_penalty (float):
            The frequency penalty to use for the agent.
        temperature (float):
            The temperature to use for the agent.
        
        format_json (bool):
            Whether to format the response as JSON.
        
        allow_thinking (bool):
            Whether to allow the agent to think.
        
        stream (bool):
            Whether to stream the response.
        stream_interval (float):
            The interval to stream the response.
        
        stop_words (list[str]):
            The words to stop the response.
    """
    model_config = ConfigDict(extra="forbid")
    
    # Tool parameters
    tools: list[FastMcpTool] = Field(default=[])
    tool_choice: str = Field(default="auto")
    exclude_tools: list[str] = Field(default=[])
    
    # Generation parameters
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=8192)
    frequency_penalty: float = Field(default=0.0)
    temperature: float = Field(default=0.7)
    
    # Format parameters
    format_json: bool = Field(default=False)
    
    # Thinking parameters
    allow_thinking: bool = Field(default=True)
    
    # Streaming parameters
    stream: bool = Field(default=False)
    stream_interval: float = Field(default=1.0)
    stream_chunk_size: int = Field(default=1024)
    stream_queue: Queue = Field(default=None)
    
    # Stop parameters
    stop_words: list[str] = Field(default=[])
    
    def to_dict(self, provider: Provider) -> dict:
        """Convert the completion config to a dictionary.
        
        Returns:
            dict:
                The completion config as a dictionary.
        """
        match provider:
            case Provider.OPENAI:
                return self.to_openai()
            case Provider.ANTHROPIC:
                return self.to_anthropic()
            case _:
                raise ValueError(f"Unsupported provider: {provider}")
            
    def update(self, **kwargs: Any) -> None:
        """Update the completion config.
        
        Args:
            **kwargs:
                The keyword arguments to update the completion config.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_openai(self) -> dict:
        """Convert the completion config to the OpenAI format.
        
        Returns:
            dict:
                The completion config in the OpenAI format.
        """
        kwargs = {
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "temperature": self.temperature,
        }
        
        # Process format_json
        if self.format_json:
            kwargs["response_format"] = {
                "type": "json_object",
            }
        
        # Add thinking control, this is conflicted with format_json
        elif self.allow_thinking:
            kwargs["thinking"] = {
                "type": "object",
            }
        
        if not self.format_json:
            # Add tools, this is conflicted with format_json
            tools = [tool_schema(tool, Provider.OPENAI) for tool in self.tools if tool not in self.exclude_tools]
            if len(tools) > 0:
                kwargs["tools"] = tools
            
                # Add tool_choice
                if self.tool_choice is not None:
                    # Get tool_choice schema
                    tool_choice_schema = tool_schema(self.tools[self.tool_choice], Provider.OPENAI)
                    kwargs["tool_choice"] = tool_choice_schema
        
        return kwargs
