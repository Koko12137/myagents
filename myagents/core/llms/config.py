from typing import Any

from pydantic import BaseModel, Field, ConfigDict
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface.llm import Queue
from myagents.core.utils.tools import tool_schema, Provider


class BaseCompletionConfig(BaseModel):
    """CompletionConfig is the configuration for the completion.
    
    Attributes:
        tools (list[FastMcpTool], optional, defaults to []):
            The tools to use for the agent.
        tool_choice (str, optional, defaults to None):
            The tool choice to use for the agent.
        exclude_tools (list[str], optional, defaults to []):
            The tools to exclude from the tool choice.
        
        top_p (float, optional, defaults to 1.0):
            The top p to use for the agent.
        max_tokens (int, optional, defaults to 8192):
            The max tokens to use for the agent.
        frequency_penalty (float, optional, defaults to 1.0):
            The frequency penalty to use for the agent.
        temperature (float, optional, defaults to 0.9):
            The temperature to use for the agent.
        
        format_json (bool, optional, defaults to False):
            Whether to format the response as JSON.
        
        allow_thinking (bool, optional, defaults to True):
            Whether to allow the agent to think.
        
        stream (bool, optional, defaults to False):
            Whether to stream the response.
        stream_interval (float, optional, defaults to 1.0):
            The interval to stream the response.
        
        stop_words (list[str], optional, defaults to []):
            The words to stop the response.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Tool parameters
    tools: list[FastMcpTool] = Field(default=[])
    tool_choice: str = Field(default=None)
    exclude_tools: list[str] = Field(default=[])
    
    # Generation parameters
    top_p: float = Field(default=1.0)
    max_tokens: int = Field(default=8192)
    frequency_penalty: float = Field(default=1.0)
    temperature: float = Field(default=0.9)
    
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
            
            # Truncate all the other parameters processing
            return kwargs
        
        # Add thinking control
        if self.allow_thinking:
            kwargs["extra_body"] = {
                "enable_thinking": True,
            }
        else:
            kwargs["extra_body"] = {
                "enable_thinking": False,
            }
        
        # Add tools
        tools = [tool_schema(tool, Provider.OPENAI) for tool in self.tools if tool.name not in self.exclude_tools]
        if len(tools) > 0:
            kwargs["tools"] = tools
        
            # Add tool_choice
            if self.tool_choice is not None:
                tool_choice = [tool for tool in self.tools if tool.name == self.tool_choice]
                
                if len(tool_choice) > 0:
                    # Get tool_choice schema
                    tool_choice_schema = tool_schema(tool_choice[0], Provider.OPENAI)
                    kwargs["tool_choice"] = tool_choice_schema
        
        return kwargs
