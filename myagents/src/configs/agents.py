from typing import Union, Optional

from pydantic import BaseModel, Field

from myagents.src.configs.llms import LLMConfig
from myagents.src.configs.mcps import MCPConfig


class CounterConfig(BaseModel):
    """The configuration for the step counter.

    Args:
        name (str):
            The name of the step counter.
        limit (Union[int, float]):
            The limit of the step counter.
    """
    name: str = Field(description="The name of the step counter.")
    limit: Union[int, float] = Field(description="The limit of the step counter.")

    
class AgentConfig(BaseModel):
    """The configuration for the agent.

    Args:
        name (str):
            The name of the agent.
        llm (LLMConfig):
            The configuration for the LLM.
        mcp_client (MCPConfig, optional):
            The configuration for the MCP client. 
    """
    name: str = Field(description="The name of the agent.")
    llm: LLMConfig = Field(description="The configuration for the LLM.")
    mcp_client: Optional[MCPConfig] = Field(
        description="The configuration for the MCP client.", 
        default=None, 
    )
