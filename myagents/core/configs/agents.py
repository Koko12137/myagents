from typing import Union, Optional, Any

from pydantic import BaseModel, Field

from myagents.core.configs.llms import LLMConfig
from myagents.core.configs.mcps import MCPConfig


class CounterConfig(BaseModel):
    """步骤计数器的配置

    参数:
        name (str):
            步骤计数器的名称
        limit (Union[int, float]):
            步骤计数器的限制
    """
    name: str = Field(description="步骤计数器的名称")
    limit: Union[int, float] = Field(description="步骤计数器的限制")

    
class AgentConfig(BaseModel):
    """代理的配置

    参数:
        type (str):
            代理的类型
        llm (LLMConfig):
            语言模型的配置
        mcp_client (MCPConfig, 可选):
            MCP 客户端的配置
        extra_config (dict[str, Any]):
            代理的额外配置
    """
    type: str = Field(description="代理的类型")
    llm: LLMConfig = Field(description="语言模型的配置")
    mcp_client: Optional[MCPConfig] = Field(
        description="MCP 客户端的配置", 
        default=None, 
    )
    extra_config: dict[str, Any] = Field(
        description="代理的额外配置",
        default={},
    )
