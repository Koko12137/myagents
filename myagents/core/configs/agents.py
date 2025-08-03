from typing import Union, Optional, Any

from pydantic import BaseModel, Field

from myagents.core.configs.llms import LLMConfig
from myagents.core.configs.mcps import MCPConfig
from myagents.core.configs.memories import VectorCollectionConfig


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
            Agent 的类型
        llm (LLMConfig):
            语言模型的配置
        mcp_client (MCPConfig, 可选):
            MCP 客户端的配置
        use_memory (bool, 可选):
            是否使用记忆
        embedding_llm (LLMConfig, 可选):
            嵌入语言模型的配置
        memory_config (VectorCollectionConfig, 可选):
            向量记忆集合的配置
        extra_config (dict[str, Any], 可选):
            Agent 的额外配置
    """
    type: str = Field(description="Agent 的类型")
    llm: LLMConfig = Field(description="语言模型的配置")
    mcp_client: Optional[MCPConfig] = Field(
        description="MCP 客户端的配置", 
        default=None, 
    )
    use_memory: Optional[bool] = Field(
        description="是否使用记忆", 
        default=False, 
    )
    embedding_llm: Optional[LLMConfig] = Field(
        description="嵌入语言模型的配置", 
        default=None, 
    )
    memory_config: Optional[VectorCollectionConfig] = Field(
        description="向量记忆集合的配置", 
        default=None, 
    )   
    extra_config: dict[str, Any] = Field(
        description="Agent 的额外配置",
        default={},
    )
