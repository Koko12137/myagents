from typing import Optional

from pydantic import BaseModel, Field

from myagents.core.configs.agents import AgentConfig, CounterConfig


class EnvironmentConfig(BaseModel):
    """环境的配置类
    
    属性:
        type (str):
            环境的类型
        agents (list[AgentConfig]):
            代理的配置列表
        step_counters (list[CounterConfig]):
            步骤计数器的配置
    """
    type: str = Field(description="环境的类型")
    agents: list[AgentConfig] = Field(description="代理的配置列表")
    step_counters: list[CounterConfig] = Field(description="步骤计数器的配置")
