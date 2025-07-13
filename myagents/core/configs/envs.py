from typing import Optional

from pydantic import BaseModel, Field

from myagents.core.configs.agents import AgentConfig, CounterConfig


class EnvironmentConfig(BaseModel):
    """EnvironmentConfig is the configuration for the environment.
    
    Attributes:
        type (str):
            The type of the environment.
        agents (list[AgentConfig]):
            The configurations for the agents.
        step_counters (list[CounterConfig]):
            The configuration for the step counters.
    """
    type: str = Field(description="The type of the environment.")
    agents: list[AgentConfig] = Field(description="The configuration for the agents.")
    step_counters: list[CounterConfig] = Field(description="The configuration for the step counters.")
