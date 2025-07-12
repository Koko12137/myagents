from typing import Optional

from pydantic import BaseModel, Field

from myagents.core.configs.agents import AgentConfig, CounterConfig
from myagents.core.envs import EnvironmentType


class EnvironmentConfig(BaseModel):
    """EnvironmentConfig is the configuration for the environment.
    
    Attributes:
        type (EnvironmentType):
            The type of the environment.
        agents (list[AgentConfig]):
            The configurations for the agents.
        step_counters (list[CounterConfig]):
            The configuration for the step counters.
    """
    type: EnvironmentType = Field(description="The type of the environment.")
    agents: list[AgentConfig] = Field(description="The configuration for the agents.")
    step_counters: list[CounterConfig] = Field(description="The configuration for the step counters.")
