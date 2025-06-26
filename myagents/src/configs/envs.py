from typing import Optional

from pydantic import BaseModel, Field

from myagents.src.configs.agents import AgentConfig
from myagents.src.configs.workflows import WorkflowConfig


class EnvironmentConfig(BaseModel):
    """EnvironmentConfig is the configuration for the environment.
    
    Attributes:
        name (str):
            The name of the environment.
        agent (AgentConfig):
            The configuration for the agent.
        workflows (list['WorkflowConfig'], optional):
            The configurations for the workflows.
    """
    name: str = Field(description="The name of the environment.")
    agent: AgentConfig = Field(description="The configuration for the agent.")
    workflows: Optional[list['WorkflowConfig']] = Field(
        description="The configurations for the workflows.", 
        default=[]
    )
 