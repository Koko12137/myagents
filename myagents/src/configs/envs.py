from pydantic import BaseModel, Field

from myagents.src.configs.agents import AgentConfig
from myagents.src.configs.workflows import WorkflowConfig


class EnvironmentConfig(BaseModel):
    name: str = Field(description="The name of the environment.")
    agent: AgentConfig = Field(description="The configuration for the agent.")
    workflows: list['WorkflowConfig'] = Field(
        description="The configurations for the workflows.", 
        default=[]
    )
 