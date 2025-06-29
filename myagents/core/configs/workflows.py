from typing import Optional

from pydantic import BaseModel, Field

from myagents.core.configs.agents import AgentConfig, CounterConfig


class WorkflowConfig(BaseModel):
    """WorkflowConfig is the configuration for the workflow.
    
    Args:
        name (str):
            The name of the workflow.
        agent (AgentConfig):
            The configuration for the agent.
        step_counters (list[CounterConfig], optional):
            The step counters for the workflow.
        workflows (list['WorkflowConfig'], optional):
            The configurations for the workflows.
    """
    name: str = Field(description="The name of the workflow.")
    agent: AgentConfig = Field(description="The configuration for the agent.")
    step_counters: Optional[list[CounterConfig]] = Field(
        description="The step counters for the workflow.", 
        default=[]
    )
    workflows: Optional[list['WorkflowConfig']] = Field(
        description="The configurations for the workflows.", 
        default=[]
    )
