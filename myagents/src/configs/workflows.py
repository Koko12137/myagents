from pydantic import BaseModel, Field

from myagents.src.configs.agents import AgentConfig, CounterConfig


class WorkflowConfig(BaseModel):
    """The configuration for the workflow.
    
    Args:
        name (str):
            The name of the workflow.
        agent (AgentConfig):
            The configuration for the agent.
        step_counters (list[CounterConfig]):
            The step counters for the workflow.
        workflows (list['WorkflowConfig']):
            The configurations for the workflows.
    """
    name: str = Field(description="The name of the workflow.")
    agent: AgentConfig = Field(description="The configuration for the agent.")
    step_counters: list[CounterConfig] = Field(
        description="The step counters for the workflow.", 
        default=[]
    )
    workflows: list['WorkflowConfig'] = Field(
        description="The configurations for the workflows.", 
        default=[]
    )
