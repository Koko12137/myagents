from asyncio import Lock
from typing import Optional

from fastmcp.client import Client as MCPClient

from myagents.core.agents.base import BaseAgent
from myagents.core.agents.types import AgentType
from myagents.core.interface import LLM, Workflow, Environment, StepCounter
from myagents.core.workflows import PlanAndExecFlow
from myagents.prompts.workflows.plan_and_exec import PROFILE as WORKFLOW_PROFILE


PROFILE = """
我叫{name}，是一个会按照“规划-执行-反思”的流程来执行任务的助手。

以下是我的工作流信息：
{workflow}
"""


class PlanAndExecAgent(BaseAgent):
    """PlanAndExecAgent is the agent that is used to plan and execute the environment.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        type (AgentType):
            The type of the agent.
        profile (str):
            The profile of the agent.
        llm (LLM):tee
            The LLM to use for the agent. 
        mcp_client (MCPClient):
            The MCP client to use for the agent.
        workflow (Workflow):
            The workflow to that the agent is running on.
        env (Environment):
            The environment to that the agent is running on.
        step_counters (dict[str, StepCounter]):
            The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
        lock (Lock):
            The synchronization lock of the agent. The agent can only work on one task at a time. 
            If the agent is running concurrently, the global context may not be working properly.
    """
    # Basic information
    uid: str
    name: str
    type: AgentType
    profile: str
    # LLM and MCP client
    llm: LLM
    mcp_client: MCPClient
    # Workflow and environment
    workflow: Workflow
    env: Environment
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    # Concurrency limit
    lock: Lock
    
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        *args, 
        **kwargs, 
    ) -> None: 
        """
        Initialize the PlanAndExecAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(
            llm=llm, 
            name=name, 
            type=AgentType.PLAN_AND_EXECUTE, 
            profile=PROFILE.format(name=name, workflow=WORKFLOW_PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            *args, 
            **kwargs,
        )
        
        # Read the workflow profile
        # Initialize the workflow for the agent
        self.workflow = PlanAndExecFlow()
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"PlanAndExecAgent(uid={self.uid}, name={self.name}, profile={self.profile})"
    
    def __repr__(self) -> str:
        return self.__str__()
