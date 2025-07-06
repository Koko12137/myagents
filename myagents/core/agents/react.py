from asyncio import Lock
from uuid import uuid4
from typing import Optional

from fastmcp.client import Client as MCPClient

from myagents.core.agents.base import BaseAgent
from myagents.core.interface import AgentType, LLM, Workflow, Environment, StepCounter
from myagents.core.workflows.react import ReActFlow
from myagents.prompts.workflows.react import PROFILE as WORKFLOW_PROFILE


NAME = "铁柱"


PROFILE = """
我叫铁柱，是一个会按照“观察-思考-行动-反思”的流程来执行任务的助手。

以下是我的工作流信息：
{workflow}
"""



class ReactAgent(BaseAgent):
    """ReactAgent is the agent that is used to react to the environment.
    
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
        llm: LLM, 
        workflow: Workflow,
        env: Environment,
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        *args, 
        **kwargs, 
    ) -> None:
        super().__init__(llm, step_counters, mcp_client, *args, **kwargs)
        
        # Initialize the basic information
        self.uid = str(uuid4())
        self.name = kwargs.get("name", NAME)
        self.type = AgentType.REACT
        self.profile = kwargs.get("profile", PROFILE.format(workflow=WORKFLOW_PROFILE))
        # Initialize the LLM and MCP client
        self.llm = llm
        self.mcp_client = mcp_client
        
        # Check if the workflow is a ReactWorkflow
        if not isinstance(workflow, ReActFlow):
            raise ValueError("The workflow must be a ReActFlow.")
        # Initialize the workflow for the agent
        self.workflow = workflow
        # Register the agent to the workflow
        self.workflow.register_agent(self)
        # Initialize the environment for the agent
        self.env = env
        # Register the agent to the environment
        self.env.register_agent(self)

    def __str__(self) -> str:
        return f"ReactAgent(uid={self.uid}, name={self.name}, profile={self.profile})"
    
    def __repr__(self) -> str:
        return self.__str__()
