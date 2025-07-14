from typing import Union

from fastmcp import FastMCP, Client as MCPClient

from myagents.core.interface.core import Agent, Workflow, Environment, StepCounter, Task
from myagents.core.interface.llm import LLM
from myagents.core.interface.message import Message


class RemoteAgentServer(Agent):
    """RemoteAgentServer is a server that can be used to register the agent to the remote server.
    
    Attributes:
        mcp_server (FastMCP):
            The MCP server to run the agent.
    
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        profile (str):
            The profile of the agent.
        llm (LLM):
            The LLM to use for the agent.
        mcp_client (MCPClient):
            The MCP client to use for the agent.
        workflow (Workflow):
            The workflow to that the agent is running on.
        env (Environment):
            The environment to that the agent is running on.
        step_counters (dict[str, StepCounter]):
            The step counters to use for the agent. Any of one reach the limit, the agent will be stopped.
    """
    # Basic information
    mcp_server: FastMCP
    
    # Basic information
    uid: str
    name: str
    profile: str
    llm: LLM
    mcp_client: MCPClient
    # Workflow and environment
    workflow: Workflow
    env: Environment
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    
    def __init__(
        self, 
        name: str, 
        profile: str, 
        llm: LLM,   
        workflow: Workflow, 
        env: Environment, 
        *args, 
        **kwargs,
    ) -> None:
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        # Set the mcp server
        self.mcp_server = FastMCP()
        # Set the basic information
        self.name = name
        self.profile = profile
        self.llm = llm
        # Set the workflow and environment
        self.workflow = workflow
        self.env = env
        # Set the step counters
        self.step_counters = {}
    
    def register_agent(self, agent: Agent):
        self.mcp_server.register_tool(agent.tools)
        
    async def run(self, target: Union[Task, Environment], **kwargs) -> Message:
        pass
