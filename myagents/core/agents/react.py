from asyncio import Lock
from typing import Optional

from fastmcp.client import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import LLM, Workflow, Environment, StepCounter
from myagents.core.agents.base import BaseAgent
from myagents.core.agents.types import AgentType
from myagents.core.workflows import ReActFlow, ReActStage
from myagents.prompts.workflows.react import PROFILE, SYSTEM_PROMPT, THINK_PROMPT, REFLECT_PROMPT



AGENT_PROFILE = """
我叫 {name} ，是一个会按照“观察-思考-行动-反思”的流程来执行任务的助手。

以下是我的工作流信息：
{workflow}
"""


class ReActAgent(BaseAgent):
    """ReActAgent is the agent that is used to react to the environment.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        type (AgentType):
            The type of the agent.
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
        lock (Lock):
            The synchronization lock of the agent. The agent can only work on one task at a time. 
            If the agent is running concurrently, the global context may not be working properly.
        prompts (dict[ReActStage, str]):
            The prompts for running specific workflow of the workflow. The following prompts are supported:
            - "ReActStage.REASON_ACT": The reason and act prompt of the workflow.
            - "ReActStage.REFLECT": The reflect prompt of the workflow.
        observe_format (dict[ReActStage, str]):
            The format of the observation. The key is the workflow type and the value is the format content. 
    """
    # Basic information
    uid: str
    name: str
    type: AgentType
    profile: str
    # LLM and MCP client
    llm: LLM
    mcp_client: MCPClient
    # Tools
    tools: dict[str, FastMcpTool]
    # Workflow and environment
    workflow: Workflow
    env: Environment
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    # Concurrency limit
    lock: Lock
    # Prompts and observe format
    prompts: dict[ReActStage, str]
    observe_format: dict[ReActStage, str]
    
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        react_system_prompt: str = SYSTEM_PROMPT, 
        react_think_prompt: str = THINK_PROMPT, 
        react_reflect_prompt: str = REFLECT_PROMPT, 
        react_think_format: str = "document", 
        react_reflect_format: str = "todo", 
        *args, 
        **kwargs, 
    ) -> None:        
        """
        Initialize the ReActAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            react_system_prompt (str):
                The system prompt of the workflow.
            react_think_prompt (str):
                The think prompt of the workflow.
            react_reflect_prompt (str):
                The reflect prompt of the workflow.
            react_think_format (str):
                The observation format of the think stage.
            react_reflect_format (str):
                The observation format of the reflect stage.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        if react_system_prompt == SYSTEM_PROMPT:
            react_system_prompt = SYSTEM_PROMPT.format(profile=PROFILE)
        
        # Prepare the prompts
        self.react_system_prompt = react_system_prompt
        self.react_think_prompt = react_think_prompt
        self.react_reflect_prompt = react_reflect_prompt
        # Prepare the observe formats
        self.react_think_format = react_think_format
        self.react_reflect_format = react_reflect_format
        
        # Initialize the parent class
        super().__init__(
            llm=llm, 
            name=name, 
            type=AgentType.REACT, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                ReActStage.INIT: self.react_system_prompt, 
                ReActStage.REASON_ACT: self.react_think_prompt, 
                ReActStage.REFLECT: self.react_reflect_prompt, 
            }, 
            observe_format={
                ReActStage.REASON_ACT: self.react_think_format, 
                ReActStage.REFLECT: self.react_reflect_format, 
            }, 
            *args, 
            **kwargs,
        )
        
        # Initialize the workflow for the agent
        self.workflow = ReActFlow()
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"ReActAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
