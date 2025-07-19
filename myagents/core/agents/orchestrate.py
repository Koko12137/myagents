from asyncio import Lock
from typing import Optional, Union, Callable, Awaitable, Any

from fastmcp.client import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Stateful, LLM, Workflow, Environment, StepCounter
from myagents.core.agents.base import BaseAgent
from myagents.core.agents.types import AgentType
from myagents.core.workflows import OrchestrateFlow
from myagents.prompts.workflows.orchestrate import (
    PROFILE, 
    SYSTEM_PROMPT, 
    THINK_PROMPT, 
    ACTION_PROMPT, 
    REFLECT_PROMPT, 
    REACT_SYSTEM_PROMPT, 
    BLUEPRINT_FORMAT,
)


AGENT_PROFILE = """
我叫 {name} ，是一个会按照“编排-反思”的流程来编排任务总体目标的助手。

以下是我的工作流信息：
{workflow}
"""


class OrchestrateAgent(BaseAgent):
    """OrchestrateAgent is the agent that is used to orchestrate the environment.
    
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
        prompts (dict[str, str]):
            The prompts for running the workflow. 
        observe_format (dict[str, str]):
            The format of the observation the target. 
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
    prompts: dict[str, str]
    observe_format: dict[str, str]
    
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        plan_system_prompt: str = SYSTEM_PROMPT, 
        plan_reason_act_prompt: str = THINK_PROMPT, 
        plan_reflect_prompt: str = REACT_SYSTEM_PROMPT, 
        exec_system_prompt: str = SYSTEM_PROMPT, 
        exec_reason_act_prompt: str = ACTION_PROMPT, 
        exec_reflect_prompt: str = REFLECT_PROMPT, 
        agent_prompt: str = AGENT_PROMPT,
        plan_reason_act_format: str = "todo", 
        plan_reflect_format: str = "todo", 
        exec_reason_act_format: str = "todo", 
        exec_reflect_format: str = "todo", 
        agent_format: str = "todo", 
        **kwargs, 
    ) -> None:        
        """
        Initialize the OrchestrateAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            plan_system_prompt (str, optional):
                The system prompt of the reason stage.
            plan_reason_act_prompt (str, optional):
                The think prompt of the reason stage.
            plan_reflect_prompt (str, optional):
                The react system prompt of the react stage.
            exec_system_prompt (str, optional):
                The action prompt of the action stage.
            exec_reason_act_prompt (str, optional):
                The reflect prompt of the reflect stage.
            exec_reflect_prompt (str, optional):
                The observation format of the reason stage.
            plan_reason_act_format (str, optional):
                The observation format of the action stage.
            plan_reflect_format (str, optional):
                The observation format of the reflect stage.
            exec_reason_act_format (str, optional):
                The observation format of the reason stage.
            exec_reflect_format (str, optional):
                The observation format of the reflect stage.
            agent_format (str, optional):
                The observation format of the agent.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(
            llm=llm, 
            name=name, 
            agent_type=AgentType.ORCHESTRATE, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                "plan_system_prompt": plan_system_prompt, 
                "plan_reason_act_prompt": plan_reason_act_prompt, 
                "plan_reflect_prompt": plan_reflect_prompt, 
                "exec_system_prompt": exec_system_prompt, 
                "exec_reason_act_prompt": exec_reason_act_prompt, 
                "exec_reflect_prompt": exec_reflect_prompt, 
                "agent_prompt": agent_prompt, 
            }, 
            observe_format={
                "plan_reason_act_format": plan_reason_act_format, 
                "plan_reflect_format": plan_reflect_format, 
                "exec_reason_act_format": exec_reason_act_format, 
                "exec_reflect_format": exec_reflect_format, 
                "agent_format": agent_format, 
            }, 
            **kwargs,
        )
        
        # Read the workflow profile
        # Initialize the workflow for the agent
        self.workflow = OrchestrateFlow(
            prompts=self.prompts, 
            observe_format=self.observe_format, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"OrchestrateAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    async def observe(
        self, 
        target: Union[Stateful, Any], 
        observe_func: Optional[Callable[..., Awaitable[Union[str, list[dict]]]]] = None, 
        **kwargs, 
    ) -> Union[str, list[dict]]:
        """Observe the target. If the target is not a task or environment, you should provide the observe 
        function to get the string or list of dicts observation. 
        
        Args:
            target (Union[Stateful, Any]): 
                The stateful entity or any other entity to observe. 
            format (str):
                The format of the observation. 
            observe_func (Callable[..., Awaitable[Union[str, list[dict]]]], optional):
                The function to observe the target. If not provided, the default observe function will 
                be used. The function should have the following signature:
                - target (Union[Stateful, Any]): The stateful entity or any other entity to observe.
                - **kwargs: The additional keyword arguments for observing the target.
                The function should return the observation in the following format:
                - str: The string observation. 
                - list[dict]: The list of dicts observation. If the observation is multi-modal.
            **kwargs:
                The additional keyword arguments for observing the target. 
            
        Returns:
            Union[str, list[dict]]:
                The up to date information observed from the stateful entity or any other entity.  
        """
        raw_observe = await super().observe(
            target, 
            prompt=self.workflow.prompts[self.workflow.stage], 
            observe_func=observe_func, 
            **kwargs,
        )
        
        # Check the stage of the workflow
        if self.workflow.stage == "plan_reason_act":
            # Get the blueprint
            blueprint = self.env.context.get("blueprint")
            # Format the blueprint
            blueprint_format = BLUEPRINT_FORMAT.format(blueprint=blueprint)
            # Concatenate the blueprint and the raw observation
            return f"{blueprint_format}\n\n{raw_observe}"
        
        return raw_observe