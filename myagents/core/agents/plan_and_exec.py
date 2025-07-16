from asyncio import Lock
from typing import Optional, Union, Callable, Awaitable, Any

from fastmcp.client import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.agents.base import BaseAgent
from myagents.core.agents.types import AgentType
from myagents.core.interface import LLM, Workflow, Environment, StepCounter, Stateful
from myagents.core.tasks.task import DocumentTaskView
from myagents.core.workflows import PlanAndExecFlow, PlanAndExecStage
from myagents.prompts.workflows.plan_and_exec import (
    PROFILE, 
    PLAN_SYSTEM_PROMPT, 
    PLAN_THINK_PROMPT, 
    PLAN_REFLECT_PROMPT, 
    EXEC_SYSTEM_PROMPT, 
    EXEC_THINK_PROMPT, 
    ERROR_PROMPT, 
    PLAN_LAYER_LIMIT, 
    BLUEPRINT_FORMAT,
    TASK_RESULT_FORMAT,
)
from myagents.prompts.workflows.react import REFLECT_PROMPT


AGENT_PROFILE = """
我叫 {name} ，是一个会按照“规划-执行-反思”的流程来执行任务的助手。

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
        prompts (dict[str, str]):
            The prompts for running specific workflow of the workflow. 
            The following prompts are supported: 
            - "plan_system": The system prompt of the plan stage.
            - "plan_think": The think prompt of the plan stage.
            - "plan_reflect": The reflect prompt of the plan stage.
            - "exec_system": The system prompt of the exec stage.
            - "exec_think": The think prompt of the exec stage.
            - "exec_reflect": The reflect prompt of the exec stage.
            - "error": The error prompt of the workflow.
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
    # Prompts
    prompts: dict[str, str]
    # Observe format
    observe_format: dict[str, str]
    
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        plan_system_prompt: str = PLAN_SYSTEM_PROMPT, 
        plan_think_prompt: str = PLAN_THINK_PROMPT, 
        plan_reflect_prompt: str = PLAN_REFLECT_PROMPT, 
        exec_system_prompt: str = EXEC_SYSTEM_PROMPT, 
        exec_think_prompt: str = EXEC_THINK_PROMPT, 
        exec_reflect_prompt: str = REFLECT_PROMPT, 
        plan_layer_limit: str = PLAN_LAYER_LIMIT, 
        error_prompt: str = ERROR_PROMPT, 
        plan_think_format: str = "todo", 
        plan_reflect_format: str = "todo", 
        exec_think_format: str = "todo", 
        exec_reflect_format: str = "document", 
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
            plan_system_prompt (str, optional):
                The system prompt of the plan stage.
            plan_think_prompt (str, optional):
                The think prompt of the plan stage.
            plan_reflect_prompt (str, optional):
                The reflect prompt of the plan stage. 
            exec_system_prompt (str, optional):
                The system prompt of the exec stage.
            exec_think_prompt (str, optional):
                The think prompt of the exec stage.
            exec_reflect_prompt (str, optional):
                The reflect prompt of the exec stage.
            plan_layer_limit (str, optional):
                The sub task layer limit prompt of the plan stage.
            error_prompt (str, optional):
                The error prompt of the workflow.
            plan_think_format (str, optional):
                The observation format of the plan think stage.
            plan_reflect_format (str, optional):
                The observation format of the plan reflect stage.
            exec_think_format (str, optional):
                The observation format of the exec think stage.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Prepare the prompts
        self.plan_system_prompt = plan_system_prompt
        self.plan_think_prompt = plan_think_prompt
        self.plan_reflect_prompt = plan_reflect_prompt
        self.exec_system_prompt = exec_system_prompt
        self.exec_think_prompt = exec_think_prompt
        self.exec_reflect_prompt = exec_reflect_prompt
        self.error_prompt = error_prompt
        # Prepare the observe formats
        self.plan_think_format = plan_think_format
        self.plan_reflect_format = plan_reflect_format
        self.exec_think_format = exec_think_format
        self.exec_reflect_format = exec_reflect_format
        
        # Additional prompts
        self.plan_layer_limit = plan_layer_limit if plan_layer_limit != "" else PLAN_LAYER_LIMIT
        
        super().__init__(
            llm=llm, 
            name=name, 
            type=AgentType.PLAN_AND_EXECUTE, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                PlanAndExecStage.PLAN_INIT: self.plan_system_prompt, 
                PlanAndExecStage.PLAN_REASON_ACT: self.plan_think_prompt, 
                PlanAndExecStage.PLAN_REFLECT: self.plan_reflect_prompt, 
                PlanAndExecStage.EXEC_INIT: self.exec_system_prompt, 
                PlanAndExecStage.EXEC_REASON_ACT: self.exec_think_prompt, 
                PlanAndExecStage.EXEC_REFLECT: self.exec_reflect_prompt, 
                PlanAndExecStage.ERROR: self.error_prompt, 
            }, 
            observe_format={
                PlanAndExecStage.PLAN_REASON_ACT: self.plan_think_format, 
                PlanAndExecStage.PLAN_REFLECT: self.plan_reflect_format, 
                PlanAndExecStage.EXEC_REASON_ACT: self.exec_think_format, 
                PlanAndExecStage.EXEC_REFLECT: self.exec_reflect_format, 
            }, 
            *args, 
            **kwargs,
        )
        
        # Read the workflow profile
        # Initialize the workflow for the agent
        self.workflow = PlanAndExecFlow()
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"PlanAndExecAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()

    async def observe(
        self, 
        target: Union[Stateful, Any], 
        observe_func: Optional[Callable[..., Awaitable[Union[str, list[dict]]]]] = None, 
        **kwargs, 
    ) -> Union[str, list[dict]]:
        """Observe the target. If the target is not a task or environment, you should provide the observe 
        function to get the string or list of dicts observation. This observe method will apply the 
        plan layer limit to the observation if the workflow stage is the plan reason act stage.
        
        Args:
            target (Union[Stateful, Any]): 
                The stateful entity or any other entity to observe. 
            observe_func (Optional[Callable[..., Awaitable[Union[str, list[dict]]]]]):
                The function to observe the target. If not provided, the agent will use the default observe function.
            **kwargs:
                The keyword arguments to be passed to the observe function.
        """
        # Observe the target using the super class
        raw_result = await super().observe(target, observe_func, **kwargs)
        
        # Check the stage of the workflow
        if self.workflow.stage == PlanAndExecStage.PLAN_REASON_ACT or self.workflow.stage == PlanAndExecStage.PLAN_REFLECT:
            # Apply the plan layer limit to the observation
            observe = self.plan_layer_limit.format(
                plan_layer_limit=target.sub_task_depth, 
            )
            observe = f"{raw_result}\n\n## 其他限制\n\n{observe}"
            
            # Prepare the blueprint
            blueprint = self.env.context.get("blueprint")
            if blueprint != "" and blueprint is not None:
                blueprint_format = BLUEPRINT_FORMAT.format(blueprint=blueprint)
                # Append the blueprint to the observe
                observe = f"{observe}\n\n{blueprint_format}"
                
        elif self.workflow.stage == PlanAndExecStage.EXEC_REASON_ACT:
            # Prepare the current result
            task = self.env.context.get("task")
            task_result = DocumentTaskView(task).format()
            if task_result != "":
                task_result_format = TASK_RESULT_FORMAT.format(task_result=task_result)
                observe = f"{raw_result}\n\n{task_result_format}"
        else:
            observe = raw_result
        
        return observe
