from asyncio import Lock
from typing import Optional

from fastmcp.client import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import LLM, Workflow, Environment, StepCounter, VectorMemoryDB, EmbeddingLLM
from myagents.core.agents.base import BaseAgent
from myagents.core.agents.memory import BaseMemoryAgent
from myagents.core.agents.types import AgentType
from myagents.core.workflows import PlanAndExecFlow
from myagents.prompts.workflows.plan_and_exec import (
    PROFILE, 
    EXEC_SYSTEM_PROMPT, 
    EXEC_THINK_PROMPT, 
    ERROR_PROMPT, 
)
from myagents.prompts.workflows.orchestrate import (
    PLAN_SYSTEM_PROMPT as ORCH_PLAN_SYSTEM_PROMPT, 
    PLAN_THINK_PROMPT as ORCH_PLAN_THINK_PROMPT, 
    PLAN_REFLECT_PROMPT as ORCH_PLAN_REFLECT_PROMPT, 
    EXEC_SYSTEM_PROMPT as ORCH_EXEC_SYSTEM_PROMPT, 
    EXEC_THINK_PROMPT as ORCH_EXEC_THINK_PROMPT, 
    EXEC_REFLECT_PROMPT as ORCH_EXEC_REFLECT_PROMPT, 
)
from myagents.prompts.workflows.react import REFLECT_PROMPT
from myagents.prompts.memories import (
    SEMANTIC_MEMORY_EXTRACT_PROMPT, 
    EPISODE_MEMORY_EXTRACT_PROMPT, 
    PROCEDURAL_MEMORY_EXTRACT_PROMPT, 
    SEMANTIC_FORMAT, 
    EPISODE_FORMAT, 
    PROCEDURAL_FORMAT, 
)


AGENT_PROFILE = """
我叫 {name} ，是一个会按照“规划-执行-反思”的流程来执行任务的助手。

以下是我的工作流信息：
{workflow}
"""


class PlanAndExecAgent(BaseAgent):
    """PlanAndExecAgent is the agent that is used to plan and execute the environment.
    
    Attributes:
        uid (int):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        agent_type (AgentType):
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
        observe_formats (dict[str, str]):
            The format of the observation the target. 
    """
    # Basic information
    uid: int
    name: str
    agent_type: AgentType
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
    observe_formats: dict[str, str]
    
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        orch_plan_system_prompt: str = ORCH_PLAN_SYSTEM_PROMPT, 
        orch_plan_think_prompt: str = ORCH_PLAN_THINK_PROMPT, 
        orch_plan_reflect_prompt: str = ORCH_PLAN_REFLECT_PROMPT, 
        orch_exec_system_prompt: str = ORCH_EXEC_SYSTEM_PROMPT, 
        orch_exec_think_prompt: str = ORCH_EXEC_THINK_PROMPT, 
        orch_exec_reflect_prompt: str = ORCH_EXEC_REFLECT_PROMPT, 
        exec_system_prompt: str = EXEC_SYSTEM_PROMPT, 
        exec_think_prompt: str = EXEC_THINK_PROMPT, 
        exec_reflect_prompt: str = REFLECT_PROMPT, 
        error_prompt: str = ERROR_PROMPT, 
        orch_plan_think_format: str = "todo", 
        orch_plan_reflect_format: str = "todo", 
        orch_exec_think_format: str = "todo", 
        orch_exec_reflect_format: str = "json", 
        exec_think_format: str = "todo", 
        exec_reflect_format: str = "document", 
        agent_format: str = "todo", 
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
            error_prompt (str, optional):
                The error prompt of the workflow.
            plan_think_format (str, optional):
                The observation format of the plan think stage.
            plan_reflect_format (str, optional):
                The observation format of the plan reflect stage.
            exec_think_format (str, optional):
                The observation format of the exec think stage.
            exec_reflect_format (str, optional):
                The observation format of the exec reflect stage.
            agent_format (str, optional):
                The observation format of the agent.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(
            llm=llm, 
            name=name, 
            agent_type=AgentType.PLAN_AND_EXECUTE, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                "orch_plan_system_prompt": orch_plan_system_prompt, 
                "orch_plan_reason_act_prompt": orch_plan_think_prompt, 
                "orch_plan_reflect_prompt": orch_plan_reflect_prompt, 
                "orch_exec_system_prompt": orch_exec_system_prompt, 
                "orch_exec_reason_act_prompt": orch_exec_think_prompt, 
                "orch_exec_reflect_prompt": orch_exec_reflect_prompt, 
                "exec_system_prompt": exec_system_prompt, 
                "exec_reason_act_prompt": exec_think_prompt, 
                "exec_reflect_prompt": exec_reflect_prompt, 
                "error_prompt": error_prompt, 
            }, 
            observe_formats={
                "orch_plan_reason_act_format": orch_plan_think_format, 
                "orch_plan_reflect_format": orch_plan_reflect_format, 
                "orch_exec_reason_act_format": orch_exec_think_format, 
                "orch_exec_reflect_format": orch_exec_reflect_format, 
                "exec_reason_act_format": exec_think_format, 
                "exec_reflect_format": exec_reflect_format, 
                "agent_format": agent_format, 
            }, 
            **kwargs,
        )
        
        # Read the workflow profile
        # Initialize the workflow for the agent
        self.workflow = PlanAndExecFlow(
            prompts=self.prompts, 
            observe_formats=self.observe_formats, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"PlanAndExecAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class MemoryPlanAndExecAgent(PlanAndExecAgent, BaseMemoryAgent):
    """MemoryPlanAndExecAgent is the agent that is used to plan and execute the environment with memory.
    
    Attributes:
        uid (int):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        agent_type (AgentType):
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
        observe_formats (dict[str, str]):
            The format of the observation the target. 
    """
    # Basic information
    uid: int
    name: str
    agent_type: AgentType
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
    observe_formats: dict[str, str]
    
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        embedding_llm: EmbeddingLLM, 
        step_counters: list[StepCounter], 
        vector_memory: VectorMemoryDB, 
        # trajectory_memory: TableMemoryDB, # TODO: 暂时不使用轨迹记忆
        mcp_client: Optional[MCPClient] = None, 
        orch_plan_system_prompt: str = ORCH_PLAN_SYSTEM_PROMPT, 
        orch_plan_think_prompt: str = ORCH_PLAN_THINK_PROMPT, 
        orch_plan_reflect_prompt: str = ORCH_PLAN_REFLECT_PROMPT, 
        orch_exec_system_prompt: str = ORCH_EXEC_SYSTEM_PROMPT, 
        orch_exec_think_prompt: str = ORCH_EXEC_THINK_PROMPT, 
        orch_exec_reflect_prompt: str = ORCH_EXEC_REFLECT_PROMPT, 
        exec_system_prompt: str = EXEC_SYSTEM_PROMPT, 
        exec_think_prompt: str = EXEC_THINK_PROMPT, 
        exec_reflect_prompt: str = REFLECT_PROMPT, 
        error_prompt: str = ERROR_PROMPT, 
        orch_plan_think_format: str = "todo", 
        orch_plan_reflect_format: str = "todo", 
        orch_exec_think_format: str = "todo", 
        orch_exec_reflect_format: str = "json", 
        exec_think_format: str = "todo", 
        exec_reflect_format: str = "document", 
        agent_format: str = "todo", 
        semantic_memory_extract: str = SEMANTIC_MEMORY_EXTRACT_PROMPT, 
        episode_memory_extract: str = EPISODE_MEMORY_EXTRACT_PROMPT, 
        procedural_memory_extract: str = PROCEDURAL_MEMORY_EXTRACT_PROMPT, 
        semantic_memory_prompt: str = SEMANTIC_FORMAT, 
        episode_memory_prompt: str = EPISODE_FORMAT, 
        procedural_memory_prompt: str = PROCEDURAL_FORMAT, 
        **kwargs, 
    ) -> None: 
        """
        Initialize the MemoryPlanAndExecAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            embedding_llm (EmbeddingLLM):
                The embedding LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            vector_memory (VectorMemoryDB):
                The vector memory to use for the agent.
            trajectory_memory (TableMemoryDB, optional):
                The trajectory memory to use for the agent.
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
            error_prompt (str, optional):
                The error prompt of the workflow.
            plan_think_format (str, optional):
                The observation format of the plan think stage.
            plan_reflect_format (str, optional):
                The observation format of the plan reflect stage.
            exec_think_format (str, optional):
                The observation format of the exec think stage.
            exec_reflect_format (str, optional):
                The observation format of the exec reflect stage.
            agent_format (str, optional):
                The observation format of the agent.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the vector memory
        self.vector_memory = vector_memory
        self.embedding_llm = embedding_llm
        # Initialize the trajectory memory
        # self.trajectory_memory = trajectory_memory # TODO: 暂时不使用轨迹记忆
        
        super().__init__(
            llm=llm, 
            name=name, 
            agent_type=AgentType.PLAN_AND_EXECUTE, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                "orch_plan_system_prompt": orch_plan_system_prompt, 
                "orch_plan_reason_act_prompt": orch_plan_think_prompt, 
                "orch_plan_reflect_prompt": orch_plan_reflect_prompt, 
                "orch_exec_system_prompt": orch_exec_system_prompt, 
                "orch_exec_reason_act_prompt": orch_exec_think_prompt, 
                "orch_exec_reflect_prompt": orch_exec_reflect_prompt, 
                "exec_system_prompt": exec_system_prompt, 
                "exec_reason_act_prompt": exec_think_prompt, 
                "exec_reflect_prompt": exec_reflect_prompt, 
                "error_prompt": error_prompt, 
            }, 
            observe_formats={
                "orch_plan_reason_act_format": orch_plan_think_format, 
                "orch_plan_reflect_format": orch_plan_reflect_format, 
                "orch_exec_reason_act_format": orch_exec_think_format, 
                "orch_exec_reflect_format": orch_exec_reflect_format, 
                "exec_reason_act_format": exec_think_format, 
                "exec_reflect_format": exec_reflect_format, 
                "agent_format": agent_format, 
            }, 
            memory_prompts={
                "semantic_extract_prompt": semantic_memory_extract, 
                "episode_extract_prompt": episode_memory_extract, 
                "procedural_extract_prompt": procedural_memory_extract, 
                "semantic_prompt_template": semantic_memory_prompt, 
                "episode_prompt_template": episode_memory_prompt, 
                "procedural_prompt_template": procedural_memory_prompt, 
            }, 
            **kwargs,
        )
        
        # Read the workflow profile
        # Initialize the workflow for the agent
        self.workflow = PlanAndExecFlow(
            prompts=self.prompts, 
            observe_formats=self.observe_formats, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"MemoryPlanAndExecAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
