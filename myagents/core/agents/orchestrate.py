from asyncio import Lock
from typing import Optional

from fastmcp.client import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import LLM, Workflow, Environment, StepCounter, VectorMemoryCollection, EmbeddingLLM
from myagents.core.agents.base import BaseAgent
from myagents.core.agents.memory import BaseMemoryAgent
from myagents.core.agents.types import AgentType
from myagents.core.workflows import OrchestrateFlow, MemoryOrchestrateFlow, MemoryCompressWorkflow
from myagents.prompts.workflows.orchestrate import (
    PROFILE, 
    PLAN_SYSTEM_PROMPT, 
    PLAN_THINK_PROMPT, 
    PLAN_REFLECT_PROMPT, 
    EXEC_SYSTEM_PROMPT, 
    EXEC_THINK_PROMPT, 
    EXEC_REFLECT_PROMPT, 
)
from myagents.prompts.memories.compress import (
    SYSTEM_PROMPT as MEMORY_COMPRESS_SYSTEM_PROMPT, 
    REASON_ACT_PROMPT as MEMORY_COMPRESS_REASON_ACT_PROMPT, 
)
from myagents.prompts.memories.episode import (
    SYSTEM_PROMPT as MEMORY_EPISODE_SYSTEM_PROMPT, 
    REASON_ACT_PROMPT as MEMORY_EPISODE_REASON_ACT_PROMPT, 
    REFLECT_PROMPT as MEMORY_EPISODE_REFLECT_PROMPT, 
)
from myagents.prompts.memories.template import MEMORY_PROMPT_TEMPLATE, SYSTEM_MEMORY_TEMPLATE


AGENT_PROFILE = """
我叫 {name} ，是一个会按照“编排-反思”的流程来编排任务总体目标的助手。

以下是我的工作流信息：
{workflow}
"""


class OrchestrateAgent(BaseAgent):
    """OrchestrateAgent is the agent that is used to orchestrate the environment.
    
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
        need_user_check: bool = False, 
        plan_system_prompt: str = PLAN_SYSTEM_PROMPT, 
        plan_reason_act_prompt: str = PLAN_THINK_PROMPT, 
        plan_reflect_prompt: str = PLAN_REFLECT_PROMPT, 
        exec_system_prompt: str = EXEC_SYSTEM_PROMPT, 
        exec_reason_act_prompt: str = EXEC_THINK_PROMPT, 
        exec_reflect_prompt: str = EXEC_REFLECT_PROMPT, 
        plan_reason_act_format: str = "todo", 
        plan_reflect_format: str = "todo", 
        exec_reason_act_format: str = "json", 
        exec_reflect_format: str = "json", 
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
            need_user_check (bool, optional, defaults to False):
                Whether to need the user to check the orchestration blueprint.
            plan_system_prompt (str, optional):
                The system prompt of the orchestation plan reason stage.
            plan_reason_act_prompt (str, optional):
                The think prompt of the orchestation plan reason stage.
            plan_reflect_prompt (str, optional):
                The react system prompt of the orchestation plan reflect stage.
            exec_system_prompt (str, optional):
                The action prompt of the orchestation execution reason stage.
            exec_reason_act_prompt (str, optional):
                The reflect prompt of the orchestation execution reason stage.
            exec_reflect_prompt (str, optional):
                The reflect prompt of the orchestation execution reflect stage.
            plan_reason_act_format (str, optional, defaults to "todo"):
                The observation format of the orchestation execution reason stage.
            plan_reflect_format (str, optional, defaults to "todo"):
                The observation format of the orchestation execution reflect stage.
            exec_reason_act_format (str, optional, defaults to "json"):
                The observation format of the orchestation execution reason stage.
            exec_reflect_format (str, optional, defaults to "json"):
                The observation format of the orchestation execution reflect stage.
            agent_format (str, optional, defaults to "todo"):
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
            }, 
            observe_formats={
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
            observe_formats=self.observe_formats, 
            need_user_check=need_user_check, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"OrchestrateAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()

class MemoryOrchestrateAgent(OrchestrateAgent, BaseMemoryAgent):
    """MemoryOrchestrateAgent is the agent that is used to orchestrate the environment with memory.
    
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
        episode_memory: VectorMemoryCollection, 
        embedding_llm: EmbeddingLLM, 
        extraction_llm: LLM, 
        mcp_client: Optional[MCPClient] = None, 
        need_user_check: bool = False, 
        plan_system_prompt: str = PLAN_SYSTEM_PROMPT, 
        plan_reason_act_prompt: str = PLAN_THINK_PROMPT, 
        plan_reflect_prompt: str = PLAN_REFLECT_PROMPT, 
        exec_system_prompt: str = EXEC_SYSTEM_PROMPT, 
        exec_reason_act_prompt: str = EXEC_THINK_PROMPT, 
        exec_reflect_prompt: str = EXEC_REFLECT_PROMPT, 
        plan_reason_act_format: str = "todo", 
        plan_reflect_format: str = "todo", 
        exec_reason_act_format: str = "json", 
        exec_reflect_format: str = "json", 
        agent_format: str = "todo", 
        # Memory Compress
        memory_compress_system_prompt: str = MEMORY_COMPRESS_SYSTEM_PROMPT, 
        memory_compress_reason_act_prompt: str = MEMORY_COMPRESS_REASON_ACT_PROMPT, 
        # Episode Memory
        episode_memory_system_prompt: str = MEMORY_EPISODE_SYSTEM_PROMPT, 
        episode_memory_reason_act_prompt: str = MEMORY_EPISODE_REASON_ACT_PROMPT, 
        episode_memory_reflect_prompt: str = MEMORY_EPISODE_REFLECT_PROMPT, 
        # Memory Format Template
        memory_prompt_template: str = MEMORY_PROMPT_TEMPLATE, 
        system_memory_template: str = SYSTEM_MEMORY_TEMPLATE, 
        **kwargs, 
    ) -> None: 
        """
        Initialize the MemoryOrchestrateAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            episode_memory (VectorMemoryDB):
                The vector memory to use for the agent.
            embedding_llm (EmbeddingLLM):
                The embedding LLM to use for the agent.
            extraction_llm (LLM):
                The extraction LLM to use for the agent.
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            need_user_check (bool, optional, defaults to False):
                Whether to need the user to check the orchestration blueprint.
            plan_system_prompt (str, optional):
                The system prompt of the orchestation plan reason stage.
            plan_reason_act_prompt (str, optional):
                The think prompt of the orchestation plan reason stage.
            plan_reflect_prompt (str, optional):
                The react system prompt of the orchestation plan reflect stage.
            exec_system_prompt (str, optional):
                The action prompt of the orchestation execution reason stage.
            exec_reason_act_prompt (str, optional):
                The reflect prompt of the orchestation execution reason stage.
            exec_reflect_prompt (str, optional):
                The reflect prompt of the orchestation execution reflect stage.
            plan_reason_act_format (str, optional, defaults to "todo"):
                The observation format of the orchestation execution reason stage.
            plan_reflect_format (str, optional, defaults to "todo"):
                The observation format of the orchestation execution reflect stage.
            exec_reason_act_format (str, optional, defaults to "json"):
                The observation format of the orchestation execution reason stage.
            exec_reflect_format (str, optional, defaults to "json"):
                The observation format of the orchestation execution reflect stage.
            agent_format (str, optional, defaults to "todo"):
                The observation format of the agent.
            system_memory_template (str, optional):
                The system memory template of the memory.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(
            llm=llm, 
            name=name, 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            need_user_check=need_user_check, 
            plan_system_prompt=plan_system_prompt, 
            plan_reason_act_prompt=plan_reason_act_prompt, 
            plan_reflect_prompt=plan_reflect_prompt, 
            exec_system_prompt=exec_system_prompt, 
            exec_reason_act_prompt=exec_reason_act_prompt, 
            exec_reflect_prompt=exec_reflect_prompt, 
            plan_reason_act_format=plan_reason_act_format, 
            plan_reflect_format=plan_reflect_format, 
            exec_reason_act_format=exec_reason_act_format, 
            exec_reflect_format=exec_reflect_format, 
            agent_format=agent_format, 
            # Memory
            episode_memory=episode_memory, 
            embedding_llm=embedding_llm, 
            extraction_llm=extraction_llm, 
            # Memory Compress
            memory_compress_system_prompt=memory_compress_system_prompt, 
            memory_compress_reason_act_prompt=memory_compress_reason_act_prompt, 
            # Episode Memory
            episode_memory_system_prompt=episode_memory_system_prompt, 
            episode_memory_reason_act_prompt=episode_memory_reason_act_prompt, 
            episode_memory_reflect_prompt=episode_memory_reflect_prompt, 
            # Memory Format Template
            memory_prompt_template=memory_prompt_template, 
            system_memory_template=system_memory_template, 
            **kwargs,
        )
        
        # Read the workflow profile
        # Initialize the workflow for the agent
        self.workflow = MemoryOrchestrateFlow(
            prompts=self.prompts, 
            observe_formats=self.observe_formats, 
            need_user_check=need_user_check, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"MemoryOrchestrateAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
