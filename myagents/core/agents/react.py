from asyncio import Lock
from typing import Optional

from fastmcp.client import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import LLM, Workflow, Environment, StepCounter, VectorMemoryDB, EmbeddingLLM
from myagents.core.agents.base import BaseAgent
from myagents.core.agents.memory import BaseMemoryAgent
from myagents.core.agents.types import AgentType
from myagents.core.workflows import BaseReActFlow, TreeTaskReActFlow
from myagents.prompts.workflows.react import PROFILE, SYSTEM_PROMPT, THINK_PROMPT, REFLECT_PROMPT
from myagents.prompts.workflows.plan_and_exec import (
    EXEC_SYSTEM_PROMPT, 
    EXEC_THINK_PROMPT, 
)
from myagents.prompts.memories import (
    SEMANTIC_MEMORY_EXTRACT_PROMPT, 
    EPISODE_MEMORY_EXTRACT_PROMPT, 
    PROCEDURAL_MEMORY_EXTRACT_PROMPT, 
    SEMANTIC_FORMAT, 
    EPISODE_FORMAT, 
    PROCEDURAL_FORMAT, 
)


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
        agent_type (AgentType):
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
        system_prompt: str = SYSTEM_PROMPT, 
        reason_act_prompt: str = THINK_PROMPT, 
        reflect_prompt: str = REFLECT_PROMPT, 
        reason_act_format: str = "document", 
        reflect_format: str = "todo", 
        agent_format: str = "todo", 
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
            system_prompt (str):
                The system prompt of the workflow.
            reason_act_prompt (str):
                The think prompt of the workflow.
            reflect_prompt (str):
                The reflect prompt of the workflow.
            reason_act_format (str):
                The observation format of the think stage.
            reflect_format (str):
                The observation format of the reflect stage.
            agent_format (str):
                The observation format of the agent.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(
            llm=llm, 
            name=name, 
            agent_type=AgentType.REACT, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                "system_prompt": system_prompt, 
                "reason_act_prompt": reason_act_prompt, 
                "reflect_prompt": reflect_prompt, 
            }, 
            observe_formats={
                "reason_act_format": reason_act_format, 
                "reflect_format": reflect_format, 
                "agent_format": agent_format, 
            }, 
            **kwargs,
        )
        
        # Initialize the workflow for the agent
        self.workflow = BaseReActFlow(
            prompts=self.prompts, 
            observe_formats=self.observe_formats, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"ReActAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    
class MemoryReActAgent(ReActAgent, BaseMemoryAgent):
    """MemoryReActAgent is the agent that is used to react to the environment with memory.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
    """
    
    
    def __init__(
        self, 
        name: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        vector_memory: VectorMemoryDB, 
        embedding_llm: EmbeddingLLM, 
        # trajectory_memory: TableMemoryDB, # TODO: 暂时不使用轨迹记忆
        mcp_client: Optional[MCPClient] = None, 
        system_prompt: str = SYSTEM_PROMPT, 
        reason_act_prompt: str = THINK_PROMPT, 
        reflect_prompt: str = REFLECT_PROMPT, 
        reason_act_format: str = "document", 
        reflect_format: str = "todo", 
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
        Initialize the MemoryReActAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            system_prompt (str):
                The system prompt of the workflow.
            reason_act_prompt (str):
                The think prompt of the workflow.
            reflect_prompt (str):
                The reflect prompt of the workflow.
            reason_act_format (str):
                The observation format of the think stage.
            reflect_format (str):
                The observation format of the reflect stage.
            agent_format (str):
                The observation format of the agent.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the vector memory
        self.vector_memory = vector_memory
        self.embedding_llm = embedding_llm
        # Initialize the trajectory memory
        # self.trajectory_memory = trajectory_memory # TODO: 暂时不使用轨迹记忆
        
        # Initialize the parent class
        super().__init__(
            llm=llm, 
            name=name, 
            agent_type=AgentType.REACT, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                "system_prompt": system_prompt, 
                "reason_act_prompt": reason_act_prompt, 
                "reflect_prompt": reflect_prompt, 
            }, 
            observe_formats={
                "reason_act_format": reason_act_format, 
                "reflect_format": reflect_format, 
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
        
        # Initialize the workflow for the agent
        self.workflow = BaseReActFlow(
            prompts=self.prompts, 
            observe_formats=self.observe_formats, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"MemoryReActAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
            

class TreeReActAgent(BaseAgent):
    """TreeReActAgent is the agent that is used to react to the environment.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        agent_type (AgentType):
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
        system_prompt: str = EXEC_SYSTEM_PROMPT, 
        reason_act_prompt: str = EXEC_THINK_PROMPT, 
        reflect_prompt: str = REFLECT_PROMPT, 
        reason_act_format: str = "document", 
        reflect_format: str = "todo", 
        agent_format: str = "todo", 
        **kwargs, 
    ) -> None:        
        """
        Initialize the TreeReActAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            system_prompt (str):
                The system prompt of the workflow.
            reason_act_prompt (str):
                The think prompt of the workflow.
            reflect_prompt (str):
                The reflect prompt of the workflow.
            reason_act_format (str):
                The observation format of the think stage.
            reflect_format (str):
                The observation format of the reflect stage.
            agent_format (str):
                The observation format of the agent.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(
            llm=llm, 
            name=name, 
            agent_type=AgentType.TREE_REACT, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                "system_prompt": system_prompt, 
                "reason_act_prompt": reason_act_prompt, 
                "reflect_prompt": reflect_prompt, 
            }, 
            observe_formats={
                "reason_act_format": reason_act_format, 
                "reflect_format": reflect_format, 
                "agent_format": agent_format, 
            }, 
            **kwargs,
        )
        
        # Initialize the workflow for the agent
        self.workflow = TreeTaskReActFlow(
            prompts=self.prompts, 
            observe_formats=self.observe_formats, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"TreeReActAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class MemoryTreeReActAgent(TreeReActAgent, BaseMemoryAgent):
    """MemoryTreeReActAgent is the agent that is used to react to the environment with memory.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        agent_type (AgentType):
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
        vector_memory: VectorMemoryDB, 
        embedding_llm: EmbeddingLLM, 
        # trajectory_memory: TableMemoryDB, # TODO: 暂时不使用轨迹记忆
        mcp_client: Optional[MCPClient] = None, 
        system_prompt: str = EXEC_SYSTEM_PROMPT, 
        reason_act_prompt: str = EXEC_THINK_PROMPT, 
        reflect_prompt: str = REFLECT_PROMPT, 
        reason_act_format: str = "document", 
        reflect_format: str = "todo", 
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
        Initialize the MemoryTreeReActAgent.
        
        Args:
            name (str):
                The name of the agent.
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            system_prompt (str):
                The system prompt of the workflow.
            reason_act_prompt (str):
                The think prompt of the workflow.
            reflect_prompt (str):
                The reflect prompt of the workflow.
            reason_act_format (str):
                The observation format of the think stage.
            reflect_format (str):
                The observation format of the reflect stage.
            agent_format (str):
                The observation format of the agent.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the vector memory
        self.vector_memory = vector_memory
        self.embedding_llm = embedding_llm
        # Initialize the trajectory memory
        # self.trajectory_memory = trajectory_memory # TODO: 暂时不使用轨迹记忆
        
        # Initialize the parent class
        super().__init__(
            llm=llm, 
            name=name, 
            agent_type=AgentType.TREE_REACT, 
            profile=AGENT_PROFILE.format(name=name, workflow=PROFILE), 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompts={
                "system_prompt": system_prompt, 
                "reason_act_prompt": reason_act_prompt, 
                "reflect_prompt": reflect_prompt, 
            }, 
            observe_formats={
                "reason_act_format": reason_act_format, 
                "reflect_format": reflect_format, 
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
        
        # Initialize the workflow for the agent
        self.workflow = TreeTaskReActFlow(
            prompts=self.prompts, 
            observe_formats=self.observe_formats, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"MemoryTreeReActAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
