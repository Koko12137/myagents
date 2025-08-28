from asyncio import Lock
from typing import Optional

from fastmcp.client import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import LLM, Workflow, Environment, StepCounter, VectorMemoryCollection, EmbeddingLLM, CallStack, Workspace, PromptGroup
from myagents.core.agents.base import BaseAgent
from myagents.core.agents.memory import BaseMemoryAgent
from myagents.core.agents.types import AgentType
from myagents.core.workflows import BaseReActFlow, TreeTaskReActFlow, MemoryReActFlow, MemoryTreeTaskReActFlow
from myagents.prompts.workflows.react import ReactPromptGroup
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



class ReActAgent(BaseAgent):
    """ReActAgent is the agent that is used to react to the environment.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        agent_type (AgentType):
            The type of the agent.
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
        prompt_group (PromptGroup):
            The prompt group of the workflow.
        observe_formats (dict[str, str]):
            The format of the observation the target. 
    """
    # Basic information
    uid: int
    name: str
    agent_type: AgentType
    # LLM and MCP client
    llms: dict[str, LLM]
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
    prompt_group: PromptGroup
    observe_formats: dict[str, str]
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        name: str, 
        llms: dict[str, LLM], 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        prompt_group: ReactPromptGroup = None, 
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
            llms (dict[str, LLM]):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            prompt_group (ReactPromptGroup):
                The prompt group of the workflow.
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
        # Check if the prompt group is a ReactPromptGroup
        if prompt_group is None:
            prompt_group = ReactPromptGroup()
        elif not isinstance(prompt_group, ReactPromptGroup):
            raise TypeError("prompt_group must be a ReactPromptGroup")
        
        # Initialize the parent class
        super().__init__(
            call_stack=call_stack,
            workspace=workspace,
            name=name, 
            agent_type=AgentType.REACT, 
            llms=llms, 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompt_group=prompt_group, 
            observe_formats={
                "reason_act_format": reason_act_format, 
                "reflect_format": reflect_format, 
                "agent_format": agent_format, 
            }, 
            **kwargs,
        )
        
        # Initialize the workflow for the agent
        self.workflow = BaseReActFlow(
            call_stack=call_stack,
            workspace=workspace,
            prompt_group=self.prompt_group, 
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
    """
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        name: str, 
        llms: dict[str, LLM], 
        step_counters: list[StepCounter], 
        episode_memory: VectorMemoryCollection, 
        embedding_llm: EmbeddingLLM, 
        mcp_client: Optional[MCPClient] = None, 
        prompt_group: ReactPromptGroup = None, 
        reason_act_format: str = "document", 
        reflect_format: str = "todo", 
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
        Initialize the MemoryReActAgent.
        
        Args:
            name (str):
                The name of the agent.
            llms (dict[str, LLM]):
                The LLM to use for the agent.
            episode_memory (VectorMemoryCollection):
                The episode memory to use for the agent.
            embedding_llm (EmbeddingLLM):
                The embedding LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            prompt_group (ReactPromptGroup):
                The prompt group of the workflow.
            reason_act_format (str):
                The observation format of the think stage.
            reflect_format (str):
                The observation format of the reflect stage.
            agent_format (str):
                The observation format of the agent.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Check if the prompt group is a ReactPromptGroup
        if prompt_group is None:
            prompt_group = ReactPromptGroup()
        elif not isinstance(prompt_group, ReactPromptGroup):
            raise TypeError("prompt_group must be a ReactPromptGroup")
        
        # Initialize the parent class
        super().__init__(
            call_stack=call_stack,
            workspace=workspace,
            name=name, 
            llms=llms, 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompt_group=prompt_group, 
            reason_act_format=reason_act_format, 
            reflect_format=reflect_format, 
            agent_format=agent_format, 
            # Memory
            episode_memory=episode_memory, 
            embedding_llm=embedding_llm, 
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
        
        # Initialize the workflow for the agent
        self.workflow = MemoryReActFlow(
            call_stack=call_stack,
            workspace=workspace,
            prompt_group=self.prompt_group, 
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
        prompt_group (PromptGroup):
            The prompt group of the workflow.
        observe_formats (dict[str, str]):
            The format of the observation the target. 
    """
    # Basic information
    uid: int
    name: str
    agent_type: AgentType
    # LLM and MCP client
    llms: dict[str, LLM]
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
    prompt_group: PromptGroup
    observe_formats: dict[str, str]
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        name: str, 
        llms: dict[str, LLM], 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        prompt_group: ReactPromptGroup = None, 
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
            llms (dict[str, LLM]):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            prompt_group (ReactPromptGroup):
                The prompt group of the workflow.
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
        # Check if the prompt group is a ReactPromptGroup
        if prompt_group is None:
            prompt_group = ReactPromptGroup()
        elif not isinstance(prompt_group, ReactPromptGroup):
            raise TypeError("prompt_group must be a ReactPromptGroup")
        
        # Initialize the parent class
        super().__init__(
            call_stack=call_stack,
            workspace=workspace,
            name=name, 
            agent_type=AgentType.TREE_REACT, 
            llms=llms, 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompt_group=prompt_group, 
            observe_formats={
                "reason_act_format": reason_act_format, 
                "reflect_format": reflect_format, 
                "agent_format": agent_format, 
            }, 
            **kwargs,
        )
        
        # Initialize the workflow for the agent
        self.workflow = TreeTaskReActFlow(
            call_stack=call_stack,
            workspace=workspace,
            prompt_group=self.prompt_group, 
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
        prompt_group (PromptGroup):
            The prompt group of the workflow.
        observe_formats (dict[str, str]):
            The format of the observation the target. 
    """
    # Basic information
    uid: int
    name: str
    agent_type: AgentType
    # LLM and MCP client
    llms: dict[str, LLM]
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
    prompt_group: PromptGroup
    observe_formats: dict[str, str]
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        name: str, 
        llms: dict[str, LLM], 
        step_counters: list[StepCounter], 
        episode_memory: VectorMemoryCollection, 
        embedding_llm: EmbeddingLLM, 
        mcp_client: Optional[MCPClient] = None, 
        prompt_group: ReactPromptGroup = None, 
        reason_act_format: str = "document", 
        reflect_format: str = "todo", 
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
        Initialize the MemoryTreeReActAgent.
        
        Args:
            call_stack (CallStack):
                The call stack to use for the agent.
            workspace (Workspace):
                The workspace to use for the agent.
            name (str):
                The name of the agent.
            llms (dict[str, LLM]):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            episode_memory (VectorMemoryCollection):
                The episode memory to use for the agent.
            embedding_llm (EmbeddingLLM):
                The embedding LLM to use for the agent.
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent.
            prompt_group (ReactPromptGroup):
                The prompt group of the workflow.
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
        # 检查 llms 是否包含 reason_act_llm 和 reflect_llm
        if prompt_group is None:
            prompt_group = ReactPromptGroup()
        elif not isinstance(prompt_group, ReactPromptGroup):
            raise TypeError("prompt_group must be a ReactPromptGroup")
        
        # Initialize the parent class
        super().__init__(
            call_stack=call_stack,
            workspace=workspace,
            name=name, 
            llms=llms, 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
            prompt_group=prompt_group, 
            reason_act_format=reason_act_format, 
            reflect_format=reflect_format, 
            agent_format=agent_format, 
            # Memory
            episode_memory=episode_memory, 
            embedding_llm=embedding_llm, 
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
        
        # Initialize the workflow for the agent
        self.workflow = MemoryTreeTaskReActFlow(
            call_stack=call_stack,
            workspace=workspace,
            prompt_group=self.prompt_group, 
            observe_formats=self.observe_formats, 
            **kwargs,
        )
        # Register the agent to the workflow
        self.workflow.register_agent(self)

    def __str__(self) -> str:
        return f"MemoryTreeReActAgent({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
