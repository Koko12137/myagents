from abc import abstractmethod
from asyncio import Semaphore
from enum import Enum
from typing import Protocol, runtime_checkable, Callable, Awaitable, Any, Union

from fastmcp.tools import Tool as FastMCPTool

from myagents.core.interface.core import Agent, AgentType
from myagents.core.interface.context import Context
from myagents.core.messages.message import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, ToolCallRequest


class EnvironmentStatus(Enum):
    """The status of the environment.
    
    - CREATED (int): The environment is created.
    - RUNNING (int): The environment is running.
    - COMPLETED (int): The environment is completed.
    - FAILED (int): The environment is failed.
    """
    CREATED = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = -1
    
    
@runtime_checkable
class Environment(Protocol):
    """Environment is a stateful object that containing workflows. The workflows can be used to think about how to 
    modify the environment. The tools can be used to modify the environment. 
    
    Attributes:
        uid (str):
            The unique identifier of the environment. 
        name (str):
            The name of the environment.
        profile (str):
            The profile of the environment. 
        system_prompt (str):
            The system prompt of the environment. 
        leader (Agent):
            The leader agent of the environment. 
        agents (dict[str, Agent]):
            The agents in the environment. The key is the agent name and the value is the agent. 
        required_agents (list[AgentType]):
            The agents in the list must be registered to the environment. 
        agent_type_map (dict[AgentType, list[str]]):
            The map of the agent type to the agent name. The key is the agent type and the value is the agent name list. 
        agent_type_semaphore (dict[AgentType, Semaphore]):
            The semaphore of the agent type. The key is the agent type and the value is the semaphore. 
        tools (dict[str, FastMCPTool]):
            The tools that can be used to modify the environment. The key is the tool name and the value is the tool. 
        context (Context):
            The context of the environment.
        status (EnvironmentStatus):
            The status of the environment.
        history (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
            The history messages of the environment. 
    """
    uid: str
    name: str
    profile: str
    required_agents: list[AgentType]
    system_prompt: str
    # Agents and tools
    leader: Agent
    agents: dict[str, Agent]
    agent_type_map: dict[AgentType, list[str]]
    agent_type_semaphore: dict[AgentType, Semaphore]
    # Tools
    tools: dict[str, FastMCPTool]
    # Context
    context: Context
    # Status and history
    status: EnvironmentStatus
    history: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]
    
    @abstractmethod
    async def post_init(self) -> None:
        """Post initialize the tools for control the environment. Any subclass should register the 
        tools in this method for the tools to be used in the environment. 
        
        Example:
        ```python
        async def post_init(self) -> None:
            
            @self.register_tool("tool_name")
            async def tool_function(self, *args, **kwargs) -> ToolCallResult:
                pass
        ```
        """
        pass
    
    @abstractmethod
    def add_tool(
        self, 
        name: str, 
        tool: Callable[..., Union[
            Callable[..., Awaitable[ToolCallResult]], 
            Callable[..., ToolCallResult], 
        ]], 
        tags: list[str] = [],
    ) -> None:
        """Add a tool to the mixin.
        
        Args:
            name (str):
                The name of the tool.
            tool (Callable[..., Union[Callable[..., Awaitable[ToolCallResult]], Callable[..., ToolCallResult]]]):
                The tool to add. This tool should return the tool call result. 
            tags (list[str], optional):
                The tags of the tool.
                
        Raises:
            ValueError:
                If the tool name is already registered.
        """
        pass
    
    @abstractmethod
    def register_tool(
        self, 
        name: str, 
        tags: list[str] = [], 
    ) -> Callable[..., Union[
        Callable[..., Awaitable[ToolCallResult]], 
        Callable[..., ToolCallResult], 
    ]]:
        """This is a FastAPI like decorator to register a tool to the environment.
        
        Args:
            name (str):
                The name of the tool.
            tags (list[str], optional):
                The tags of the tool.
                
        Returns:
            Callable[..., Callable[..., Awaitable[ToolCallResult]]]:
                If the tool is async, return the async function.
            Callable[..., Callable[..., ToolCallResult]]:
                If the tool is sync, return the sync function.
        """
        pass

    @abstractmethod
    async def call_tool(self, tool_call: ToolCallRequest, **kwargs) -> ToolCallResult:
        """Call a tool to control the environment.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
            **kwargs:
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult:
                The tool call result. 
                
        Raises:
            ValueError:
                If the tool call name is not registered. 
        """
        pass
    
    @abstractmethod
    def register_agent(self, agent: Agent) -> None:
        """Register an agent to the environment.
        
        Args:
            agent (Agent):
                The agent to register. 
                
        Raises:
            ValueError:
                If the agent name is already registered. 
        """
        pass
    
    @abstractmethod
    def set_leader(self, leader_agent: str) -> None:
        """Set the leader agent to the environment.
        
        Args:
            leader_agent (str):
                The name of the leader agent to set.
        """
        pass
    
    @abstractmethod
    async def call_agent(
        self, 
        agent_type: AgentType, 
        *args, 
        **kwargs, 
    ) -> AssistantMessage:
        """Call an agent to work on the environment or a task and return an assistant message. 
        
        Attention:
            If any type of agent is registered more than one, be careful to the synchronization of the agent. 
            One agent can only work on one task at a time. 
        
        Args:
            agent_type (AgentType):
                The type of the agent to call.
            *args:
                The additional arguments to pass to the agent.
            **kwargs:
                The additional keyword arguments to pass to the agent.
                
        Returns:
            AssistantMessage:
                The message returned by the agent.
                
        Raises:
            ValueError:
                If the type of this agent is not registered.
        """
        pass
    
    @abstractmethod
    async def update(self, message: Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]) -> None:
        """Update the history of environment with the message. This method will merge the message if the role is the same 
        with the last message automatically except for the tool call result. 
        
        Args:
            message (Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]):
                The message to update the environment.
        """
        pass
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """The main entrypoint of the environment. 
        
        Args:
            *args:
                The additional arguments to pass to the run method of the Environment.
            **kwargs:
                The additional keyword arguments to pass to the run method of the Environment.
                
        Returns:
            Any:
                The result returned by the environment.
        """
        pass
