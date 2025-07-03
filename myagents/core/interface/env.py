from abc import abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable, Callable, Awaitable, Any

from fastmcp.tools import Tool as FastMCPTool

from myagents.core.interface.core import Agent
from myagents.core.interface.context import Context
from myagents.core.interface.message import Message, ToolCallRequest


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
        proxy (Agent):
            The proxy agent. 
        agents (dict[str, Agent]):
            The agents in the environment. The key is the agent name and the value is the agent. 
        context (Context):
            The context of the environment.
        tools (dict[str, FastMCPTool]):
            The tools that can be used to modify the environment. The key is the tool name and the value is the tool. 
        status (EnvironmentStatus):
            The status of the environment.
        history (list[Message]):
            The history messages of the environment. 
    """
    uid: str
    name: str
    profile: str
    
    proxy: Agent
    agents: dict[str, Agent]
    context: Context
    tools: dict[str, FastMCPTool]
    
    status: EnvironmentStatus
    history: list[Message]
    
    @abstractmethod
    async def post_init(self) -> None:
        """Post initialize the tools for the workflow.
        This method should be called after the initialization of the workflow. And you should register the tools in this method. 
        
        Example:
        ```python
        async def post_init(self) -> None:
            
            @self.register_tool("tool_name")
            def tool_function(self, *args, **kwargs) -> Any:
                pass
        ```
        """
        pass
    
    @abstractmethod
    def add_tool(
        self, 
        name: str, 
        tool: Callable[..., Awaitable[Message]], 
        tags: list[str] = [],
    ) -> None:
        """Add a tool to the mixin.
        
        Args:
            name (str):
                The name of the tool.
            tool (Callable[..., Awaitable[Message]]):
                The tool to add. This tool should return the message. 
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
    ) -> Callable[..., Awaitable[Message]]:
        """This is a FastAPI like decorator to register a tool to the mixin.
        
        Args:
            name (str):
                The name of the tool.
            tags (list[str], optional):
                The tags of the tool.
                
        Returns:
            Callable[..., Awaitable[Message]]:
                The function registered.
        """
        pass

    @abstractmethod
    async def call_tool(
        self, 
        ctx: 'Environment', 
        tool_call: ToolCallRequest, 
        **kwargs, 
    ) -> Message:
        """Call a tool to control the environment.
        
        Args:
            ctx (Environment):
                The environment to call the tool.
            tool_call (ToolCallRequest):
                The tool call request.
            **kwargs:
                The additional keyword arguments for calling the tool.
                
        Returns:
            Message:
                The message returned by the tool call. 
                
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
    def set_proxy(self, proxy_agent: str) -> None:
        """Set the proxy agent to the environment.
        
        Args:
            proxy_agent (str):
                The name of the proxy agent to set.
        """
        pass
    
    @abstractmethod
    async def call_agent(
        self, 
        agent_name: str, 
        *args, 
        **kwargs, 
    ) -> Message:
        """Call an agent to work on the environment and return the message.
        
        Args:
            agent_name (str):
                The name of the agent to call.
            *args:
                The additional arguments to pass to the agent.
            **kwargs:
                The additional keyword arguments to pass to the agent.
                
        Returns:
            Message:
                The message returned by the agent.
                
        Raises:
            ValueError:
                If the agent name is not registered.
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
