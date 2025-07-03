from abc import abstractmethod
from asyncio import Lock
from typing import Protocol, runtime_checkable, Callable, Union, Optional, Awaitable

from mcp import Tool as MCPTool
from fastmcp import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface.logger import Logger
from myagents.core.interface.llm import LLM
from myagents.core.interface.context import Context
from myagents.core.interface.task import Task
from myagents.core.interface.env import Environment
from myagents.core.interface.message import Message, ToolCallRequest


class StepCounter(Protocol):
    """StepCounter is a protocol for the step counter. The limit can be max auto steps or max balance cost. It is better to use 
    the same step counter for all agents. 
    
    Attributes:
        uid (str):
            The unique identifier of the step counter. 
        limit (Union[int, float]):
            The limit of the step counter. 
        current (Union[int, float]):
            The current step of the step counter. 
        custom_logger (Logger):
            The custom logger to use for the step counter. 
    """
    uid: str
    limit: Union[int, float]
    current: Union[int, float]
    lock: Lock
    custom_logger: Logger
    
    @abstractmethod
    async def reset(self) -> None:
        """Reset the current step of the step counter.
        
        Returns:
            None
        """
        pass
    
    @abstractmethod
    async def update_limit(self, limit: Union[int, float]) -> None:
        """Update the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        pass
    
    @abstractmethod
    async def check_limit(self) -> bool:
        """Check if the limit of the step counter is reached.
        
        Returns:
            bool:
                Whether the limit of the step counter is reached.
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        pass
    
    @abstractmethod
    async def step(self, step: Union[int, float]) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (Union[int, float]):
                The step to increment. 
        
        Returns:
            None 
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        pass
    
    @abstractmethod
    async def recharge(self, limit: Union[int, float]) -> None:
        """Recharge the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        pass


@runtime_checkable
class Workflow(Protocol):
    """Workflow is stateless, it does not store any information about the state, it is only used to orchestrate the task or environment. 
    The workflow is not responsible for the state of the task or environment. 
    
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent. 
        context (Context):
            The context of the workflow.
        tools (dict[str, FastMCPTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
    """
    profile: str
    agent: 'Agent'
    context: Context
    tools: dict[str, FastMcpTool]
    
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
    def register_agent(self, agent: 'Agent') -> None:
        """Register a agent to the workflow.
        
        Args:
            agent (Agent):
                The agent to register.
                
        Raises:
            ValueError:
                If the workflow already has an agent.
        """
        pass

    @abstractmethod
    async def call_tool(
        self, 
        task: Task, 
        tool_call: ToolCallRequest, 
        **kwargs, 
    ) -> Message:
        """Call a tool to control the workflow.
        
        Args:
            task (Task):
                The task to call the tool.
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
    async def run(self, task: Task) -> Task:
        """Run the workflow from the environment or task.

        Args:
            task (Task): 
                The task to run the workflow.

        Returns:
            Task: 
                The task after running the workflow.
                
        Example:
        ```python
        async def run(self, task: Task) -> Task:
            
            # A while loop to run the workflow until the task is finished.
            while task.status != TaskStatus.FINISHED:
                # 1. Observe the task
                observe = await self.observe(task)
                # 2. Think about the task
                completion = await self.think(observe, allow_tools=True)
                # 3. Act on the task
                task = await self.act(task, completion.tool_calls)
                # 4. Reflect the task
                task = await self.reflect(task)
                
            return task
        ```
        """
        pass


@runtime_checkable
class Agent(Protocol):
    """Agent running on an environment, and working on a task according to the workflow.
    
    Attributes:
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
    uid: str
    name: str
    profile: str
    # LLM and MCP client
    llm: LLM
    mcp_client: MCPClient
    # Workflow and environment
    workflow: Workflow
    env: Environment
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    
    @abstractmethod
    async def observe(self, target: Union[Task, Environment], **kwargs) -> str:
        """Observe the task or environment.
        
        Args:
            target (Union[Task, Environment]):
                The task or environment to observe. 
            **kwargs:
                The additional keyword arguments for observing the task or environment. 

        Returns:
            str:
                The up to date information observed from the task or environment.  
        """
        pass
    
    @abstractmethod
    async def think(
        self, 
        observe: list[Message], 
        tools: dict[str, Union[FastMcpTool, MCPTool]] = {}, 
        tool_choice: Optional[str] = 'none', 
        **kwargs, 
    ) -> Message:
        """Think about the observation of the task or environment.
        
        Args:
            observe (list[Message]):
                The messages observed from the task or environment. 
            tools (Optional[dict[str, Union[FastMcpTool, MCPTool]]], defaults to {}):
                The tools allowed to be used for the agent. 
            tool_choice (Optional[str], defaults to None):
                The tool choice to use for the agent. This is used to control the tool calling. 
                - "auto": The agent will automatically choose the tool to use. 
                - "none": The agent will not use any tool. 
                - "all": The agent will use all the tools. 
            **kwargs:
                The additional keyword arguments for thinking about the task or environment. 
                
        Returns:
            Message:
                The completion message thought about by the LLM. 
        """
        pass
    
    @abstractmethod
    async def act(self, target: Union[Task, Environment], tool_call: ToolCallRequest, **kwargs) -> Message:
        """Act on the environment or task.
        
        Args:
            target (Union[Task, Environment]):
                The task or environment to act on.
            tool_call (ToolCallRequest):
                The tool call request to act on the environment.
            **kwargs:
                The additional keyword arguments for acting on the environment.
                
        Returns:
            Message:
                The message returned by the agent after acting on the environment or task.
            
        Raises:
            ValueError:
                If the tool call name is not registered to the workflow or environment.  
        """
        pass
    
    @abstractmethod
    async def run(self, target: Union[Task, Environment], **kwargs) -> Message:
        """Run the agent on the task or environment.
        
        Args:
            target (Union[Task, Environment]):
                The task or environment to run the agent on.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            Message:
                The message returned by the agent after running on the environment or task.
        """
        pass
    
    @abstractmethod
    def register_counter(self, counter: StepCounter) -> None:
        """Register a step counter to the agent.
        
        Args:
            counter (StepCounter):
                The step counter to register.
        """
        pass


class ProxyAgent(Agent):
    """ProxyAgent is a main agent that represent for other agents.
    
    Attributes:
        out_env (Environment):
            The outside environment that the proxy agent is running on.
    """
    out_env: Environment
    
    @abstractmethod
    async def run(self, target: Union[Task, Environment], **kwargs) -> Message:
        """Run the proxy agent on the task or environment.
        
        Args:
            target (Union[Task, Environment]):
                The task or environment to run the proxy agent on.
            **kwargs:
                The additional keyword arguments for running the proxy agent.
                
        Returns:
            Message:
                The message returned by the proxy agent after running on the environment or task.
        """
        pass
