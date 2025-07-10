from abc import abstractmethod
from asyncio import Semaphore, Lock
from enum import Enum
from typing import Callable, Awaitable, Any, Union, Optional, Protocol, runtime_checkable

from fastmcp.tools import Tool as FastMCPTool
from fastmcp import Client as MCPClient
from mcp import Tool as MCPTool

from myagents.core.interface.core import Stateful, ToolsCaller, Context
from myagents.core.interface.llm import LLM
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, ToolCallRequest


class StepCounter(Protocol):
    """StepCounter is a protocol for the step counter. The limit can be max auto steps or max balance cost. It is better to use 
    the same step counter for all agents. 
    
    Attributes:
        uname (str):
            The unique name of the step counter. 
        limit (Union[int, float]):
            The limit of the step counter. 
        current (Union[int, float]):
            The current step of the step counter. 
        lock (Lock):
            The lock of the step counter. 
    """
    uname: str
    limit: Union[int, float]
    current: Union[int, float]
    lock: Lock
    
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


class AgentType(Enum):
    """The type of the agent.
    
    Attributes:
        REACT (str):
            The reason and act agent. This agent works on a basic reason and act workflow. 
        ORCHESTRATE (str):
            The orchestrator agent. This agent works on an objective and key outputs orchestration workflow. 
        PLAN_AND_EXECUTE (str):
            The plan and executor agent. This agent works on a plan and executor workflow. 
    """
    REACT = "react"
    ORCHESTRATE = "orchestrate"
    PLAN_AND_EXECUTE = "plan_and_execute"


@runtime_checkable
class Agent(Protocol):
    """Agent running on an environment, and working on a task according to the workflow.
    
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
    """
    # Basic information
    uid: str
    name: str
    type: AgentType
    profile: str
    # LLM and MCP client
    llm: LLM
    mcp_client: MCPClient
    # Workflow and environment and running context
    workflow: 'Workflow'
    env: 'Environment'
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    # Synchronization lock
    lock: Lock
    
    @abstractmethod
    async def observe(
        self, 
        target: Stateful, 
        observe_func: Optional[Callable[..., Awaitable[Union[str, list[dict]]]]] = None, 
        **kwargs, 
    ) -> Union[str, list[dict]]:
        """Observe the target. If the target is not a task or environment, you should provide the observe 
        function to get the string or list of dicts observation. 
        
        Args:
            target (Stateful):
                The stateful entity to observe. 
            observe_func (Callable[..., Awaitable[Union[str, list[dict]]]], optional):
                The function to observe the target. If not provided, the default observe function will be used. 
            **kwargs:
                The additional keyword arguments for observing the target. 

        Returns:
            Union[str, list[dict]]:
                The up to date information observed from the target.  
        """
        pass
    
    @abstractmethod
    async def think(
        self, 
        observe: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]], 
        tools: dict[str, Union[FastMCPTool, MCPTool]] = {}, 
        tool_choice: Optional[str] = 'none', 
        **kwargs, 
    ) -> AssistantMessage:
        """Think about the observation of the task or environment.
        
        Args:
            observe (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
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
            AssistantMessage:
                The completion message thought about by the LLM. 
        """
        pass
    
    @abstractmethod
    async def act(self, tool_call: ToolCallRequest, **kwargs) -> ToolCallResult:
        """Take an Action according to the tool call. Other arguments can be provided to the tool 
        calling through the keyword arguments. 
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request including the tool call id and the tool call arguments.
            **kwargs:
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult:
                The tool call result returned by the agent after acting on the environment or task.
            
        Raises:
            ValueError:
                If the tool call name is not registered to the workflow or environment.  
        """
        pass
    
    @abstractmethod
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [],  
        *args, 
        **kwargs
    ) -> AssistantMessage:
        """Run the agent on the task or environment. Before running the agent, you should get the lock of the agent. 
        
        Args:
            target (Stateful):
                The stateful entity to run the agent on.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent. 
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            AssistantMessage:
                The assistant message returned by the agent after running on the stateful entity.
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


class LeaderAgent(Agent):
    """LeaderAgent is a main agent that represent for other agents.
    
    Attributes:
        out_env (Environment):
            The outside environment that the leader agent is running on.
    """
    out_env: 'Environment'


class Workflow(ToolsCaller):
    """Workflow is stateless, it does not store any information about the state, it is only used to orchestrate the task or environment. 
    The workflow is not responsible for the state of the task or environment. 
    
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to work with the workflow.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        context (Context):
            The context of the workflow.
        tools (dict[str, FastMCPTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
    """
    profile: str
    agent: Agent
    prompts: dict[str, str]
    
    @abstractmethod
    def register_agent(self, agent: Agent) -> None:
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
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        tool_choice: str, 
        exclude_tools: list[str], 
        running_checker: Callable[[Stateful], bool], 
        *args, 
        **kwargs, 
    ) -> Stateful:
        """Run the workflow from the environment or task.

        Args:
            target (Stateful): 
                The stateful entity to run the workflow.
            max_error_retry (int):
                The maximum number of times to retry the workflow when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the workflow.
            tool_choice (str):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str]):
                The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool]):
                The checker to check if the workflow should be running.
            *args:
                The additional arguments for running the workflow.
            **kwargs:
                The additional keyword arguments for running the workflow.

        Returns:
            Stateful: 
                The stateful entity after running the workflow.
                
        Example:
        ```python
        async def run(
            self, 
            target: Stateful, 
            max_error_retry: int, 
            max_idle_thinking: int, 
            tool_choice: str, 
            exclude_tools: list[str], 
            running_checker: Callable[[Stateful], bool], 
            *args, 
            **kwargs,
        ) -> Stateful:
            # Update system prompt to history
            message = SystemMessage(content=self.system_prompt)
            
            # Check if the target is running
            if running_checker(target):
                # Run the workflow
                # Observe the task
                observe = await self.observe(target)
                # Think about the task
                completion = await self.think(observe)
                # Act on the task
                target = await self.act(target, completion)
                # Reflect the task
                target = await self.reflect(target)
            else:
                # Set the target to error
                target.to_error()
                
            # Return the target
            return target
        ```
        """
        pass


class EnvironmentStatus(Enum):
    """The status of the environment.
    
    - CREATED (int): The environment is created.
    - PLANNING (int): The environment is planning.
    - RUNNING (int): The environment is running.
    - FINISHED (int): The environment is finished.
    - ERROR (int): The environment is errored.
    - CANCELLED (int): The environment is cancelled.
    """
    CREATED = 0
    PLANNING = 1
    RUNNING = 2
    FINISHED = 3
    ERROR = 4
    CANCELLED = 5


class Environment(Stateful, ToolsCaller):
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
    leader: 'LeaderAgent'
    agents: dict[str, Agent]
    agent_type_map: dict[AgentType, list[str]]
    agent_type_semaphore: dict[AgentType, Semaphore]
    # Tools Mixin
    tools: dict[str, FastMCPTool]
    context: Context
    # Stateful Mixin
    status: EnvironmentStatus
    history: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]
    
    @abstractmethod
    def register_agent(self, agent: Agent) -> None:
        """Register an agent to the environment.
        
        Args:
            agent (Agent):
                The agent to register. 
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
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        designated_agent: str = None, 
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
            target (Stateful):
                The target to pass to the agent. 
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
            designated_agent (str, optional, defaults to None):
                The name of the designated agent to call. If not provided, a random agent will be selected. 
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
