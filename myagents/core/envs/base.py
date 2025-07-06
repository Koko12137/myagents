import asyncio
import random
from abc import abstractmethod
from asyncio import Semaphore
from typing import Union, Any

from fastmcp.tools import Tool as FastMCPTool

from myagents.core.messages.message import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, AgentType, Environment, EnvironmentStatus, Context, Stateful
from myagents.core.utils.context import BaseContext
from myagents.core.tools_mixin import ToolsMixin
from myagents.core.state_mixin import StateMixin


class BaseEnvironment(Environment, ToolsMixin, StateMixin):
    """BaseEnvironment is the base class for all the environments.
    
    Attributes:
        uid (str):
            The unique identifier of the environment.
        name (str):
            The name of the environment.
        profile (str):
            The profile of the environment.
        system_prompt (str):
            The system prompt of the environment.
        required_agents (list[AgentType]):
            The required agents to work on the environment. The agents in the list must be registered to the environment. 
        leader (Agent):
            The leader agent of the environment.
        agents (dict[str, Agent]):
            The agents in the environment. The key is the agent name and the value is the agent. 
        agent_type_map (dict[AgentType, str]):
            The map of the agent type to the agent name. The key is the agent type and the value is the agent name. 
        agent_type_semaphore (dict[AgentType, Semaphore]):
            The semaphore of the agent type. The key is the agent type and the value is the semaphore. 
        tools (dict[str, FastMCPTool]):
            The tools that can be used to modify the environment. The key is the tool name and the value is the tool. 
        context (Context):
            The context of the environment.
        status (EnvironmentStatus):
            The status of the environment.
        history (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
            The history of the environment.
    """
    # Basic information
    uid: str
    name: str
    profile: str
    system_prompt: str
    required_agents: list[AgentType]
    # Core components
    leader: Agent
    agents: dict[str, Agent]
    agent_type_map: dict[AgentType, list[str]]
    agent_type_semaphore: dict[AgentType, Semaphore]
    # Tools Mixin
    tools: dict[str, FastMCPTool]
    context: Context
    # Stateful Mixin
    status: EnvironmentStatus
    history: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]
    
    def __init__(
        self, 
        *args, 
        **kwargs, 
    ) -> None:
        """Initialize the BaseEnvironment.
        
        Args:
            name (str):
                The name of the environment.
            profile (str):
                The profile of the environment.
            required_agents (list[AgentType]):
                The required agents to work on the environment. The agents in the list must be registered to the environment. 
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the basic information
        self.name = None
        self.profile = None
        # Initialize the core components
        self.leader = None
        self.agents = {}
        self.required_agents = []
        self.agent_type_map = {}
        self.agent_type_semaphore = {}
        self.tools = {}
        # Initialize the context
        self.context = BaseContext()
        # Initialize the status and history
        self.status = EnvironmentStatus.CREATED
        self.history = []
        
        # Initialize the tools
        try:
            loop = asyncio.get_running_loop()
            # 如果已经有事件循环，创建任务并等待完成
            task = loop.create_task(self.post_init())
            loop.run_until_complete(task)  # 这行在已运行的loop下会报错
        except RuntimeError:
            # 没有事件循环，直接新建一个
            asyncio.run(self.post_init())
    
    def register_agent(self, agent: Agent) -> None: 
        """Register an agent to the environment.
        
        Args:
            agent (Agent):
                The agent to register.
        """
        # Check if the agent name is already registered
        if agent.name in self.agents:
            return 
        
        # Register the agent to the environment
        self.agents[agent.name] = agent
        # Set the agent type map
        if agent.type not in self.agent_type_map:
            self.agent_type_map[agent.type] = []
            # Initialize the semaphore
            self.agent_type_semaphore[agent.type] = Semaphore(1)
        else:
            # Increase the semaphore
            self.agent_type_semaphore[agent.type].release()
        
        self.agent_type_map[agent.type].append(agent.name)
    
    def set_leader(self, leader_agent: str) -> None:
        """Set the leader agent to the environment.
        
        Args:
            leader_agent (str):
                The name of the leader agent to set.
                
        Raises:
            ValueError:
                If the leader agent name is not registered.
        """
        # Check if the leader agent name is registered
        if leader_agent not in self.agents:
            raise ValueError(f"Leader agent {leader_agent} is not registered.")
        
        # Set the leader agent to the environment
        self.leader = self.agents[leader_agent]
    
    async def call_agent(
        self, 
        agent_type: AgentType, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        designated_agent: str = None, 
        *args, 
        **kwargs
    ) -> AssistantMessage:
        """Call an agent to work on the environment and return the message.
        
        Attention:
            If any type of agent is registered more than one, be careful to the synchronization of the agent. 
            One agent can only work on one task at a time. 
        
        Args:
            agent_type (AgentType):
                The type of the agent to call.
            target (Stateful):
                The target to work on.
            max_error_retry (int, optional):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional):
                The maximum number of times to idle thinking the agent.
            designated_agent (str, optional):
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
        # Check if the agent type is registered
        if agent_type not in self.agent_type_map:
            raise ValueError(f"Agent type `{agent_type}` is not registered. Please register the agent type to the environment.")
        
        # Check if any agent is designed to work on the environment
        if designated_agent is not None:
            # Check if the designated agent is registered
            if designated_agent not in self.agents:
                raise ValueError(f"Agent {designated_agent} is not registered. Please register the agent to the environment.")
            # Check if the designated agent is of the same type
            if self.agents[designated_agent].type != agent_type: 
                raise ValueError(f"Agent {designated_agent} is not of type {agent_type}. Please register the agent to the environment.")
            # Acquire the semaphore
            await self.agent_type_semaphore[agent_type].acquire()
            # Select the designated agent
            agent = self.agents[designated_agent]
        else:
            # Try to get the agent type from the agent type map
            agent_names = self.agent_type_map.get(agent_type, [])
            # Acquire the semaphore
            await self.agent_type_semaphore[agent_type].acquire()
            # Select a random agent from the list of agents without locking
            free_agents = [self.agents[agent_name] for agent_name in agent_names if self.agents[agent_name].lock.locked() is False]
            agent: Agent = random.choice(free_agents)
        
        # Call the agent
        message: AssistantMessage = await agent.run(
            target, 
            max_error_retry, 
            max_idle_thinking, 
            *args, 
            **kwargs,
        )
        # Release the semaphore
        self.agent_type_semaphore[agent_type].release()
        # Return the message
        return message
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """The main entrypoint of the environment. 
        
        Args:
            *args:
                The additional arguments to pass to the run method of the Environment.
            **kwargs:
                The additional keyword arguments to pass to the run method of the Environment.
        """
        pass
    
    def __str__(self) -> str:
        """Get the string representation of the environment.
        """
        return f"BaseEnvironment(name={self.name}, profile={self.profile}, status={self.status})"
    
    def __repr__(self) -> str:
        """Get the string representation of the environment.
        """
        return self.__str__()
    
    def to_created(self) -> None:
        """Set the environment to created status.
        """
        self.status = EnvironmentStatus.CREATED
        
    def is_created(self) -> bool:
        """Check if the environment is created.
        """
        return self.status == EnvironmentStatus.CREATED
    
    def to_planning(self) -> None:
        """Set the environment to planning status.
        """
        self.status = EnvironmentStatus.PLANNING
    
    def is_planning(self) -> bool:
        """Check if the environment is planning.
        """
        return self.status == EnvironmentStatus.PLANNING
    
    def to_running(self) -> None:
        """Set the environment to running status.
        """
        self.status = EnvironmentStatus.RUNNING
    
    def is_running(self) -> bool:
        """Check if the environment is running.
        """
        return self.status == EnvironmentStatus.RUNNING
    
    def to_finished(self) -> None:
        """Set the environment to finished status.
        """
        self.status = EnvironmentStatus.FINISHED
        
    def is_finished(self) -> bool:
        """Check if the environment is finished.
        """
        return self.status == EnvironmentStatus.FINISHED
    
    def to_error(self) -> None:
        """Set the environment to error status.
        """
        self.status = EnvironmentStatus.ERROR
        
    def is_error(self) -> bool:
        """Check if the environment is error.
        """
        return self.status == EnvironmentStatus.ERROR
    
    def to_cancelled(self) -> None:
        """Set the environment to cancelled status.
        """
        self.status = EnvironmentStatus.CANCELLED

    def is_cancelled(self) -> bool:
        """Check if the environment is cancelled.
        """
        return self.status == EnvironmentStatus.CANCELLED
