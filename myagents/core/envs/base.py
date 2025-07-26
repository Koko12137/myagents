import random
from abc import abstractmethod
from asyncio import Semaphore
from enum import Enum
from uuid import uuid4
from typing import Union, Any, Callable

from fastmcp.tools import Tool as FastMCPTool

from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, Environment, Context, Stateful
from myagents.core.agents import AgentType
from myagents.core.state_mixin import StateMixin
from myagents.core.tools_mixin import ToolsMixin
from myagents.core.llms.config import BaseCompletionConfig


class EnvironmentStatus(Enum):
    """EnvironmentStatus is the status of the environment.
    
    - CREATED: The environment is created.
    - RUNNING: The environment is running.
    - FINISHED: The environment is finished.
    - ERROR: The environment is error.
    - CANCELLED: The environment is cancelled.
    """
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    CANCELLED = "cancelled"


class BaseEnvironment(Environment, ToolsMixin, StateMixin):
    """BaseEnvironment is the base class for all the environments.
    
    Attributes:
        uid (str):
            The unique identifier of the environment.
        name (str):
            The name of the environment.
        profile (str):
            The profile of the environment.
        prompts (dict[str, str]):
            The prompts of the environment. 
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
    prompts: dict[str, str]
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
        name: str, 
        profile: str, 
        prompts: dict[str, str], 
        required_agents: list[AgentType], 
        **kwargs, 
    ) -> None:
        """Initialize the BaseEnvironment.
        
        Args:
            name (str):
                The name of the environment.
            profile (str):
                The profile of the environment.
            prompts (dict[str, str]):
                The prompts of the environment. The key is the prompt name and the value is the prompt content. 
            required_agents (list[AgentType]):
                The required agents to work on the environment. The agents in the list must be registered to the environment. 
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(status_class=EnvironmentStatus, **kwargs)
        
        # Initialize the basic information
        self.uid = str(uuid4())
        self.name = name
        self.profile = profile
        self.prompts = prompts
        self.required_agents = required_agents
        # Initialize the core components
        self.leader = None
        self.agents = {}
        self.agent_type_map = {}
        self.agent_type_semaphore = {}
        # Initialize the tools
        self.post_init()
        # Initialize the status
        self.to_created()
    
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
        if agent.agent_type not in self.agent_type_map:
            self.agent_type_map[agent.agent_type] = []
            # Initialize the semaphore
            self.agent_type_semaphore[agent.agent_type] = Semaphore(1)
        else:
            # Increase the semaphore
            self.agent_type_semaphore[agent.agent_type].release()
        
        self.agent_type_map[agent.agent_type].append(agent.name)
    
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
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: BaseCompletionConfig = None, 
        running_checker: Callable[[Stateful], bool] = None, 
        designated_agent: str = None, 
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
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (BaseCompletionConfig):
                The completion config of the agent. 
            running_checker (Callable[[Stateful], bool]):
                The checker to check if the workflow should be running.
            designated_agent (str):
                The name of the designated agent to call. If not provided, a random agent will be selected. 
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
            if self.agents[designated_agent].agent_type != agent_type: 
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
            completion_config, 
            running_checker=running_checker, 
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
    
    def to_created(self) -> None:
        """Set the environment to created status.
        """
        self.status = EnvironmentStatus.CREATED
        
    def is_created(self) -> bool:
        """Check if the environment is created.
        """
        return self.status == EnvironmentStatus.CREATED
    
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
