import asyncio
import random
import json
from abc import abstractmethod
from asyncio import Semaphore
from enum import Enum
from uuid import uuid4
from typing import Union, Any

from json_repair import repair_json
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, ToolCallRequest
from myagents.core.interface import Agent, Environment, Context, Stateful
from myagents.core.agents import AgentType
from myagents.core.tasks.task import BaseTreeTaskNode
from myagents.core.state_mixin import StateMixin
from myagents.core.tools_mixin import ToolsMixin
from myagents.core.utils.context import BaseContext
from myagents.core.utils.strings import normalize_string


class EnvironmentStatus(Enum):
    """EnvironmentStatus is the status of the environment.
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
        *args, 
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
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(status_class=EnvironmentStatus, *args, **kwargs)
        
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
    
    def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the environment.
        
        This method will be called after the initialization of the environment.
        """
        # Register the finish tool
        @self.register_tool("finish_env")
        async def finish_env() -> ToolCallResult:
            """
            完成当前环境，使用这个工具来结束当前环境运行。
            
            Args:
                None
            
            Returns:
                ToolCallResult:
                    The tool call result.
            """
            # Get the tool call
            tool_call: ToolCallRequest = self.context.get("tool_call")
            # Set the environment status to finished
            self.to_finished()
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"环境已设置为 {self.get_status().value} 状态。",
            )
            return result
        
        # Register the create task tool
        @self.register_tool("create_task")
        async def create_task(orchestration: dict[str, dict]) -> ToolCallResult:
            """
            创建一个新的任务，并将其添加到当前任务的子任务中。
            
            Args:
                orchestration (dict[str, dict]): 
                    当前任务的规划蓝图。应该是一个json格式的输入，下面是输入举例:
                    ```json
                    {
                        "任务目标1": {
                            "关键产出 1.1": "关键产出1.1的描述",
                            "关键产出 1.2": "关键产出1.2的描述",
                            ...
                            "关键产出 1.n": "关键产出1.n的描述",
                        },
                        "任务目标2": {
                            "关键产出 2.1": "关键产出2.1的描述",
                            "关键产出 2.2": "关键产出2.2的描述",
                            ...
                            "关键产出 2.n": "关键产出2.n的描述",
                        }
                        ...(其他更多的任务目标)
                    }
                    ```
                
            Returns:
                ToolCallResult: 
                    创建子任务的工具调用结果。
            """
            # Get the parent task from the context
            parent = self.context.get("target")
            # Get the function call details
            tool_call = self.context.get("tool_call")
            
            # Traverse the orchestration
            for key, value in orchestration.items():
                # Create a new task
                new_task = BaseTreeTaskNode(
                    question=normalize_string(key), 
                    description=str(value), 
                    sub_task_depth=parent.sub_task_depth - 1,
                )
                # Link the new task to the parent task
                new_task.parent = parent
                # Add the new task to the parent task
                parent.sub_tasks[new_task.question] = new_task
                # If the sub task depth is 0, then set the task status to running
                if new_task.sub_task_depth == 0:
                    new_task.to_running()
            
            # Create a new tool call result
            tool_call_result = ToolCallResult(
                tool_call_id=tool_call.id,
                content="任务创建成功", 
            )
            return tool_call_result
    
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
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        observe_args: dict[str, dict[str, Any]] = {}, 
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
            prompts (dict[str, str], optional):
                The prompts for running specific workflow of the agent. 
            completion_config (dict[str, Any], optional):
                The completion config of the agent. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            observe_args (dict[str, dict[str, Any]], optional):
                The additional keyword arguments for observing the target. 
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
            prompts, 
            completion_config, 
            observe_args, 
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
