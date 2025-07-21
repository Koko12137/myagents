from abc import abstractmethod
from asyncio import Semaphore, Lock
from enum import Enum
from typing import Callable, Any, Union, Protocol, runtime_checkable

from fastmcp.tools import Tool as FastMcpTool
from fastmcp import Client as MCPClient

from myagents.core.interface.core import Stateful, ToolsCaller
from myagents.core.interface.llm import LLM, CompletionConfig
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, ToolCallRequest


class StepCounter(Protocol):
    """StepCounter is a protocol for the step counter. The limit can be max auto steps or max balance cost. It is better to use 
    the same step counter for all agents. 
    
    Attributes:
        uid (str):
            The unique name of the step counter. 
        limit (Union[int, float]):
            The limit of the step counter. 
        current (Union[int, float]):
            The current step of the step counter. 
        lock (Lock):
            The lock of the step counter. 
    """
    uid: str
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


@runtime_checkable
class Agent(Protocol):
    """Agent running on an environment, and working on a task according to the workflow.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        agent_type (Enum):
            The type of the agent.
        profile (str):
            The profile of the agent.
        llm (LLM):
            The LLM to use for the agent. 
        mcp_client (MCPClient):
            The MCP client to use for the agent.
        tools (dict[str, FastMcpTool]):
            The tools to use for the agent.
        workflow (Workflow):
            The workflow to that the agent is running on.
        env (Environment):
            The environment to that the agent is running on.
        step_counters (dict[str, StepCounter]):
            The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
        lock (Lock):
            The synchronization lock of the agent. The agent can only work on one task at a time. 
        prompts (dict[str, str]):
            The prompts for running the workflow. 
        observe_format (dict[str, str]):
            The format of the observation the target. 
    """
    # Basic information
    uid: str
    name: str
    agent_type: Enum
    profile: str
    # LLM and MCP client
    llm: LLM
    mcp_client: MCPClient
    tools: dict[str, FastMcpTool]
    # Workflow and environment and running context
    workflow: 'Workflow'
    env: 'Environment'
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    # Synchronization lock
    lock: Lock
    # Prompts and observe format
    prompts: dict[str, str]
    observe_format: dict[str, str]
    
    # @abstractmethod
    # async def memory(self, *args, **kwargs) -> Any:
    #     """Get the memory of the agent.
        
    #     Returns:
    #         Any:
    #             The memory of the agent.
    #     """
    #     pass
    
    @abstractmethod
    async def observe(
        self, 
        target: Stateful, 
        prompt: str, 
        observe_format: str, 
        **kwargs, 
    ) -> list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
        """Observe the target. 
        
        Args:
            target (Stateful):
                The stateful entity to observe. 
            prompt (str): 
                The prompt before the observation. 
            observe_format (str):
                The format of the observation. This must be a valid observe format of the target
            **kwargs:
                The additional keyword arguments for observing the target. 

        Returns:
            list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
                The up to date information observed from the target.  
        """
        pass    # TODO: 以后需要在 observe 中实现调用 memory，同时通过 Agent属性 来确定 观察格式，禁止通过参数传递，定义一个 workflow 的状态，传入 observe 中，以确定当前的观察方式
    
    @abstractmethod
    async def think(
        self, 
        observe: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]], 
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> AssistantMessage:
        """Think about the observation of the task or environment.
        
        Args:
            observe (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
                The messages observed from the task or environment. 
            completion_config (CompletionConfig):
                The completion config of the agent.
            **kwargs:
                The additional keyword arguments for thinking about the task or environment. 
                
        Returns:
            AssistantMessage:
                The completion message thought about by the LLM. 
        """
        pass    # TODO: 以后需要在 think 中实现更新 memory, 同时通过 Agent属性 和 workflow状态 来确定 Prompt，禁止通过参数传递，定义一个 workflow 的状态，传入 think 中，以确定当前的思考方式
    
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
        pass    # TODO: 以后需要在 act 中实现更新 memory
    
    @abstractmethod
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig, 
        running_checker: Callable[[Stateful], bool], 
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
            completion_config (CompletionConfig):
                The completion config of the agent. 
            running_checker (Callable[[Stateful], bool]):
                The checker to check if the workflow should be running.
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
    
    @abstractmethod
    def register_workflow(self, workflow: 'Workflow') -> None:
        """Register a workflow to the agent.
        
        Args:
            workflow (Workflow):
                The workflow to register.
        """
        pass
    
    @abstractmethod
    def register_env(self, env: 'Environment') -> None:
        """Register an environment to the agent.
        
        Args:
            env (Environment):
                The environment to register.
        """
        pass


class Workflow(ToolsCaller):
    """Workflow is stateless, it does not store any information about the state, it is only used to orchestrate the task or environment. 
    The workflow is not responsible for the state of the task or environment. 
    
    Attributes:
        context (Context):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
        
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to work with the workflow.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        observe_formats (dict[str, str]):
            The formats of the observation. The key is the observation name and the value is the format method name. 
        sub_workflows (dict[str, 'Workflow']):
            The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the sub-workflow instance. 
    """
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_formats: dict[str, str]
    # Sub-worflows
    sub_workflows: dict[str, 'Workflow']
    
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
        completion_config: CompletionConfig, 
        running_checker: Callable[[Stateful], bool], 
        **kwargs, 
    ) -> Stateful:
        """Run the workflow to modify the stateful entity.

        Args:
            target (Stateful): 
                The stateful entity to run the workflow.
            max_error_retry (int):
                The maximum number of times to retry the workflow when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the workflow.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
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
            completion_config: dict[str, Any], 
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
    
    
class ReActFlow(Workflow):
    """ReActFlow is a workflow that can reason and act on the target.
    
    Attributes:
        context (Context):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
        
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to reason and act.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content.
        observe_formats (dict[str, str]):
            The formats of the observation. The key is the observation name and the value is the format method name.
        sub_workflows (dict[str, 'Workflow']):
            The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the sub-workflow instance. 
    """
    
    @abstractmethod
    async def reason_act_reflect(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig, 
        running_checker: Callable[[Stateful], bool], 
        **kwargs, 
    ) -> tuple[Stateful, bool, bool]:
        """Reason and act on the target, and reflect on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on.
            max_error_retry (int):
                The maximum number of times to retry the workflow when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the workflow.
            completion_config (CompletionConfig):
                The completion config of the workflow.
            running_checker (Callable[[Stateful], bool]):
                The checker to check if the workflow should be running.
            *args:
                The additional arguments for running the workflow.
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            tuple[Stateful, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        pass
    
    @abstractmethod
    async def reason_act(
        self, 
        target: Stateful, 
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> tuple[Stateful, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            *args:
                The additional arguments for running the workflow.
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            tuple[Stateful, bool, bool]:
                The target and the error flag and the tool call flag.
        """
        pass
    
    @abstractmethod
    async def reflect(
        self, 
        target: Stateful, 
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> tuple[Stateful, bool]:
        """Reflect on the target.
        
        Args:
            target (Stateful):
                The target to reflect on. 
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            *args:
                The additional arguments for running the workflow.
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            tuple[Stateful, bool]:
                The target and the finish flag.
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
        status (EnvironmentStatus):
            The status of the environment.
        history (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
            The history messages of the environment. 
            
        tools (dict[str, FastMcpTool]):
            The tools that can be used to modify the environment. The key is the tool name and the value is the tool. 
        context (Context):
            The context of the environment.
        
        uid (str):
            The unique identifier of the environment. 
        name (str):
            The name of the environment.
        profile (str):
            The profile of the environment. 
        prompts (dict[str, str]):
            The prompts of the environment. The key is the prompt name and the value is the prompt content. 
        leader (Agent):
            The leader agent of the environment. 
        agents (dict[str, Agent]):
            The agents in the environment. The key is the agent name and the value is the agent. 
        required_agents (list[Enum]):
            The agents in the list must be registered to the environment. 
        agent_type_map (dict[Enum, list[str]]):
            The map of the agent type to the agent name. The key is the agent type and the value is the agent name list. 
        agent_type_semaphore (dict[Enum, Semaphore]):
            The semaphore of the agent type. The key is the agent type and the value is the semaphore. 
    """
    uid: str
    name: str
    profile: str
    prompts: dict[str, str]
    required_agents: list[Enum]
    # Agents and tools
    leader: Agent
    agents: dict[str, Agent]
    agent_type_map: dict[Enum, list[str]]
    agent_type_semaphore: dict[Enum, Semaphore]
    
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
        agent_type: Enum, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig, 
        running_checker: Callable[[Stateful], bool], 
        designated_agent: str, 
        **kwargs, 
    ) -> AssistantMessage:
        """Call an agent to work on the environment or a task and return an assistant message. 
        
        Attention:
            If any type of agent is registered more than one, be careful to the synchronization of the agent. 
            One agent can only work on one task at a time. 
        
        Args:
            agent_type (Enum):
                The type of the agent to call.
            target (Stateful):
                The target to pass to the agent. 
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent. 
            completion_config (CompletionConfig):
                The completion config of the agent. 
            running_checker (Callable[[Stateful], bool], optional):
                The checker to check if the workflow should be running.
            designated_agent (str):
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
