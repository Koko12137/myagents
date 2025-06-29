from abc import abstractmethod
from asyncio import Lock
from enum import Enum
from typing import Protocol, runtime_checkable, Any, OrderedDict, Callable, Union, Optional, AsyncGenerator, Hashable

from fastmcp import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool
from mcp import Tool as MCPTool
    
from myagents.core.message import CompletionUsage, CompletionMessage, ToolCallRequest, ToolCallResult, MessageRole, StopReason


@runtime_checkable
class Queue(Protocol):
    """Queue is a protocol for the queue.
    """
    @abstractmethod
    async def put(self, *args, **kwargs) -> None:
        """Put an item into the queue.
        
        Args:
            *args:
                The additional arguments to pass to the put method.
            **kwargs:
                The additional keyword arguments to pass to the put method.
        """
        pass
    
    @abstractmethod
    async def get(self, *args, **kwargs) -> Any:
        """Get an item from the queue.
        
        Args:
            *args:
                The additional arguments to pass to the get method.
            **kwargs:
                The additional keyword arguments to pass to the get method.
        """
        pass


class Provider(Enum):
    DUMMY = "dummy"
    OPENAI = "openai"
    QUEUE = "queue"     # This could be a adapter for the AReaL framework.


@runtime_checkable
class LLM(Protocol):
    """LLM is a protocol for Language Model.
    
    Attributes:
        provider (Provider) :
            The provider of the LLM.
        model (str) :
            The model of the LLM. 
        base_url (str) :
            The base URL of the LLM. 
        custom_logger (Logger):
            The custom logger. If not provided, the default loguru logger will be used. 
        debug (bool):
            The debug flag. 
    """
    provider: Provider
    model: str
    base_url: str
    custom_logger: 'Logger'
    debug: bool
    
    @abstractmethod
    async def completion(
        self, 
        messages: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]], 
        available_tools: Optional[list[dict[str, str]]] = None, 
        **kwargs: dict, 
    ) -> CompletionMessage:
        """Completion the messages.
        
        Args:
            messages (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]) :
                The messages to complete. 
            available_tools (list[dict[str, str]], optional) :
                The available tools.
            **kwargs (dict) :
                The additional keyword arguments.

        Returns:
            CompletionMessage:
                The completed message.
        """
        pass

    
class StreamLLM(LLM):
    """StreamLLM is a LLM that is used to stream the LLM calls.
    
    Attributes:
        provider (Provider) :
            The provider of the LLM.
        model (str) :
            The model of the LLM. 
        base_url (str) :
            The base URL of the LLM. 
        custom_logger (Logger):
            The custom logger. If not provided, the default loguru logger will be used. 
        debug (bool):
            The debug flag. 
    """
    provider: Provider
    model: str
    base_url: str
    custom_logger: 'Logger'
    debug: bool
    
    # Stream queue
    queue: Queue
    
    @abstractmethod
    async def stream(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        """Stream the LLM calls.
        
        Args:
            *args:
                The additional arguments to pass to the stream method.
            **kwargs:
                The additional keyword arguments to pass to the stream method.
        """
        pass
    
    

class Context(Protocol):
    """Context records the runtime information for global context. It is used to pass the information between 
    tool calling and the workflow. It can also be a general variable container for the life cycle of the workflow.  
    One context contains key-value pairs that set by the workflow, and the context visibility can be controlled 
    by layer of the context. 
    
    Attributes:
        prev (Context):
            The previous context.
        next (Context):
            The next context.
        key_values (dict[str, Any]):
            The key values of the context.
    """
    prev: Optional['Context']
    next: Optional['Context']
    key_values: dict[str, Any]
    
    @abstractmethod
    def append(self, key: str, value: Any) -> None:
        """Append a key-value pair to the context.
        
        Args:
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
        """
        pass
    
    @abstractmethod
    def update(self, key: str, value: Any) -> None:
        """Update the value of the key.
        
        Args:
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """Get the value of the key.
        
        Args:
            key (str):
                The key of the value.

        Returns:
            Any:
                The value of the key.
        """
        pass
    
    @abstractmethod
    def pop(self, key: str) -> Any:
        """Pop the value of the key.
        
        Args:
            key (str):
                The key of the value.
                
        Returns:
            Any:
                The value of the key.
        """
        pass
    
    @abstractmethod
    def create_next(self, **kwargs: dict) -> 'Context':
        """Create the next context.
        
        Args:
            **kwargs (dict):
                The keyword arguments to create the next context.
                
        Returns:
            Context:
                The next context.
        """
        pass
    
    @abstractmethod
    def done(self) -> 'Context':
        """Done the context and return the previous context.
        
        Returns:
            Context:
                The previous context.
        """
        pass


class TaskStatus(Enum):
    """Task status indicates the status of the task. This is not the final status of the task, but the 
    status of the task in the current work flow. If the answer is still None, the task is not done. 
    
    Attributes:
        CREATED (int):
            The task is created. This needs to be orchestrated by the workflow. 
        PLANNING (int):
            The task is planning. This is the status of the task in the current workflow. 
        RUNNING (int):
            The task is running. This is the status of the task in the current workflow.
        FINISHED (int):
            The task is finished. This means the task is finished and the answer is not None.
        FAILED (int):
            The task is failed. This means the task is failed.
        CANCELLED (int):
            The task is cancelled. This means the task is cancelled.
    """
    CREATED = 0
    PLANNING = 1
    RUNNING = 2
    FINISHED = 3
    FAILED = 4
    CANCELLED = 5
    
    
class TaskParallelStrategy(Enum):
    """TaskParallelStrategy controls the parallel strategy of the task.
    
    Attributes:
        SEQUENTIAL (str):
            The sub-tasks should be running sequentially. This is the default strategy.
        PARALLEL (str):
            The sub-tasks can be running in parallel.
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

    
@runtime_checkable
class Task(Protocol):
    """Task is the protocol for all the tasks.
    
    Attributes:
        uid (str): 
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
            
        question (str): 
            The question to be answered. 
        description (str):
            The detail information and limitation of the task. 
        detail_level (int):
            The detail level is the number of layers of sub-question layers that can be split from the question.
        parent (Task):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, Task]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
            
        status (TaskStatus):
            The status of the current task.
        parallel_strategy (TaskParallelStrategy):
            The parallel strategy of the current task. 
        is_leaf (bool):
            Whether the current task is a leaf task. If the task is a leaf task, the task will not be orchestrated by the workflow.
        answer (str):
            The answer to the question. If the task is not finished, the answer is None.
            
        history (dict[TaskStatus, list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the stateful object. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
    """
    uid: str
    
    # Context
    question: str
    description: str
    detail_level: int
    parent: 'Task'
    # NOTE: The key should be the question of the sub-task, the value should be the sub-task instance. 
    sub_tasks: OrderedDict[str, 'Task']
    
    # Status
    status: TaskStatus
    parallel_strategy: TaskParallelStrategy
    is_leaf: bool
    answer: Optional[str]
    
    history: dict[TaskStatus, list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]]
    
    @abstractmethod
    def update(
        self, 
        status: TaskStatus, 
        message: Union[CompletionMessage, ToolCallRequest, ToolCallResult], 
    ) -> None:
        """Update the task status.
        
        Args:
            status (TaskStatus):
                The status of the task.
            message (Union[CompletionMessage, ToolCallRequest, ToolCallResult]):
                The message to be updated.
        """
        pass


@runtime_checkable
class TaskView(Protocol):
    """TaskView is the view of the task. This view is used to format the task for the running task. 
    
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str
    
    @abstractmethod
    def format(self, *args, **kwargs) -> str:
        """Format the task view to a string. 
        
        Args:
            *args:
                The additional arguments to pass to the format method.
            **kwargs:
                The additional keyword arguments to pass to the format method.
        
        Returns: 
            str:
                The formatted task view. 
        """
        pass


@runtime_checkable
class Workflow(Protocol):
    """Workflow is stateless, it does not store any information about the state, it is only used to orchestrate the task or environment. 
    The workflow is not responsible for the state of the task or environment. 
    
    Attributes:
        system_prompt (str):
            The system prompt of the workflow. This is used to set the system prompt of the workflow. 
        agent (Agent):
            The agent. 
        debug (bool):
            The debug flag. 
        custom_logger (Logger):
            The custom logger. 
        context (Context):
            The context of the workflow.
        
        tools (dict[str, FastMCPTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the workflow. These functions can be used to control the workflow. 
        workflows (dict[str, Workflow]):
            The workflows that will be orchestrated to process the task.
    """
    system_prompt: str
    
    agent: 'Agent'
    debug: bool
    custom_logger: 'Logger'
    context: Context
    
    tools: dict[str, FastMcpTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, 'Workflow']
    
    @abstractmethod
    def post_init(self) -> None:
        """Post initialize the tools for the workflow.
        This method should be called after the initialization of the workflow. And you should register the tools in this method. 
        
        Example:
        ```python
        def post_init(self) -> None:
            
            @self.register_tool("tool_name")
            def tool_function(self, *args, **kwargs) -> Any:
                pass
        ```
        """
        pass
    
    @abstractmethod
    def add_tool(self, name: str, tool: Callable[[Any], Union[Task, 'Environment']]) -> None:
        """Add a tool to the workflow.
        
        Args:
            name (str):
                The name of the tool.
            tool (Callable[[Any], Union[Task, 'Environment']]):
                The tool to add. This tool should return a task or environment. 
        """
        pass
    
    @abstractmethod
    def register_tool(self, name: str) -> Callable:
        """This is a FastAPI like decorator to register a tool to the workflow.
        
        Args:
            name (str):
                The name of the tool.
                
        Returns:
            Callable:
                The function register.
        """
        pass

    @abstractmethod
    async def call_tool(
        self, 
        ctx: Union['Task', 'Environment'], 
        tool_call: ToolCallRequest, 
        **kwargs, 
    ) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            ctx (Union[Task, Environment]):
                The task or environment to call the tool.
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
    async def run(self, env: Union['Environment', Task]) -> Union['Environment', Task]:
        """Run the workflow from the environment or task.

        Args:
            env (Union[Environment, Task]): 
                The environment or task to run the workflow.

        Returns:
            Union[Environment, Task]: 
                The environment or task after running the workflow.
        """
        
        """ Observe the environment or task """
        # Your observation code here
        
        """ Append Environment or Task Prompt and Call for Completion """
        # Your prompt code here
        
        """ Check the stop reason """
        # Your stop reason code here
        
        """ If the stop reason is tool call, call the tool """
        # Your tool call code here
        
        """ If the stop reason is not tool call, return the environment or task """
        # Your return code here
        pass
    
    @abstractmethod
    def register_counter(self, counter: 'StepCounter') -> None:
        """Register a step counter to all the agents in the workflow.
        
        Args:
            counter (StepCounter):
                The step counter to register.
        """
        pass


class EnvironmentStatus(Enum):
    """EnvironmentStatus is the status of the environment.
    
    Attributes:
        CREATED (int):
            The environment is created. This means the environment is created.
        RUNNING (int):
            The environment is running. This is the status of the environment in the current workflow.
        FINISHED (int):
            The environment is finished. This means the environment is finished and the answers are not None.
        ERROR (int):
            Therer are some errors in the environment. This means the environment is not working properly.
    """
    CREATED = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3


class Environment(Workflow):
    """Environment is a stateful object that containing workflows. The workflows can be used to think about how to 
    modify the environment. The tools can be used to modify the environment. 
    
    Attributes:
        system_prompt (str):
            The system prompt of the environment. This is used to set the system prompt of the environment. 
            
        agent (Agent):
            The agent. 
        debug (bool):
            The debug flag. 
        custom_logger (Logger):
            The custom logger.
        context (Context):
            The context of the environment.
            
        tools (dict[str, FastMCPTool]):
            The tools that can be used to modify the environment. The key is the tool name and the value is the tool. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the environment. These functions can be used to modify the environment. 
        workflows (dict[str, Workflow]):
            The workflows that will be orchestrated to process the task.
        
        tasks (OrderedDict[str, Task]):
            The sub-tasks of the environment. The key is the sub-task name and the value is the sub-task.  
        answers (OrderedDict[str, str]):
            The answers of the tasks. The key is the task name and the value is the answer. 
        status (EnvironmentStatus):
            The status of the environment.
        history (list[Union[CompletionMessage, ToolCallResult, ToolCallRequest]]):
            The history state information of the environment. 
    """
    system_prompt: str
    
    agent: 'Agent'
    debug: bool
    custom_logger: 'Logger'
    context: Context
    
    tools: dict[str, FastMcpTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, 'Workflow']
    
    tasks: OrderedDict[str, Task]
    answers: OrderedDict[str, str] 
    status: EnvironmentStatus
    history: list[Union[CompletionMessage, ToolCallResult, ToolCallRequest]]
    
    @abstractmethod
    def post_init(self) -> None:
        """Post initialize the tools for the workflow.
        This method should be called after the initialization of the workflow. And you should register the tools in this method. 
        
        Example:
        ```python
        def post_init(self) -> None:
            
            @self.register_tool("tool_name")
            def tool_function(self, *args, **kwargs) -> Any:
                pass
        ```
        """
        pass
    
    @abstractmethod
    def add_tool(self, name: str, tool: Callable[[Any], Union[Task, 'Environment']]) -> None:
        """Add a tool to the workflow.
        
        Args:
            name (str):
                The name of the tool.
            tool (Callable[[Any], Union[Task, 'Environment']]):
                The tool to add. This tool should return a task or environment. 
        """
        pass
    
    @abstractmethod
    def register_tool(self, name: str) -> Callable:
        """This is a FastAPI like decorator to register a tool to the workflow.
        
        Args:
            name (str):
                The name of the tool.
                
        Returns:
            Callable:
                The function register.
        """
        pass

    @abstractmethod
    async def call_tool(
        self, 
        ctx: Union[Task, 'Environment'], 
        tool_call: ToolCallRequest, 
        **kwargs, 
    ) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            ctx (Union[Task, Environment]):
                The task or environment to call the tool.
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
    async def run(self, *args, **kwargs) -> Any:
        """Run the environment entrypoint. This should override the run method of the Environment. The `run` method of the 
        Environment can be called by `self.workflows["flow_name"].run(...)`.
        
        Args:
            *args:
                The additional arguments to pass to the run method of the Environment.
            **kwargs:
                The additional keyword arguments to pass to the run method of the Environment.
                
        Returns:
            Any:
                The result of the run method of the Environment.
        """
        pass
    
    @abstractmethod
    def update(
        self, 
        message: Union[CompletionMessage, ToolCallResult, ToolCallRequest],
    ) -> None:
        """Update the environment status.
        
        Args:
            message (Union[CompletionMessage, ToolCallResult, ToolCallRequest]):
                The message to update the environment.
        """
        pass


@runtime_checkable
class Logger(Protocol):
    """Logger is the protocol for the logger.
    """
    @abstractmethod
    def add(self, sink: str, format: str, level: str, colorize: bool, **kwargs) -> None:
        """Add a sink to the logger.
        
        Args:
            sink (str):
                The sink to add.
        """
    
    @abstractmethod
    def enable(self, name: str) -> None:
        """Enable the logger. This is used to enable the logger for a specific name.
        
        Args:
            name (str):
                The name of the logger.
        """
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """Debug the message. This is the lowest level of the logger.
        
        Args:
            message (str):
                The message to debug.
        """
        pass
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Info the message. This is the second lowest level of the logger.  
        
        This is the default level of the logger.
        
        Args:
            message (str):
                The message to info.
        """
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Warning the message. This is the third lowest level of the logger.  
        
        Args:
            message (str):
                The message to warning.
        """
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Error the message. This is the fourth lowest level of the logger.
        
        Args:
            message (str):
                The message to error.
        """
        pass
    
    @abstractmethod
    def critical(self, message: str) -> None:
        """Critical the message. This is the highest level of the logger.
        
        Args:
            message (str):
                The message to critical.
        """
        pass


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
    async def step(self, step: CompletionUsage) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (CompletionUsage):
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
    """Agent is a protocol for all the agents.
    
    Attributes:
        llm (LLM):
            The LLM to use for the agent. 
        debug (bool):
            The debug flag to use for the agent. 
        custom_logger (Logger):
            The custom logger to use for the agent.
            
        mcp_client (MCPClient):
            The MCP client to use for the agent.
        tools (list[dict[str, str]]):
            The tools can be used for the agent. 
            
        step_counters (dict[str, StepCounter]):
            The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
    """
    llm: LLM
    debug: bool
    custom_logger: Logger
    
    # Stateless tools
    mcp_client: MCPClient
    tools: list[dict[str, str]]
    
    # Max auto steps
    step_counters: dict[str, StepCounter]
    
    @abstractmethod
    async def observe(self, env: Union[Task, Environment], **kwargs) -> str:
        """Observe the task.
        
        Args:
            env (Union[Task, Environment]):
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
        observe: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]], 
        allow_tools: bool,  
        external_tools: dict[str, Union[FastMcpTool, MCPTool]] = {}, 
        **kwargs,
    ) -> CompletionMessage:
        """Think about the environment.
        
        Args:
            observe (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]):
                The messages observed from the environment. 
            allow_tools (bool):
                Whether to allow the tools provided by the agent to be used. This do not affect the 
                external tools provided by the workflow. 
            external_tools (Optional[dict[str, Union[FastMcpTool, MCPTool]]], defaults to {}):
                The external tools to use for the agent. 
            **kwargs:
                The additional keyword arguments for thinking about the environment. 
                
        Returns:
            CompletionMessage:
                The completion message thought about by the LLM. 
        """
        pass
    
    @abstractmethod
    async def call_tool(
        self, 
        ctx: Union[Task, Environment], 
        tool_call: ToolCallRequest, 
        **kwargs, 
    ) -> ToolCallResult:
        """Call a tool. 
        If there is any error caused by the tool call, the flag `is_error` will be set to True. 
        However, if there is any error caused by the MCP client connection, this should raise a RuntimeError.  
        
        Args:
            ctx (Union[Task, Environment]):
                The task or environment to call the tool.
            tool_call (ToolCallRequest): 
                The tool call request including the tool call id and the tool call arguments.
            **kwargs:
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult: 
                The result of the tool call. 
                
        Raises:
            RuntimeError:
                The runtime error raised by the MCP client connection. 
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


class GroupEnvironment(Environment):
    """GroupEnvironment containing multiple agents. 
    
    Attributes:
        agents (list[Agent]):
            The agents to use for the group environment.
    """
    agents: list[Agent]
    

class Worker(Protocol):
    """Worker is combined the Agent, the workflow and the environment. 
    
    Attributes:
        agent (Agent):
            The agent to use for the worker.
        workflow (Workflow):
            The workflow to execute.
        environment (Environment):
            The environment to execute.
    """
    agent: Agent
    workflow: Workflow
    environment: Environment
    
    def run(self, *args, **kwargs) -> Any:
        """Run the worker.
        
        Args:
            *args:
                The additional arguments to pass to the run method of the workflow.
            **kwargs:
                The additional keyword arguments to pass to the run method of the workflow.
        """
        pass
