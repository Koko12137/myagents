from abc import abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable, overload, Any, OrderedDict, Callable

from fastmcp import Client as MCPClient
from fastmcp.tools import Tool as FastMcpTool
from mcp import Tool as MCPTool
    
from myagents.src.message import CompletionMessage, ToolCallRequest, ToolCallResult


class Provider(Enum):
    OPENAI = "openai"
    TONGYI = "tongyi"


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
        custom_logger (Logger, defaults to logger):
            The custom logger. If not provided, the default loguru logger will be used. 
        debug (bool, defaults to False):
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
        messages: list[CompletionMessage | ToolCallRequest | ToolCallResult], 
        available_tools: list[dict[str, str]] | None = None, 
        **kwargs: dict, 
    ) -> CompletionMessage:
        """Completion the messages.
        
        Args:
            messages (list[Message | ToolCallRequest | ToolCallResult]) :
                The messages to complete. 
            available_tools (list[dict[str, str]] | None) :
                The available tools.
            **kwargs (dict) :
                The additional keyword arguments.

        Returns:
            Message:
                The completed message.
        """
        pass


class MaxStepsError(Exception):
    """MaxStepsError is an exception for the max steps error.
    """
    def __init__(self, message: str, current: int | float, limit: int | float) -> None:
        self.message = message
        self.current = current
        self.limit = limit
        super().__init__(self.message)
        
    def __str__(self) -> str:
        return f"MaxStepsError: {self.message}, current: {self.current}, limit: {self.limit}"


class StepCounter(Protocol):
    """StepCounter is a protocol for the step counter. The limit can be max auto steps or max balance cost. It is better to use 
    the same step counter for all agents. 
    
    Attributes:
        limit (int | float):
            The limit of the step counter. 
        current (int | float):
            The current step of the step counter. 
    """
    limit: int | float
    current: int | float
    
    def reset(self) -> None:
        """Reset the current step of the step counter.
        
        Returns:
            None
        """
        pass
    
    def update_limit(self, limit: int | float) -> None:
        """Update the limit of the step counter.
        
        Args:
            limit (int | float):
                The limit of the step counter. 
        """
        pass
    
    def step(self, *args, **kwargs) -> None:
        """Increment the current step of the step counter.
        
        Args:
            *args:
                The additional arguments to pass to the step method.
            **kwargs:
                The additional keyword arguments to pass to the step method.
        
        Returns:
            None 
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
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
            
        step_counter (StepCounter):
            The step counter to use for the agent. 
    """
    llm: LLM
    debug: bool
    custom_logger: 'Logger'
    
    # Stateless tools
    mcp_client: MCPClient
    tools: list[dict[str, str]]
    
    # Max auto steps
    step_counter: StepCounter
    
    @overload
    async def observe(
        self, 
        env: 'Environment',  
        **kwargs: dict, 
    ) -> tuple[list[CompletionMessage | ToolCallRequest | ToolCallResult], str]:
        """Observe the environment.
        
        Args:
            env (Environment):
                The environment to observe. 
            **kwargs (dict, optional):
                The keyword arguments to pass to the environment observe method. 

        Returns:
            list[CompletionMessage | ToolCallRequest | ToolCallResult]:
                The observed history messages of the environment. 
            str:
                The up to date information observed from the environment.  
        """
        pass

    @overload
    async def observe(
        self, 
        env: 'Task', 
        **kwargs: dict, 
    ) -> tuple[list[CompletionMessage | ToolCallRequest | ToolCallResult], str]:
        """Observe the task.
        
        Args:
            env (Task):
                The task to observe. 
            **kwargs (dict, optional):
                The keyword arguments to pass to the task observe method.

        Returns:
            list[CompletionMessage | ToolCallRequest | ToolCallResult]:
                The observed history messages of the task from the root task to the current task. 
            str:
                The up to date information observed from the task.  
        """
        pass
    
    async def think(
        self, 
        observe: list[CompletionMessage | ToolCallRequest | ToolCallResult], 
        allow_tools: bool,  
        external_tools: dict[str, FastMcpTool | MCPTool] = {}, 
        **kwargs: dict, 
    ) -> CompletionMessage:
        """Think about the environment.
        
        Args:
            observe (list[CompletionMessage | ToolCallRequest | ToolCallResult]):
                The messages observed from the environment. 
            allow_tools (bool):
                Whether to allow tools to be used.  
            external_tools (dict[str, FastMcpTool | MCPTool], optional):
                The external tools to use for the agent. 
            **kwargs (dict, optional):
                The additional keyword arguments for thinking about the observed messages. 
                
        Returns:
            CompletionMessage:
                The completion message thought about by the LLM. 
        """
        pass
    
    async def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool. 
        If there is any error caused by the tool call, the flag `is_error` will be set to True. 
        However, if there is any error caused by the MCP client connection, this should raise a RuntimeError.  
        
        Args:
            tool_call (ToolCallRequest): 
                The tool call request including the tool call id and the tool call arguments.

        Returns:
            ToolCallResult: 
                The result of the tool call. 
                
        Raises:
            RuntimeError:
                The runtime error raised by the MCP client connection. 
        """
        pass


@runtime_checkable
class Environment(Protocol):
    """Environment is a protocol for all the environments.
    
    Attributes:
        tools (dict[str, FastMcpTool]):
            The tools that can be used to modify the environment. The key is the tool name and the value is the tool. 
        history (list[CompletionMessage | ToolCallResult | ToolCallRequest]):
            The history of the environment. 
    """
    tools: dict[str, FastMcpTool]
    history: list[CompletionMessage | ToolCallResult | ToolCallRequest]
    
    @abstractmethod
    def observe(self, *args, **kwargs) -> str:
        """Observe the environment.

        Returns:
            str: 
                The observed information of the environment.
        """
        pass

    @abstractmethod
    def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool to modify the environment.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
                
        Returns:
            ToolCallResult:
                The tool call result.
        """
        pass


class RunnableEnvironment(Environment):
    """RunnableEnvironment is a protocol for all the runnable environments.
    
    Attributes:
        history (list[CompletionMessage | ToolCallResult | ToolCallRequest]):
            The history of the environment. 
            
        agent (Agent):
            The agent. 
        debug (bool, defaults to False):
            The debug flag. 
        custom_logger (Logger, defaults to None):
            The custom logger.
        tools (dict[str, FastMCPTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the workflow. These functions can be used to control the workflow or the environment. 
        workflows (dict[str, Workflow]):
            The workflows that will be orchestrated to process the task.
    """
    
    agent: Agent
    debug: bool
    custom_logger: 'Logger'
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
    async def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
                
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
        """Run the environment entrypoint. This should override the run method of the OrchestratedFlows. The `run` method of the 
        OrchestratedFlows can be called by `self.workflows["flow_name"].run(...)`.
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
        FAILED (str):
            The task is failed. This means the task is failed.
        CANCELLED (str):
            The task is cancelled. This means the task is cancelled.
    """
    CREATED = 0
    PLANNING = 1
    RUNNING = 2
    FINISHED = 3
    FAILED = 4
    CANCELLED = 5
    

class TaskStrategy(Enum):
    """Task strategy indicates the completion condition of the task. 

    Attributes:
        ALL (str):
            The task is completed if all the sub-tasks are finished.
        ANY (str):
            The task is completed if any of the sub-tasks are finished.
    """
    ALL = "all"
    ANY = "any"


@runtime_checkable
class Task(Protocol):
    """Task is the protocol for all the tasks.
    
    Attributes:
        uid (str): 
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
            
        question (str): 
            The question to be answered. 
        description (str):
            The description of the task. 
        parent (Task | None):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, Task]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
            
        status (TaskStatus):
            The status of the current task.
        strategy (TaskStrategy):
            The strategy of the current task. 
        is_leaf (bool):
            Whether the current task is a leaf task. If the task is a leaf task, the task will not be orchestrated by the workflow.
        answer (str | None):
            The answer to the question. If the task is not finished, the answer is None.
            
        history (list[CompletionMessage | ToolCallRequest | ToolCallResult]):
            The history messages of the current task. 
    """
    uid: str
    
    # Context
    question: str
    description: str
    parent: 'Task'
    sub_tasks: OrderedDict[str, 'Task']
    
    # Status
    status: TaskStatus
    is_leaf: bool
    strategy: TaskStrategy
    answer: str
    
    # History Messages
    history: list[CompletionMessage | ToolCallRequest | ToolCallResult]
    
    # Observe the task
    def observe(self, *args, **kwargs) -> str:
        """Observe the task according to the current status.
        
        - CREATED:
            This task needs to be orchestrated by the workflow. 
            Question, parent information, dependencies information, and sub-tasks information are needed.
        - PLANNING:
            This task is planning. 
            Question, parent information, dependencies information, and sub-tasks information are needed.
        - RUNNING:
            This task is running. 
            Question, parent information, dependencies information, and sub-tasks information are needed.
        - FINISHED:
            This task is finished. 
            Question, parent information, dependencies information, and sub-tasks information are needed.
        - FAILED:
            This task is failed. Both question, status and error message are needed.
        - CANCELLED:
            This task is cancelled. Both question and status are needed. 

        Returns:
            str: 
                The observed information of the task.
        """
        pass


@runtime_checkable
class TaskView(Protocol):
    """TaskView is the view of the task. This view is used to format the task for the running task. 
    """
    model: Task
    
    def format(self, **kwargs) -> str:
        """Format the task view to a string.
        
        Args:
            **kwargs:
                The additional arguments to be formatted.
                
        Returns: 
            str:
                The formatted task view. 
        """


@runtime_checkable
class Workflow(Protocol):
    """Workflow is the protocol for all the workflows.
    
    Attributes:
        agent (Agent):
            The agent. 
        debug (bool, defaults to False):
            The debug flag. 
        custom_logger (Logger, defaults to None):
            The custom logger. 
        tools (dict[str, FastMCPTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the workflow. These functions can be used to control the workflow. 
    """
    agent: Agent
    debug: bool
    custom_logger: 'Logger'
    tools: dict[str, FastMcpTool]
    tool_functions: dict[str, Callable]
    
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
    async def run(self, env: Environment | Task) -> Environment | Task:
        """Run the workflow from the environment or task.

        Args:
            env (Environment | Task): 
                The environment or task to run the workflow.

        Returns:
            Environment | Task: 
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
    async def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
                
        Returns:
            ToolCallResult:
                The tool call result. 
                
        Raises:
            ValueError:
                If the tool call name is not registered. 
        """
        pass
    

class OrchestratedFlows(Workflow):
    """OrchestratedFlows is the protocol for all the orchestrated flows. Multiple flows can be orchestrated to process the task. 
    
    Attributes:
        agent (Agent):
            The agent. 
        debug (bool, defaults to False):
            The debug flag. 
        custom_logger (Logger, defaults to None):
            The custom logger.
        tools (dict[str, FastMCPTool]):
            The tools provided by the workflow. These tools can be used to control the workflow. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the workflow. These functions can be used to control the workflow.  
        workflows (dict[str, Workflow]):
            The workflows that will be orchestrated to process the task.
    """
    workflows: dict[str, Workflow]


@runtime_checkable
class Logger(Protocol):
    """Logger is the protocol for the logger.
    """
    def add(self, sink: str, format: str, level: str, colorize: bool, **kwargs) -> None:
        """Add a sink to the logger.
        
        Args:
            sink (str):
                The sink to add.
        """
    
    def enable(self, name: str) -> None:
        """Enable the logger. This is used to enable the logger for a specific name.
        
        Args:
            name (str):
                The name of the logger.
        """
        pass
    
    def debug(self, message: str) -> None:
        """Debug the message. This is the lowest level of the logger.
        
        Args:
            message (str):
                The message to debug.
        """
        pass
    
    def info(self, message: str) -> None:
        """Info the message. This is the second lowest level of the logger.  
        
        This is the default level of the logger.
        
        Args:
            message (str):
                The message to info.
        """
        pass
    
    def warning(self, message: str) -> None:
        """Warning the message. This is the third lowest level of the logger.  
        
        Args:
            message (str):
                The message to warning.
        """
        pass
    
    def error(self, message: str) -> None:
        """Error the message. This is the fourth lowest level of the logger.
        
        Args:
            message (str):
                The message to error.
        """
        pass
    
    def critical(self, message: str) -> None:
        """Critical the message. This is the highest level of the logger.
        
        Args:
            message (str):
                The message to critical.
        """
        pass
