from abc import abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable, Union, Optional, Any, Callable, Awaitable, OrderedDict

from mcp import Tool as MCPTool
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages.message import AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult


@runtime_checkable
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
                
        Raises:
            KeyError:
                If the key is not found.
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
                
        Raises:
            KeyError:
                If the key is not found.
        """
        pass
    
    @abstractmethod
    def create_next(self, **kwargs) -> 'Context':
        """Create the next context.
        
        Args:
            **kwargs:
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


@runtime_checkable
class Status(Protocol):
    """Status is a protocol for the class that maintaining several statuses, and it can be observed
    according to the current status and the format of the observation.
    
    Attributes:
        CREATED (Union[str, int]): The created status.
        RUNNING (Union[str, int]): The running status.
        FINISHED (Union[str, int]): The finished status.
        ERROR (Union[str, int]): The error status.
        CANCELLED (Union[str, int]): The cancelled status.
    """
    CREATED: Union[str, int]
    RUNNING: Union[str, int]
    FINISHED: Union[str, int]
    ERROR: Union[str, int]
    CANCELLED: Union[str, int]


@runtime_checkable
class Stateful(Protocol):
    """Stateful is a protocol for the class that maintaining several statuses, and it can be observed
    according to the current status and the format of the observation.
    
    Attributes:
        status (Status):
            The status of the stateful entity.
        history (dict[Status, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]]):
            The history of the stateful entity.
    """
    status: Status
    history: dict[Status, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]]
    
    @abstractmethod
    async def observe(self, format: str, **kwargs) -> str:
        """Observe the state of the stateful entity according to the current status and the 
        format of the observation.
        
        Args:
            format (str):
                The format of the observation.
            **kwargs:
                The additional keyword arguments for the observation.
                
        Returns:
            str:
                The observation of the stateful entity.
        """
        pass

    @abstractmethod
    def update(
        self, 
        message: Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult],
    ) -> None:
        """Update the history of the stateful entity according to the current status.
        
        Args:
            message (Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]):
                The message to be updated.
        """
        pass
    
    @abstractmethod
    def get_history(self) -> list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]:
        """Get the history of the stateful entity according to the current status.
        
        Returns:
            list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]:
                The history of the stateful entity according to the current status.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the history for all the statuses and set the current status to created.
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Enum:
        """Get the current status of the stateful entity. 
        
        Returns:
            Enum:
                The current status of the stateful entity.
        """
        pass
    
    @abstractmethod
    def to_created(self) -> None:
        """Set the current status of the stateful entity to created.
        """
        pass
    
    @abstractmethod
    def is_created(self) -> bool:
        """Check if the current status of the stateful entity is created.
        """
        pass
    
    @abstractmethod
    def to_running(self) -> None:
        """Set the current status of the stateful entity to running.
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the current status of the stateful entity is running.
        """
        pass
    
    @abstractmethod
    def to_finished(self) -> None:
        """Set the current status of the stateful entity to finished.
        """
        pass
    
    @abstractmethod
    def is_finished(self) -> bool:
        """Check if the current status of the stateful entity is finished.
        """
        pass
    
    @abstractmethod
    def to_error(self) -> None:
        """Set the current status of the stateful entity to error.
        """
        pass
    
    @abstractmethod
    def is_error(self) -> bool:
        """Check if the current status of the stateful entity is error.
        """
        pass
    
    @abstractmethod
    def to_cancelled(self) -> None:
        """Set the current status of the stateful entity to cancelled.
        """
        pass
    
    @abstractmethod
    def is_cancelled(self) -> bool:
        """Check if the current status of the stateful entity is cancelled.
        """
        pass


@runtime_checkable
class ToolsCaller(Protocol):
    """ToolsCaller is a protocol for the tools caller. It is used to call the tools.
    
    Attributes:
        tools (dict[str, Union[FastMcpTool, MCPTool]]):
            The tools of the caller.
        context (Context):
            The context of the caller.
    """
    # Tools and global context container
    tools: dict[str, Union[FastMcpTool, MCPTool]]
    context: Context
    
    @abstractmethod
    def post_init(self) -> None:
        """Post init the tools caller.
        
        Example:
        ```python
        async def post_init(self) -> None:
            # Register a tool to the caller in the post init method. 
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
        tool: Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]], 
        tags: list[str] = [], 
        replace: bool = True,
    ) -> None:
        """Add a tool to the caller.
        
        Args:
            name (str):
                The name of the tool.
            tool (Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]):
                The tool to add. This tool should return the tool call result. 
            tags (list[str], optional):
                The tags of the tool.
            replace (bool, optional, defaults to True):
                Whether to replace the tool if it is already registered.
                
        Returns:
            Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]:
                The tool to register.
        """
        pass
    
    @abstractmethod
    def register_tool(
        self, 
        name: str, 
        tags: list[str] = [], 
        replace: bool = True,
    ) -> Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]:
        """Register a tool to the caller.
        
        Args:
            name (str):
                The name of the tool.
            tags (list[str], optional):
                The tags of the tool.
            replace (bool, optional, defaults to True):
                Whether to replace the tool if it is already registered.
                
        Returns:
            Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]:
                The tool to register.
        """
        pass

    @abstractmethod
    async def call_tool(self, tool_call: ToolCallRequest, **kwargs) -> ToolCallResult:
        """Call a tool.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
            **kwargs:
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult:
                The tool call result.
        """
        pass


class TaskStatus(Enum):
    """Task status indicates the status of the task. This is not the final status of the task, but the 
    status of the task in the current work flow. If the answer is still None, the task is not done. 
    
    Attributes:
        CREATED (str):
            `[[ CREATED ]]` This needs to be orchestrated by the workflow. 
        PLANNING (str):
            `[[ PLANNING ]]` This Task is creating sub-tasks. If the planning is finished, the task will be set to `- [c]` checking status.
        RUNNING (str):
            `[[ RUNNING ]]` This Task is running the sub-tasks. If the running is finished, the task will be set to `[[ FINISHED ]]` finished status. 
            If there is any error during the running, the task will be set to `[[ ERROR ]]` error status directly. 
        FINISHED (str):
            `[[ FINISHED ]]` This means the task is finished. But the answer may be None. So it needs double check. 
        ERROR (str):
            `[[ ERROR ]]` This Task is error but could be recovered. If the error is not recovered, the task will be set to `- [k]` cancelled status.
        CANCELED (str):
            `[[ CANCELED ]]` This Task is cancelled and could not be recovered. This will cause an system error. 
    """
    CREATED     = "[ CREATED ]"
    RUNNING     = "[ RUNNING ]"
    FINISHED    = "[ FINISHED ]"
    ERROR       = "[ ERROR ]"
    CANCELED    = "[ CANCELED ]"


class Task(Stateful):
    """Task is the protocol for all the tasks. It is a general task that can be used for the workflow.
    
    Attributes:
        status (TaskStatus):
            The status of the current task.
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the stateful object. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
            
        uid (str):
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
        objective (str):
            The objective of the task.
        key_results (str):
            The key results of the task and the verification method for the results.
        results (str):
            The results of the task. If the task is not finished, the results is None.
    """
    uid: str
    objective: str
    key_results: str
    results: str


class TreeTaskNode(Task):
    """TreeTaskNode is the protocol for all the tasks. It is a tree structure of the tasks.
    
    Attributes:
        status (TaskStatus):
            The status of the current task.
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the stateful object. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
        
        uid (str): 
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
        objective (str):
            The objective of the task.
        key_results (str):
            The key results of the task and the verification method for the results.
        results (str):
            The results of the task. If the task is not finished, the results is None
            
        parent (TreeTaskNode):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, TreeTaskNode]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
        sub_task_depth (int):
            The sub task depth is the number of layers of sub-objective layers that can be split from the objective.
    """
    # Parent and sub-tasks
    parent: 'TreeTaskNode'
    # NOTE: The key should be the objective of the sub-task, the value should be the sub-task instance. 
    sub_tasks: OrderedDict[str, 'TreeTaskNode']
    sub_task_depth: int
    
    @abstractmethod
    def to_created(self) -> None:
        """Set the current status of the stateful entity to created. This will also set the sub-tasks to cancelled if the 
        sub-tasks are not finished.
        """
        pass
    
    @abstractmethod
    def to_running(self) -> None:
        """Set the current status of the stateful entity to running. This will also set the parent task to running if the 
        parent task is created and all the sub-tasks are running.
        """
        pass
    
    @abstractmethod
    def to_finished(self) -> None:
        """Set the current status of the stateful entity to finished. This will also set the parent task to finished if the 
        parent task is running and all the sub-tasks are finished.
        """
        pass


class GraphTaskNode(Task):
    """GraphTaskNode is the protocol for all the tasks. It is a graph structure of the tasks.
    
    Attributes:
        status (TaskStatus):
            The status of the current task.
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the stateful object. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
        
        uid (str):
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
        objective (str):
            The objective of the task.
        key_results (str):
            The key results of the task and the verification method for the results.
        results (str):
            The results of the task. If the task is not finished, the results is None.
        
        dependencies (OrderedDict[str, GraphTaskNode]):
            The dependencies of the task. The key is the unique identifier of the dependency task, and the value is the dependency task.
    """
    # Dependencies
    dependencies: OrderedDict[str, 'GraphTaskNode']


@runtime_checkable
class TaskView(Protocol):
    """TaskView defines the format protocol for the task. 
    
    Attributes:
        task (Task):
            The task to be viewed.
    """
    task: Task
    
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
