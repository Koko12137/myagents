from abc import abstractmethod
from typing import Protocol, runtime_checkable, Optional, Union
from enum import Enum
from collections import OrderedDict

from myagents.core.messages import ToolCallRequest, ToolCallResult, AssistantMessage, UserMessage, SystemMessage


class TaskStatus(Enum):
    """Task status indicates the status of the task. This is not the final status of the task, but the 
    status of the task in the current work flow. If the answer is still None, the task is not done. 
    
    Attributes:
        CREATED (str):
            ` - [ ] ` This needs to be orchestrated by the workflow. 
        PLANNING (str):
            ` - [p] ` This Task is creating sub-tasks. If the planning is finished, the task will be set to `- [c]` checking status.
        CHECKING (str):
            ` - [c] ` This Task needs to be checked by the workflow or human. If the checking is finished, the task will be set to `- [r]` running status.
        RUNNING (str):
            ` - [r] ` This Task is running the sub-tasks. If the running is finished, the task will be set to `- [x]` finished status. 
            If there is any error during the running, the task will be set to `- [e]` error status directly. 
        FINISHED (str):
            ` - [x] ` This means the task is finished. But the answer may be None. So it needs double check. 
        ERROR (str):
            ` - [e] ` This Task is error but could be recovered. If the error is not recovered, the task will be set to `- [k]` cancelled status.
        CANCELLED (str):
            ` - [k] ` This Task is cancelled and could not be recovered. This will cause an system error. 
    """
    CREATED     = " - [ ] "
    PLANNING    = " - [p] "
    CHECKING    = " - [c] "
    RUNNING     = " - [r] "
    FINISHED    = " - [x] "
    ERROR       = " - [e] "
    CANCELLED   = " - [k] "
    
    
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
            
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]):
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
    
    history: dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]
    
    @abstractmethod
    def update(
        self, 
        status: TaskStatus, 
        message: Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult], 
    ) -> None:
        """Update the task status.
        
        Args:
            status (TaskStatus):
                The status of the task.
            message (Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]):
                The message to be updated.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the task.
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
