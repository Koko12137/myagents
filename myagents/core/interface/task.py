from abc import abstractmethod
from typing import Protocol, runtime_checkable, Optional, Union
from enum import Enum
from collections import OrderedDict

from myagents.core.interface.core import Stateful
from myagents.core.messages import ToolCallRequest, ToolCallResult, AssistantMessage, UserMessage, SystemMessage


class TaskStatus(Enum):
    """Task status indicates the status of the task. This is not the final status of the task, but the 
    status of the task in the current work flow. If the answer is still None, the task is not done. 
    
    Attributes:
        CREATED (str):
            ` - [ ] ` This needs to be orchestrated by the workflow. 
        PLANNING (str):
            ` - [p] ` This Task is creating sub-tasks. If the planning is finished, the task will be set to `- [c]` checking status.
        RUNNING (str):
            ` - [r] ` This Task is running the sub-tasks. If the running is finished, the task will be set to `- [x]` finished status. 
            If there is any error during the running, the task will be set to `- [e]` error status directly. 
        FINISHED (str):
            ` - [x] ` This means the task is finished. But the answer may be None. So it needs double check. 
        ERROR (str):
            ` - [e] ` This Task is error but could be recovered. If the error is not recovered, the task will be set to `- [k]` cancelled status.
        CANCELLED (str):
            ` - [c] ` This Task is cancelled and could not be recovered. This will cause an system error. 
    """
    CREATED     = " - [ ] "
    RUNNING     = " - [r] "
    FINISHED    = " - [x] "
    ERROR       = " - [e] "
    CANCELLED   = " - [c] "


class Task(Stateful):
    """Task is the protocol for all the tasks. It is a general task that can be used for the workflow.
    
    Attributes:
        uid (str):
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
        question (str):
            The question to be answered. 
        description (str):
            The detail information and limitation of the task. 
        status (TaskStatus):
            The status of the current task.
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the stateful object. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
    """
    uid: str
    question: str
    description: str
    # Status and history
    status: TaskStatus
    history: dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]


class TreeTaskNode(Task):
    """TreeTaskNode is the protocol for all the tasks. It is a tree structure of the tasks.
    
    Attributes:
        uid (str): 
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
        question (str): 
            The question to be answered. 
        description (str):
            The detail information and limitation of the task. 
        status (TaskStatus):
            The status of the current task.
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the stateful object. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
        parent (TreeTaskNode):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, TreeTaskNode]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
        sub_task_depth (int):
            The sub task depth is the number of layers of sub-question layers that can be split from the question.
        answer (str):
            The answer to the question. If the task is not finished, the answer is None
    """
    uid: str
    question: str
    description: str
    # Status and history
    status: TaskStatus
    history: dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]
    # Parent and sub-tasks
    parent: 'TreeTaskNode'
    # NOTE: The key should be the question of the sub-task, the value should be the sub-task instance. 
    sub_tasks: OrderedDict[str, 'TreeTaskNode']
    sub_task_depth: int
    answer: str
    
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
        uid (str):
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
        question (str):
            The question to be answered. 
        description (str):
            The detail information and limitation of the task. 
        dependencies (OrderedDict[str, GraphTaskNode]):
            The dependencies of the task. The key is the unique identifier of the dependency task, and the value is the dependency task.
        answer (str):
            The answer to the question. If the task is not finished, the answer is None.
        status (TaskStatus):
            The status of the current task.
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the stateful object. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
    """
    uid: str
    question: str
    description: str
    # Status and history
    status: TaskStatus
    history: dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]
    # Dependencies and answer
    dependencies: OrderedDict[str, 'GraphTaskNode']
    answer: Optional[str]


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
