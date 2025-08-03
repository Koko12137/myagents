from abc import abstractmethod
from enum import Enum
from collections import OrderedDict
from typing import runtime_checkable, Protocol

from myagents.core.interface.base import Stateful
from myagents.core.interface.memory import VectorMemoryCollection, TableMemoryDB


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
        name (str):
            The name of the task. 
        objective (str):
            The objective of the task.
        key_results (str):
            The key results of the task and the verification method for the results.
        results (str):
            The results of the task. If the task is not finished, the results is None.
        next (Task):
            The next task of the current task. If the task does not have a next task, the next is None.
        prev (Task):
            The previous task of the current task. If the task does not have a previous task, the prev is None.
    """
    uid: str
    name: str
    objective: str
    key_results: str
    results: str
    # Link information
    next: 'Task'
    prev: 'Task'


class TreeTaskNode(Task):
    """TreeTaskNode is the protocol for all the tasks. It is a tree structure of the tasks.
    
    Attributes:
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
    

class MemoryTreeTaskNode(TreeTaskNode):
    """MemoryTreeTaskNode is the protocol for all the tasks. It is a tree structure of the tasks.
    
    Attributes:
        semantic_memory (VectorMemory):
            The semantic memory of the task.
        episodic_memory (VectorMemory):
            The episodic memory of the task.
        procedural_memory (TableMemory):
            The procedural memory of the task.
        trajectory_memory (TableMemory):
            The trajectory memory of the task.
    """
    semantic_memory: VectorMemoryCollection
    episodic_memory: VectorMemoryCollection
    procedural_memory: VectorMemoryCollection
    trajectory_memory: TableMemoryDB
    
    @abstractmethod
    async def add(
        self, 
        text: str, 
        **kwargs,
    ) -> None:
        """Add the text to the memory.
        
        Args:
            text (str): The text to add to the memory.
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        text: str, 
        **kwargs,
    ) -> list[str]:
        """Search the text from the memory.
        
        Args:
        """
        pass
    
    @abstractmethod
    async def update(
        self, 
        text: str, 
        **kwargs,
    ) -> None:
        """Update the text in the memory.
        
        Args:
        """
        pass
    
    @abstractmethod
    async def delete(
        self, 
        text: str, 
        **kwargs,
    ) -> None:
        """Delete the text from the memory.
        
        Args:
        """
        pass
