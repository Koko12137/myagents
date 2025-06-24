import re
from uuid import uuid4
from collections import OrderedDict
from typing import Union

from pydantic import BaseModel, Field, ConfigDict

from myagents.src.message import CompletionMessage, ToolCallRequest, ToolCallResult
from myagents.src.interface import TaskStatus, TaskStrategy, TaskView, Task


class BaseTask(BaseModel):
    """BaseTask is the base class for all the tasks.
    
    Attributes:
        uid (str): 
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
            
        question (str): 
            The question to be answered. 
        description (str):
            The description of the task. 
        parent (Task, optional):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, Task]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
            
        status (TaskStatus):
            The status of the current task.
        strategy (TaskStrategy):
            The strategy of the current task.
        is_leaf (bool):
            Whether the current task is a leaf task. If the task is a leaf task, the task will not be orchestrated by the workflow. 
        answer (str, optional):
            The answer to the question. If the task is not finished, the answer is None.
            
        history (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]):
            The history messages of the current task. 
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    uid: str = Field(default_factory=lambda: uuid4().hex)
    
    # Context
    question: str = Field(description="The question to be answered.")
    description: str = Field(
        description="The description of the task. In this task, whether or not you need a utility, and what information you want to get if you do."
    )
    parent: 'Task' = Field(description="The parent task of the current task.", default=None)
    sub_tasks: OrderedDict[str, 'Task'] = Field(description="The sub-tasks of the current task.", default={})
    
    # Status
    status: TaskStatus = Field(description="The status of the current task.", default=TaskStatus.CREATED)
    strategy: TaskStrategy = Field(description="The strategy of the current task.", default=TaskStrategy.ALL)
    is_leaf: bool = Field(description="Whether the current task is a leaf task.", default=False) 
    answer: str = Field(description="The answer to the question.", default=None)
    
    # History Messages
    history: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]] = Field(
        default_factory=list,
        description="The messages of the current task.", 
    )


class TaskContextView:
    """TaskContextView is the view of the task context. This view is used to format the task context to a string.
    
    Attributes:
        model (Task):
            The task to be viewed.
    """
    model: Task
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self, layer: int = 3) -> str:
        """Format the task context to a string.
        
        Args:
            layer (int):
                The recursive layer of the task. 
        
        Returns:
            str:
                The formatted task context.
        """
        sub_tasks = []
        
        # Check if the status is not finished, failed, or cancelled
        if self.model.status == TaskStatus.CREATED:
            # Return a created task view
            view = CreatedTaskView(self.model).format()
        elif self.model.status == TaskStatus.PLANNING:
            # Return a planning task view
            view = PlanningTaskView(self.model).format()
            # Process the sub-tasks
            for sub_task in self.model.sub_tasks.values():
                sub_task_info = TaskContextView(sub_task).format(layer=layer-1)
                # Update the indentation 
                sub_task_info = re.sub(r'^', f"\t" * (layer-1), sub_task_info, flags=re.MULTILINE)
                sub_tasks.append(sub_task_info)
        elif self.model.status == TaskStatus.RUNNING:
            # Return a running task view
            view = RunningTaskView(self.model).format()
            # Process the sub-tasks
            for sub_task in self.model.sub_tasks.values():
                sub_task_info = TaskContextView(sub_task).format(layer=layer-1)
                # Update the indentation 
                sub_task_info = re.sub(r'^', f"\t" * (layer-1), sub_task_info, flags=re.MULTILINE)
                sub_tasks.append(sub_task_info)
        elif self.model.status == TaskStatus.FINISHED:
            # Return a finished task view
            view = FinishedTaskView(self.model).format()
        elif self.model.status == TaskStatus.FAILED:
            # Return a failed task view
            view = FailedTaskView(self.model).format()
        elif self.model.status == TaskStatus.CANCELLED:
            # Return a cancelled task view
            view = CancelledTaskView(self.model).format()
            
        # Check if the sub-tasks is not empty
        if len(sub_tasks) > 0:
            sub_tasks_str = '\n'.join(sub_tasks)
            sub_tasks = f"\t - Sub-Tasks: \n{sub_tasks_str}"
        else:
            sub_tasks = f"\t - Sub-Tasks: \n\t\t No sub-tasks now"
        view = f"{view}\n{sub_tasks}"
            
        return view
    

class CreatedTaskView(TaskView):
    """CreatedTaskView is the view of the created task. This view is used to format the created task to a string.
    This view is used to show the task information when the task is created. 
    
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str = """ - [ ] {question}\n\t - Description: {description}\n\t - Status: {status}\n\t - Is Leaf: {is_leaf}\n\t - Strategy: {strategy}"""
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self) -> str:
        return self.template.format(
            question=self.model.question, 
            description=self.model.description, 
            status=self.model.status, 
            is_leaf=self.model.is_leaf, 
            strategy=self.model.strategy, 
        )


class PlanningTaskView(TaskView):
    """PlanningTaskView is the view of the planning task. This view is used to format the planning task to a string.
    This view is used to show the task information when the task is planning. 
    
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str = """ - [p] {question}\n\t - Description: {description}\n\t - Status: {status}\n\t - Is Leaf: {is_leaf}\n\t - Strategy: {strategy}"""
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self) -> str:
        return self.template.format(
            question=self.model.question, 
            description=self.model.description, 
            status=self.model.status, 
            is_leaf=self.model.is_leaf, 
            strategy=self.model.strategy, 
        )
    
    
class RunningTaskView(TaskView):
    """RunningTaskView is the view of the running task. This view is used to format the running task to a string.
    This view is used to show the task information when the task is running. 
    
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str = """ - [r] {question}\n\t - Description: {description}\n\t - Status: {status}\n\t - Is Leaf: {is_leaf}\n\t - Strategy: {strategy}"""
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self) -> str:
        return self.template.format(
            question=self.model.question, 
            description=self.model.description, 
            status=self.model.status, 
            is_leaf=self.model.is_leaf, 
            strategy=self.model.strategy, 
        )


class FinishedTaskView(TaskView):
    """FinishedTaskView is the view of the finished task. This view is used to format the finished task to a string.
    This view is used to show the task information when the task is finished. 
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str = """ - [x] {question}\n\t - Description: {description}\n\t - Status: {status}\n\t - Is Leaf: {is_leaf}\n\t - Strategy: {strategy}\n\t - Answer: {answer}"""
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self) -> str:
        return self.template.format(
            question=self.model.question, 
            description=self.model.description, 
            status=self.model.status, 
            is_leaf=self.model.is_leaf, 
            strategy=self.model.strategy, 
            answer=self.model.answer, 
        )

    
class FailedTaskView(TaskView):
    """FailedTaskView is the view of the failed task. This view is used to format the failed task to a string.
    
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str = """ - [f] {question}\n\t - Description: {description}\n\t - Status: {status}\n\t - Is Leaf: {is_leaf}\n\t - Strategy: {strategy}\n\t - Failed Reason: {failed_reason}"""
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self) -> str:
        return self.template.format(
            question=self.model.question, 
            description=self.model.description, 
            status=self.model.status, 
            is_leaf=self.model.is_leaf, 
            strategy=self.model.strategy, 
            failed_reason=self.model.answer, 
        )


class CancelledTaskView(TaskView):
    """CancelledTaskView is the view of the cancelled task. This view is used to format the cancelled task to a string.
    
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str = """ - [c] {question}\n\t - Description: {description}\n\t - Status: {status}\n\t - Is Leaf: {is_leaf}\n\t - Strategy: {strategy}\n\t - Cancelled Reason: {cancelled_reason}"""
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self) -> str:
        return self.template.format(
            question=self.model.question, 
            description=self.model.description, 
            status=self.model.status, 
            is_leaf=self.model.is_leaf, 
            strategy=self.model.strategy, 
            cancelled_reason=self.model.answer, 
        )
