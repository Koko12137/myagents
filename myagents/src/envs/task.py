import re
from uuid import uuid4
from collections import OrderedDict

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
    history: list[CompletionMessage | ToolCallRequest | ToolCallResult] = Field(
        description="The messages of the current task.", default=[], 
    )
    
    # Observe the task
    def observe(self) -> str:
        """Observe the task according to the current status.
        
        - PENDING:
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
        return TaskContextView(self).format()


class TaskContextView(TaskView):
    """TaskContextView is the view of the task context. This view is used to format the task context to a string.
    
    Attributes:
        model (Task):
            The task to be viewed.
            
        question (str):
            The question to be answered.
        description (str):
            The description of the task.
        status (TaskStatus):
            The status of the current task.
        strategy (TaskStrategy):
            The strategy of the current task.
        parent (Task | None):
            The parent task of the current task.
        sub_tasks (OrderedDict[str, Task]):
            The sub-tasks of the current task.
    """
    model: Task
    
    # View fields
    question: str
    description: str
    status: TaskStatus
    strategy: TaskStrategy
    parent: 'Task'
    sub_tasks: OrderedDict[str, 'Task']
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
        # View fields
        self.question = model.question
        self.description = model.description
        self.status = model.status
        self.strategy = model.strategy
        self.parent = model.parent
        self.sub_tasks = model.sub_tasks
        
    def format(self) -> str:
        """Format the task context to a string.
        
        Returns:
            str:
                The formatted task context.
        """
        question = f"- Question: \n\t{self.question}\n"
        description = f"- Description: \n\t{self.description}\n"
        status = f"- Status: \n\t{self.status}\n"
        strategy = f"- Strategy: \n\t{self.strategy}\n"
        
        # Process the parent task
        parent_task_information_with_status = []
        if self.parent is not None:
            info = f"\t- Question: \n\t\t{self.parent.question}\n" + \
                f"\t- Description: \n\t\t{self.parent.description}\n" + \
                f"\t- Status: \n\t\t{self.parent.status}\n" + \
                f"\t- Strategy: \n\t\t{self.parent.strategy}\n"
            parent_task_information_with_status.append(info)
        else:
            parent_task_information_with_status.append("\tParent task is None.")
        parent_task_information_with_status = "\n".join(parent_task_information_with_status)
        parent = f"- Parent: \n{parent_task_information_with_status}\n"
        
        # Process the sub-tasks
        sub_tasks_information_with_status = []
        if len(self.sub_tasks) > 0:
            for i, sub_task in enumerate(self.sub_tasks.values()):
                observe = f"\t\t- Question: \n\t\t\t{sub_task.question}\n" + \
                    f"\t\t- Description: \n\t\t\t{sub_task.description}\n" + \
                    f"\t\t- Status: \n\t\t\t{sub_task.status}\n" + \
                    f"\t\t- Is Leaf: \n\t\t\t{sub_task.is_leaf}\n" + \
                    f"\t\t- Answer: \n\t\t\t{sub_task.answer}"
                info = f"\t- Sub-Task {i+1}: \n{observe}"
                sub_tasks_information_with_status.append(info)
        else:
            sub_tasks_information_with_status.append("The sub-tasks are None.")
        
        sub_tasks_information_with_status = "\n".join(sub_tasks_information_with_status)
        sub_tasks = f"- Sub-Tasks: \n\t{sub_tasks_information_with_status}\n"
        
        return f"=== TASK CONTEXT ===\n{question}\n{description}\n{status}\n{strategy}\n{parent}\n{sub_tasks}"
    