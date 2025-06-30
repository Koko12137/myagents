import re
from uuid import uuid4
from collections import OrderedDict
from typing import Union

from pydantic import BaseModel, Field, ConfigDict

from myagents.core.message import CompletionMessage, ToolCallRequest, ToolCallResult, MessageRole, StopReason
from myagents.core.interface import TaskStatus, TaskView, Task, TaskParallelStrategy


class BaseTask(BaseModel):
    """Base Task for fundamental usage.
    
    Attributes:
        uid (str): 
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
            
        question (str): 
            The question to be answered. 
        description (str):
            The detail information and limitation of the task. 
        detail_level (int):
            The max number of layers of sub-question layers that can be split from the question.
        parent (Task, optional):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, Task]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
            
        status (TaskStatus):
            The status of the current task.
        parallel_strategy (TaskParallelStrategy):
            The parallel strategy of the current task.
        is_leaf (bool):
            Whether the current task is a leaf task. If the task is a leaf task, the task will not be orchestrated by the workflow. 
        answer (str, optional):
            The answer to the question. If the task is not finished, the answer is None.
            
        history (dict[TaskStatus, list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]]):
            The history of the status of the task. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    uid: str = Field(default_factory=lambda: uuid4().hex)
    
    # Context
    question: str = Field(description="The question to be answered.")
    description: str = Field(
        description="The detail information and limitation of the task. In this task, whether or not you need a utility, and what information you want to get if you do."
    )
    detail_level: int = Field(description="The max number of layers of sub-question layers that can be split from the question.", default=3)
    parent: 'Task' = Field(description="The parent task of the current task.", default=None)
    sub_tasks: OrderedDict[str, 'Task'] = Field(description="The sub-tasks of the current task.", default={})
    
    # Status
    status: TaskStatus = Field(description="The status of the current task.", default=TaskStatus.CREATED)
    parallel_strategy: TaskParallelStrategy = Field(description="The parallel strategy of the current task.", default=TaskParallelStrategy.SEQUENTIAL)
    is_leaf: bool = Field(description="Whether the current task is a leaf task.", default=False) 
    answer: str = Field(description="The answer to the question.", default="")
    
    # History Messages
    history: dict[TaskStatus, list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]] = Field(
        default_factory=lambda: {
            TaskStatus.CREATED: [],
            TaskStatus.PLANNING: [],
            TaskStatus.CHECKING: [],
            TaskStatus.RUNNING: [],
            TaskStatus.FINISHED: [],
            TaskStatus.ERROR: [],
        },
        description="The history of the status of the task. The key is the status of the task, and it indicates the state of the task. The value is a list of the history messages.", 
    )
    
    def update(
        self, 
        status: TaskStatus, 
        message: Union[CompletionMessage, ToolCallRequest, ToolCallResult], 
    ) -> None:
        """Update the task status. If the last message is the same role as the current message, the content will be 
        concatenated. Otherwise, the message will be appended directly. 
        
        Args:
            status (TaskStatus):
                The status of the task.
            message (Union[CompletionMessage, ToolCallRequest, ToolCallResult]):
                The message to be updated.
        
        Returns:
            None
        """
        if len(self.history[status]) > 0 and isinstance(message, CompletionMessage) and message.role == MessageRole.USER:
            last_message = self.history[status][-1]
            # Check if the last message is the same role as the current message
            if last_message.role == message.role:
                # Concatenate the content of the last message and the current message
                last_message.content = f"{last_message.content}\n{message.content}"
                last_message.stop_reason = message.stop_reason
            else:
                # Append the message directly
                self.history[status].append(message)
        else:
            # Append the message directly
            self.history[status].append(message)
            
    def reset(self) -> None:
        """Reset the task.
        """
        self.status = TaskStatus.CREATED
        self.answer = ""
        self.history = {
            TaskStatus.CREATED: [], 
            TaskStatus.PLANNING: [],
            TaskStatus.CHECKING: [],
            TaskStatus.RUNNING: [],
            TaskStatus.FINISHED: [],
            TaskStatus.ERROR: [],
        }
        # Delete the sub-tasks
        self.sub_tasks = OrderedDict()

    
class OKRTask(BaseTask):
    """OKRTask is the task that is used to implement the OKR.
    
    Attributes:
        objectives (list[str]):
            The objectives of the task.
        key_results (list[str]):
            The key results of the task.
        checklist (list[str]):
            The checklist of the task.
    """
    objectives: list[str] = Field(description="The objectives of the task.")
    key_results: list[str] = Field(description="The key results of the task.")
    checklist: list[str] = Field(description="The checklist of the task.")
    

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
        
        # Return a markdown task view
        view = MarkdownTaskView(self.model).format()
        # Process the sub-tasks
        for sub_task in self.model.sub_tasks.values():
            sub_task_info = TaskContextView(sub_task).format(layer=layer-1)
            # Update the indentation 
            sub_task_info = re.sub(r'^', f"\t" * (layer-1), sub_task_info, flags=re.MULTILINE)
            sub_tasks.append(sub_task_info)
            
        # Check if the sub-tasks is not empty
        if len(sub_tasks) > 0:
            sub_tasks_str = '\n'.join(sub_tasks)
            sub_tasks = f"\t - Sub-Tasks: \n{sub_tasks_str}"
        else:
            sub_tasks = f"\t - Sub-Tasks: \n\t\t No sub-tasks now"
        view = f"{view}\n{sub_tasks}"
            
        return view
    

class MarkdownTaskView(TaskView):
    """MarkdownTaskView is the markdown type view of the task. 
    
    Attributes:
        model (Task):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    model: Task
    template: str = """{status_value}{question}\n\t - 描述: {description}\n\t - 任务状态: {status}\n\t - 是否叶子: {is_leaf}"""
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self) -> str:
        return self.template.format(
            status_value=self.model.status.value,
            question=self.model.question, 
            description=self.model.description, 
            status=self.model.status, 
            is_leaf=self.model.is_leaf, 
        )


class TaskAnswerView(TaskView):
    """TaskAnswerView is the view of the task answer. This view is used to format the task answer to a string.
    
    Attributes:
        model (Task):
            The task to be viewed.
    """
    model: Task
    
    def __init__(self, model: Task) -> None:
        self.model = model
        
    def format(self, layer: int = 3) -> str:
        answer = self.model.answer if self.model.answer else "The task is not finished."
        # Add the question and answer of the current task
        answer = f"# {self.model.question}\n\n{answer}"
        
        if not self.model.is_leaf and layer > 0:
            sub_answers = [] 
            
            # Traverse the sub-tasks and format the question and answer of the sub-tasks
            for sub_task in self.model.sub_tasks.values():
                # Get the sub-task answer recursively
                sub_answer = TaskAnswerView(sub_task).format(layer=layer-1)
                # Increase header level by adding one more # to existing headers
                sub_answer = re.sub(r'^(#+)', r'#\1', sub_answer, flags=re.MULTILINE)
                sub_answers.append(sub_answer)

            # Concatenate the answer of the sub-tasks and questions directly
            sub_answer = "\n\n".join(sub_answers)
            answer = f"{answer}\n\n{sub_answer}"
        
        return answer
