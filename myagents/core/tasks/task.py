import re
import json
from uuid import uuid4
from collections import OrderedDict
from typing import Union

from myagents.core.interface import TaskStatus, TaskView, TreeTaskNode
from myagents.core.messages.message import AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult
from myagents.core.state_mixin import StateMixin


class BaseTreeTaskNode(TreeTaskNode, StateMixin):
    """Base Tree Task Node for fundamental usage.
    
    Attributes:
        status (TaskStatus):
            The status of the current task.
        history (dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]]):
            The history of the status of the task. The key is the status of the task, and it indicates the state of the task. 
            The value is a list of the history messages. 
        uid (str): 
            The unique identifier of the task. Do not specify this field. It will be automatically generated.
        question (str): 
            The question to be answered. 
        description (str):
            The detail information and limitation of the task. 
        parent (TreeTaskNode, optional):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, TreeTaskNode]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
        sub_task_depth (int):
            The max number of layers of sub-question layers that can be split from the question.
        answer (str, optional):
            The answer to the question. If the task is not finished, the answer is None.
    """
    status: TaskStatus
    history: dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]

    uid: str
    question: str
    description: str
    parent: TreeTaskNode
    sub_tasks: OrderedDict[str, TreeTaskNode]
    sub_task_depth: int
    answer: str 
    
    def __init__(
        self, 
        question: str, 
        description: str, 
        sub_task_depth: int, 
        parent: TreeTaskNode = None, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(status_class=TaskStatus, *args, **kwargs)
        self.uid = uuid4().hex
        
        assert isinstance(question, str), "The question must be a string."
        self.question = question
        
        assert isinstance(description, str), "The description must be a string."
        self.description = description
        
        assert isinstance(sub_task_depth, int), "The sub task depth must be an integer."
        self.sub_task_depth = sub_task_depth
        
        assert parent is None or isinstance(parent, TreeTaskNode), "The parent must be a TreeTaskNode."
        self.parent = parent
        # Initialize the stateful attributes
        self.sub_tasks = OrderedDict()
        # Initialize the status
        self.to_created()
        # Initialize the answer
        self.answer = ""
        
    def __str__(self) -> str:
        return f"TreeTaskNode(question={self.question}, description={self.description}, status={self.status})"
    
    def __repr__(self) -> str:
        return self.__str__()
        
    async def observe(self, format: str, **kwargs) -> str:
        """Observe the task according to the current status and the format of the observation. 
        
        Args:
            format (str):
                The format of the observation. The format can be:
                 - "todo": The task will be formatted to a todo markdown string.
                 - "document": The task will be formatted to a document string.
                 - "json": The task will be formatted to a json string.
                 - "answer": The task will be formatted to a answer string.
            **kwargs:
                The additional keyword arguments for the observation.
                
        Returns:
            str:
                The formatted task.
        """
        if format == "todo":
            return ToDoTaskView(self).format()
        elif format == "document":
            return DocumentTaskView(self).format()
        elif format == "json":
            return JsonTaskView(self).format()
        elif format == "answer":
            return AnswerTaskView(self).format()
        else:
            raise ValueError(f"The format {format} is not supported.")
    
    def to_created(self) -> None:
        """Set the task status to created. This will also set the sub-tasks to cancelled if the sub-tasks are not finished.
        """
        self.status = TaskStatus.CREATED
        # Convert the sub-tasks to cancelled if the sub-tasks are not finished
        for sub_task in self.sub_tasks.values():
            if not sub_task.is_finished():
                sub_task.to_cancelled()
    
    def is_created(self) -> bool:
        """Check if the task status is created.
        """
        return self.status == TaskStatus.CREATED
    
    def to_running(self) -> None:
        """Set the task status to running. This will also set the parent task to running if the parent task is created and 
        all the sub-tasks are running.
        """
        self.status = TaskStatus.RUNNING
        
        # Convert the parent task to running
        if self.parent:
            # Check if the parent task is created and all the sub-tasks are running
            if self.parent.is_created() and all(sub_task.is_running() for sub_task in self.parent.sub_tasks.values()):
                # Convert the parent task to running
                self.parent.to_running()
    
    def is_running(self) -> bool:
        """Check if the task status is running.
        """
        return self.status == TaskStatus.RUNNING
    
    def to_finished(self) -> None:
        """Set the task status to finished. This will also set the parent task to finished if the parent task is running and 
        all the sub-tasks are finished.
        """
        self.status = TaskStatus.FINISHED
        
        # Convert the parent task to finished
        if self.parent:
            # Check if the parent task is running and all the sub-tasks are finished
            if self.parent.is_running() and all(sub_task.is_finished() for sub_task in self.parent.sub_tasks.values()):
                # Convert the parent task to finished
                self.parent.to_finished()
    
    def is_finished(self) -> bool:
        """Check if the task status is finished.
        """
        return self.status == TaskStatus.FINISHED
    
    def to_error(self) -> None:
        """Set the task status to error.
        """
        self.status = TaskStatus.ERROR  
    
    def is_error(self) -> bool:
        """Check if the task status is error.
        """
        return self.status == TaskStatus.ERROR
    
    def to_cancelled(self) -> None:
        """Set the task status to cancelled.
        """
        self.status = TaskStatus.CANCELLED
        
    def is_cancelled(self) -> bool:
        """Check if the task status is cancelled.
        """
        return self.status == TaskStatus.CANCELLED


class AnswerTaskView(TaskView):
    """AnswerTaskView is the view of the task answer. This view is used to format the task answer to a string.
    
    Attributes:
        task (TreeTaskNode):
            The task to be viewed.
    """
    task: TreeTaskNode
    
    def __init__(self, task: TreeTaskNode) -> None:
        self.task = task
        
    def format(self) -> str:
        return self.task.answer


class ToDoTaskView(TaskView):
    """ToDoTaskView is the view of the task context. This view is used to format the task context to a string.
    
    Example (markdown format):
    ```markdown
    [RUNNING] 主任务问题
        - 描述: 主任务描述
        - 子任务: 
            [RUNNING] 子任务1问题
                - 描述: 子任务1描述
                - 子任务: 
                    No sub-tasks now
            [RUNNING] 子任务2问题
                - 描述: 子任务2描述
                - 子任务: 
                    No sub-tasks now
    ```
    
    Attributes:
        task (TreeTaskNode):
            The task to be viewed.
        template (str):
            The template of the task view.
    """
    task: TreeTaskNode
    template: str = """{status_value}{question}\n\t - 描述: {description}\n\t - 任务状态: {status}"""
    
    def __init__(self, task: TreeTaskNode) -> None:
        self.task = task
        
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
        view = self._format_markdown()
        # Process the sub-tasks
        for sub_task in self.task.sub_tasks.values():
            sub_task_info = ToDoTaskView(sub_task).format(layer=layer-1)
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
        
    def _format_markdown(self) -> str:
        return self.template.format(
            status_value=self.task.status.value,
            question=self.task.question, 
            description=self.task.description, 
            status=self.task.status, 
        )


class DocumentTaskView(TaskView):
    """DocumentTaskView is the view of the task answer. This view is used to format the task answer to a string.
    
    Attributes:
        task (TreeTaskNode):
            The task to be viewed.
    """
    task: TreeTaskNode
    
    def __init__(self, task: TreeTaskNode) -> None:
        self.task = task
        
    def format(self, layer: int = 3) -> str:
        answer = self.task.answer if self.task.answer else "The task is not finished."
        # Add the question and answer of the current task
        answer = f"# {self.task.question}\n\n{answer}"
        
        if layer > 0 and self.task.sub_task_depth > 0:
            sub_answers = [] 
            
            # Traverse the sub-tasks and format the question and answer of the sub-tasks
            for sub_task in self.task.sub_tasks.values():
                # Get the sub-task answer recursively
                sub_answer = DocumentTaskView(sub_task).format(layer=layer-1)
                # Increase header level by adding one more # to existing headers
                sub_answer = re.sub(r'^(#+)', r'#\1', sub_answer, flags=re.MULTILINE)
                sub_answers.append(sub_answer)

            # Concatenate the answer of the sub-tasks and questions directly
            sub_answer = "\n\n".join(sub_answers)
            answer = f"{answer}\n\n{sub_answer}"
        
        return answer


class JsonTaskView(TaskView):
    """JsonTaskView is the view of the task in json format. The json format is a dictionary, and the key is the question 
    of the task, and the value is a dictionary of the task. Example:
    ```json
    {
        "question": {
            "description": "The description of the task.",
            "sub_tasks": {
                "sub_task_question": {
                    "description": "The description of the sub-task.",
                    "sub_tasks": {}
                }
            }
        }
    }
    ```
    
    Attributes:
        task (TreeTaskNode):
            The task to be viewed.
    """
    task: TreeTaskNode
    
    def __init__(self, task: TreeTaskNode) -> None:
        self.task = task
        
    def format(self) -> str:
        """Format the task to a json string, recursively including all sub-tasks."""
        json_task = self._format_dict(self.task)
        return json.dumps(json_task, indent=4, ensure_ascii=False)

    def _format_dict(self, task: TreeTaskNode) -> dict:
        """Recursively format the task and its sub-tasks to a dictionary."""
        return {
            task.question: {
                "description": task.description,
                "sub_tasks": {
                    sub_task.question: self._format_dict(sub_task)[sub_task.question]
                    for sub_task in getattr(task, 'sub_tasks', {}).values()
                }
            }
        }
