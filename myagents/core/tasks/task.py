import re
import json
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
        objective (str): 
            The objective of the task.
        key_results (str):
            The key results of the task and the verification method for the results.
        results (str):
            The results of the task. If the task is not finished, the results is None.
        next (TreeTaskNode):
            The next task of the current task. If the task does not have a next task, the next is None.
        prev (TreeTaskNode):
            The previous task of the current task. If the task does not have a previous task, the prev is None.
        
        parent (TreeTaskNode):
            The parent task of the current task. If the task does not have a parent task, the parent is None.
        sub_tasks (OrderedDict[str, TreeTaskNode]):
            The sub-tasks of the current task. If the task does not have any sub-tasks, the sub-tasks is an empty dictionary.
        sub_task_depth (int):
            The max number of layers of sub-objective layers that can be split from the objective.
    """
    status: TaskStatus
    history: dict[TaskStatus, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]]

    uid: str
    objective: str
    key_results: str
    results: str 
    next: TreeTaskNode
    prev: TreeTaskNode
    
    parent: TreeTaskNode
    sub_tasks: OrderedDict[str, TreeTaskNode]
    sub_task_depth: int
    
    def __init__(
        self, 
        uid: str, 
        objective: str, 
        key_results: str, 
        sub_task_depth: int, 
        parent: TreeTaskNode = None, 
        prev: TreeTaskNode = None, 
        **kwargs
    ) -> None:
        """
        Initialize the TreeTaskNode.
        
        Args:
            uid (str):
                The unique identifier of the task.
            objective (str):
                The objective of the task.
            key_results (str):
                The key results of the task and the verification method for the results.
            sub_task_depth (int):
                The max number of layers of sub-objective layers that can be split from the objective.
            parent (TreeTaskNode):
                The parent task of the current task. If the task does not have a parent task, the parent is None.
        """
        super().__init__(status_class=TaskStatus, **kwargs)
        self.uid = uid
        assert isinstance(objective, str), "The objective must be a string."
        self.objective = objective
        assert isinstance(key_results, str), "The key results must be a string."
        self.key_results = key_results
        # Initialize the results
        self.results = ""
        # Initialize the next and prev
        self.next = None
        self.prev = prev
        
        assert parent is None or isinstance(parent, TreeTaskNode), "The parent must be a TreeTaskNode."
        self.parent = parent
        assert isinstance(sub_task_depth, int), "The sub task depth must be an integer."
        self.sub_task_depth = sub_task_depth
        # Initialize the stateful attributes
        self.sub_tasks = OrderedDict()
        
        # Initialize the status
        self.to_created()
        
    def __str__(self) -> str:
        return f"TreeTaskNode(objective={self.objective}, key_results={self.key_results}, status={self.status})"
    
    def __repr__(self) -> str:
        return self.__str__()
        
    async def observe(self, format: str, **kwargs) -> str:
        """Observe the task according to the current status and the format of the observation. 
        
        Args:
            format (str):
                The format of the observation. The format can be:
                    "todo": The task will be formatted to a todo markdown string.
                    "document": The task will be formatted to a document string.
                    "json": The task will be formatted to a json string.
                    "answer": The task will be formatted to a answer string.
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
        
        # Check if sub task depth is 0
        if self.sub_task_depth == 0 and self.prev is None:
            self.to_running()
    
    def is_created(self) -> bool:
        """Check if the task status is created.
        """
        return self.status == TaskStatus.CREATED
    
    def to_running(self) -> None:
        """Set the task status to running. 
        """
        self.status = TaskStatus.RUNNING
    
    def is_running(self) -> bool:
        """Check if the task status is running.
        """
        return self.status == TaskStatus.RUNNING
    
    def to_finished(self) -> None:
        """Set the task status to finished. If all the sub_tasks of parent task are finished, the parent task will 
        be set to running.
        """
        self.status = TaskStatus.FINISHED
        
        # Check if the next task is created
        if self.next and self.next.is_created():
            # Convert the next task to running
            self.next.to_running()
    
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
        self.status = TaskStatus.CANCELED
        
    def is_cancelled(self) -> bool:
        """Check if the task status is cancelled.
        """
        return self.status == TaskStatus.CANCELED


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
        return self.task.results


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
    template: str = """{status_value} {uid} \n\t - 目标: {objective}\n\t - 关键结果: {key_results}"""
    
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
            sub_tasks = f"\t - 子任务: \n{sub_tasks_str}"
        else:
            sub_tasks = f"\t - 子任务: \n\t\t 没有子任务"
        view = f"{view}\n{sub_tasks}"
            
        return view
        
    def _format_markdown(self) -> str:
        return self.template.format(
            status_value=self.task.status.value,
            uid=self.task.uid,
            objective=self.task.objective, 
            key_results=self.task.key_results, 
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
        match self.task.status:
            case TaskStatus.CREATED:
                answer = f"[[placeholder]] 任务未完成。"
            case TaskStatus.RUNNING:
                answer = f"[[placeholder]] 任务正在执行。"
            case TaskStatus.FINISHED:
                answer = self.task.results
            case TaskStatus.ERROR:
                answer = f"[[placeholder]] 任务执行失败。"
            case TaskStatus.CANCELED:
                answer = f"[[placeholder]] 任务已被取消，将被删除。"
            case _:
                raise ValueError(f"The status {self.task.status} is not supported.")
        
        # Add the question and answer of the current task
        answer = f"# {self.task.uid}: {self.task.objective}\n\n{self.task.key_results}\n\n{answer}"
        
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
        "uid": {
            "objective": "The objective of the task.",
            "key_results": "The key results of the task and the verification method for the results.",
            "sub_tasks": {
                "sub_task_uid": {
                    "objective": "The objective of the sub-task.",
                    "key_results": "The key results of the sub-task and the verification method for the results.",
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
            task.uid: {
                "objective": task.objective,
                "key_results": task.key_results,
                "sub_tasks": {
                    sub_task.uid: self._format_dict(sub_task)[sub_task.uid]
                    for sub_task in getattr(task, 'sub_tasks', {}).values()
                }
            }
        }
