import json
from collections.abc import Callable

from json_repair import repair_json
from loguru import logger

from myagents.core.messages import UserMessage, SystemMessage
from myagents.core.interface import TreeTaskNode, CompletionConfig, Workflow
from myagents.core.workflows.react import BaseReActFlow
from myagents.core.tasks import ToDoTaskView, BaseTreeTaskNode, DocumentTaskView
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.utils.strings import normalize_string
from myagents.prompts.workflows.plan import PROFILE


class PlanWorkflow(BaseReActFlow):
    """PlanWorkflow is a workflow for planning the task. This overload the `reason_act` method of the BaseReActFlow 
    for generating json and creating the tree task nodes. The following schema is used for the orchestration:
    ```json
    {
        "task_id": {
            "目标描述": "...",
            "关键产出": ["...", "..."],
            "子任务": {
                "task_id": {
                    "目标描述": "...",
                    "关键产出": ["...", "..."],
                    "子任务": {}
                }
            }
        }
    ```
    
    Attributes:
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
        
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to work with the workflow.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        observe_formats (dict[str, str]):
            The format of the observation. The key is the observation name and the value is the format content. 
    """
    
    def __init__(
        self, 
        profile: str = PROFILE, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, Workflow] = {}, 
        **kwargs,
    ) -> None:
        """Initialize the PlanWorkflow.
        
        Args:
            profile (str, optional):
                The profile of the workflow.
            prompts (dict[str, str], optional):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
                The following prompts are required:
                    "system_prompt": The system prompt for the plan workflow.
                    "reason_act_prompt": The reason act prompt for the plan workflow.
                    "reflect_prompt": The reflect prompt for the plan workflow.
            observe_formats (dict[str, str], optional):
                The formats of the observation. The key is the observation name and the value is the format method name. 
                The following observe formats are required:
                    "reason_act": The reason act format for the plan workflow.
                    "reflect": The reflect format for the plan workflow.
            sub_workflows (dict[str, Workflow], optional):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
                sub-workflow instance. 
            **kwargs:
                The additional keyword arguments to be passed to the parent class.
        """
        # Check the prompts
        if "system_prompt" not in prompts:
            raise ValueError("The system prompt is required.")
        if "reason_act_prompt" not in prompts:
            raise ValueError("The reason act prompt is required.")
        if "reflect_prompt" not in prompts:
            raise ValueError("The reflect prompt is required.")
        # Check the observe formats
        if "reason_act" not in observe_formats:
            raise ValueError("The reason act format is required.")
        if "reflect" not in observe_formats:
            raise ValueError("The reflect format is required.")
        
        super().__init__(
            profile=profile, 
            prompts=prompts, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
            **kwargs, 
        )
    
    def create_task(
        self, 
        parent: TreeTaskNode, 
        orchestrate_json: str, 
        sub_task_depth: int = -1, 
    ) -> tuple[UserMessage, bool]:
        """Create a new task based on the orchestration blueprint.
        
        Args:
            parent (TreeTaskNode):
                The parent task to create the new task.
            orchestrate_json (str):
                The orchestration blueprint to create the new task.
            sub_task_depth (int):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                inferred from the target.
                
        Returns:
            UserMessage:
                The user message after creating the new task.
            bool:
                The error flag. 
        """
        def dfs_create_task(
            parent: TreeTaskNode, 
            orchestration: dict[str, dict], 
            sub_task_depth: int, 
        ) -> None:
            if sub_task_depth <= 0:
                # Set the task status to running
                parent.to_running()
                return 
            
            # Traverse the orchestration
            for uid, value in orchestration.items():
                # Convert the value to string
                key_outputs = ""
                for output in value['关键产出']:
                    key_outputs += f"{output}; "
                
                # Create a new task
                new_task = BaseTreeTaskNode(
                    uid=uid, 
                    objective=normalize_string(value['目标描述']), 
                    key_results=key_outputs, 
                    sub_task_depth=sub_task_depth - 1,
                )
                # Create the sub-tasks
                dfs_create_task(
                    parent=new_task, 
                    orchestration=value['子任务'], 
                    sub_task_depth=sub_task_depth - 1, 
                )
                # Link the new task to the parent task
                new_task.parent = parent
                # Add the new task to the parent task
                parent.sub_tasks[uid] = new_task
        
        try:
            # Repair the json
            orchestrate_json = repair_json(orchestrate_json)
            # Parse the orchestration
            orchestration: dict[str, dict[str, str]] = json.loads(orchestrate_json)
            
            # Check if the sub-task depth is -1
            if sub_task_depth == -1:
                # Get the sub-task depth from the parent
                sub_task_depth = parent.sub_task_depth - 1
            
            # Create the task
            dfs_create_task(
                parent=parent, 
                orchestration=orchestration, 
                sub_task_depth=sub_task_depth, 
            )
            # Format the task to ToDoTaskView
            view = ToDoTaskView(task=parent).format()
            # Return the user message
            return UserMessage(content=f"【成功】：任务创建成功。任务ToDo视图：\n{view}"), False
        
        except Exception as e:
            # Log the error
            logger.error(f"Error creating task: {e}")
            # Return the user message
            return UserMessage(content=f"【失败】：任务创建失败。错误信息：{e}"), True
    
    async def reason_act(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on. 
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                inferred from the target.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[TreeTaskNode, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = BaseCompletionConfig(temperature=0.0)
        else:
            # Update the temperature to 0.0
            completion_config.update(
                temperature=0.0, 
                tools=[], 
                format_json=True, 
                allow_thinking=False, 
            )
        
        # Initialize the error and tool call flag
        error_flag = False
        tool_call_flag = False
        
        # === Thinking ===
        # Observe the target
        observe = await self.agent.observe(
            target, 
            prompt=self.prompts["reason_act_prompt"], 
            observe_format=self.observe_formats["reason_act"]
        )
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Think about the target
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # Update the message to the target
        target.update(message)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")

        # === Create Task ===
        # Create new tasks based on the orchestration json
        message, error_flag = self.create_task(
            parent=target, 
            orchestrate_json=message.content, 
            sub_task_depth=sub_task_depth, 
        )
        # Log the message
        logger.info(f"Create Task Message: \n{message.content}")
        # Update the target with the user message
        target.update(message)
        
        # Return the target, error flag and tool call flag
        return target, error_flag, tool_call_flag

    async def run(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[TreeTaskNode], bool] = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Run the workflow.
        
        Args:
            target (TreeTaskNode):
                The target to run the workflow on.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                inferred from the target.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            running_checker (Callable[[TreeTaskNode], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            TreeTaskNode:
                The target after running the workflow.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default running checker
            running_checker = lambda target: target.is_created()
            
        # Run the workflow
        return await super().run(
            target=target, 
            sub_task_depth=sub_task_depth, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            running_checker=running_checker, 
            **kwargs,
        )

    async def reason_act_reflect(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[TreeTaskNode], bool] = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Reason and act on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on. 
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            running_checker (Callable[[TreeTaskNode], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            TreeTaskNode:
                The target after running the workflow.
        """
        # Check if the target has history
        if len(target.get_history()) == 0:
            # Get the system prompt from the workflow
            system_prompt = self.prompts["system_prompt"]
            # Update the system prompt to the history
            message = SystemMessage(content=system_prompt)
            target.update(message)
            # Create a new messsage announcing the blueprint
            blueprint = self.agent.env.context.get("blueprint")
            # Create a ToDoTaskView for the blueprint
            view = ToDoTaskView(task=blueprint).format()
            # Create a UserMessage for the blueprint
            blueprint_message = UserMessage(content=f"## 任务蓝图\n\n{view}")
            # Update the blueprint message to the history
            target.update(blueprint_message)
            # Create a new message for the current task results
            task_results = DocumentTaskView(task=target).format()
            # Create a UserMessage for the task results
            task_results_message = UserMessage(content=f"## 任务目前结果进度\n\n{task_results}")
            # Update the task results message to the history
            target.update(task_results_message)
        
        # Run the workflow
        return await super().reason_act_reflect(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            running_checker=running_checker, 
            **kwargs,
        )
