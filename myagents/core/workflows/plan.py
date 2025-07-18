import json

from json_repair import repair_json
from loguru import logger

from myagents.core.messages import UserMessage
from myagents.core.interface import TreeTaskNode, CompletionConfig
from myagents.core.workflows.react import BaseReActFlow
from myagents.core.tasks import ToDoTaskView, BaseTreeTaskNode
from myagents.core.utils.strings import normalize_string


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
    
    def create_task(
        self, 
        parent: TreeTaskNode, 
        orchestration: str, 
        sub_task_depth: int = -1, 
    ) -> tuple[UserMessage, bool]:
        """Create a new task based on the orchestration blueprint.
        
        Args:
            parent (TreeTaskNode):
                The parent task to create the new task.
            orchestration (str):
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
            orchestration = repair_json(orchestration)
            # Parse the orchestration
            orchestration: dict[str, dict[str, str]] = json.loads(orchestration)
            
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
        *args, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on. 
            sub_task_depth (int):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                inferred from the target.
            completion_config (CompletionConfig, optional):
                The completion config of the workflow. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[TreeTaskNode, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = CompletionConfig()
        
        # Initialize the error and tool call flag
        error_flag = False
        tool_call_flag = False
        
        # === Instruction ===
        # Get the reason act prompt
        reason_act_prompt = self.prompts["reason_act_prompt"]
        # Create new user message with the reason act prompt
        message = UserMessage(content=reason_act_prompt)
        # Update the target with the user message
        target.update(message)
        
        # === Thinking ===
        # Observe the target
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reason_act"])
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Update the completion config
        completion_config.update(tools=[], format_json=True)
        # Think about the target
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")

        # === Create Task ===
        # Create new tasks based on the orchestration json
        message, error_flag = self.create_task(
            parent=target, 
            orchestration=message.content, 
            sub_task_depth=sub_task_depth, 
        )
        # Log the message
        logger.info(f"Create Task Message: \n{message.content}")
        # Update the target with the user message
        target.update(message)
        
        # Return the target, error flag and tool call flag
        return target, error_flag, tool_call_flag
