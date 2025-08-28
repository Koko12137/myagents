import json
import traceback

from json_repair import repair_json
from loguru import logger

from myagents.schemas.messages import UserMessage, SystemMessage
from myagents.schemas.llms.config import BaseCompletionConfig
from myagents.core.interface import TreeTaskNode, CompletionConfig, Workflow, MemoryAgent, CallStack, Workspace
from myagents.core.workflows.react import BaseReActFlow
from myagents.core.tasks import ToDoTaskView, BaseTreeTaskNode, DocumentTaskView
from myagents.core.utils.strings import normalize_string
from myagents.prompts.workflows.plan import PlanPromptGroup


PLAN_LAYER_LIMIT = """
### 规划层级限制
对于这个任务，你最多可以拆解到 {detail_level} 层（层指的是层次遍历任务树的层数，如果用户要求的层数越高，则你需要越详细规划）。
【注意】：如果你的规划超出了层数限制，超出这个层级的拆解将会被视为错误，并会被强制截断。
【提示】：“问题1”表示第一层，“问题1.1”表示第二层，以此类推。
"""


BLUEPRINT_FORMAT = """
## 任务总体规划蓝图
{blueprint}

【注意】：如果规划蓝图没有给出，则请根据任务目标和用户需求，自行规划。
"""


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
        workspace (Workspace):
            The global workspace of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
        
        agent (Agent):
            The agent that is used to work with the workflow.
        prompt_group (PromptGroup):
            The prompt group of the workflow.
        observe_formats (dict[str, str]):
            The format of the observation. The key is the observation name and the value is the format content. 
    """
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        prompt_group: PlanPromptGroup = None, 
        observe_formats: dict[str, str] = None, 
        sub_workflows: dict[str, Workflow] = None, 
        **kwargs,
    ) -> None:
        """Initialize the PlanWorkflow.
        
        Args:
            call_stack (CallStack):
                The call stack of the workflow.
            workspace (Workspace):
                The workspace of the workflow.
            prompt_group (PlanPromptGroup, optional):
                The prompt group of the workflow.
            observe_formats (dict[str, str], optional):
                The formats of the observation. The key is the observation name and the value is the format method name. 
                The following observe formats are required:
                    "reason_act_format": The reason act format for the plan workflow.
                    "reflect_format": The reflect format for the plan workflow.
            sub_workflows (dict[str, Workflow], optional):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
                sub-workflow instance. 
            **kwargs:
                The additional keyword arguments to be passed to the parent class.
        """
        # Check the prompts
        if not isinstance(prompt_group, PlanPromptGroup):
            raise ValueError("The prompt group must be a PlanPromptGroup")
        # Check the observe formats
        if "reason_act_format" not in observe_formats:
            raise ValueError("The reason act format is required.")
        if "reflect_format" not in observe_formats:
            raise ValueError("The reflect format is required.")
        
        super().__init__(
            call_stack=call_stack,
            workspace=workspace,
            prompt_group=prompt_group, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
            **kwargs, 
        )
    
    def create_task(
        self, 
        parent: TreeTaskNode, 
        prev: TreeTaskNode, 
        orchestrate_json: str, 
        sub_task_depth: int, 
    ) -> tuple[UserMessage, bool]:
        """Create a new task based on the orchestration blueprint.
        
        Args:
            parent (TreeTaskNode):
                The parent task to create the new task.
            prev (TreeTaskNode):
                The previous task of the new task.
            orchestrate_json (str):
                The orchestration blueprint to create the new task.
            sub_task_depth (int):
                The depth of the sub-task. 
                
        Returns:
            UserMessage:
                The user message after creating the new task.
            bool:
                The error flag. 
        """
        def dfs_create_task(
            name: str,
            parent: TreeTaskNode, 
            prev: TreeTaskNode, 
            orchestration: dict[str, dict], 
            sub_task_depth: int, 
        ) -> TreeTaskNode:
            sub_task_depth = sub_task_depth - 1
            
            # Convert the orchestration to string
            key_outputs = ""
            for output in orchestration['关键产出']:
                key_outputs += f"{output}; "
            
            # Create a new task
            new_task = BaseTreeTaskNode(
                name=name, 
                objective=normalize_string(orchestration['目标描述']), 
                key_results=key_outputs, 
                sub_task_depth=parent.sub_task_depth - 1, 
                parent=parent, 
            )
            # Add the new task to the parent task
            parent.sub_tasks[name] = new_task
            
            # Get the sub-tasks
            sub_tasks: dict[str, dict] = orchestration.get('子任务', {})
            # Check the sub-task depth
            if len(sub_tasks) > 0 and sub_task_depth > 0:
                # Traverse and create all sub-tasks
                for name, sub_task in sub_tasks.items():
                    # Create the sub-tasks
                    prev = dfs_create_task(
                        name=name, 
                        parent=new_task, 
                        prev=prev, 
                        orchestration=sub_task, 
                        sub_task_depth=sub_task_depth, 
                    )
                    # Check status
                    if prev.is_running():
                        if prev.prev is not None and not prev.prev.is_running():
                            prev.to_created()

            # Link the dependency
            new_task.prev = prev
            if prev is not None:
                prev.next = new_task

            if new_task.sub_task_depth == 0 and new_task.prev is None:
                # Set the task status to running
                new_task.to_running()
                
            return new_task
        
        try:
            # Repair the json
            orchestrate_json = repair_json(orchestrate_json)
            # Parse the orchestration
            orchestration: dict[str, dict[str, str]] = json.loads(orchestrate_json)
            
            # Traverse all the sub-tasks
            for idx, sub_task in enumerate(orchestration):
                # Create the sub-tasks
                prev = dfs_create_task(
                    name=f"任务{idx + 1}", 
                    parent=parent, 
                    prev=prev, 
                    orchestration=sub_task, 
                    sub_task_depth=sub_task_depth, 
                )
                # Check status
                if prev.is_running():
                    if prev.prev is not None and not prev.prev.is_running():
                        prev.to_created()
                        
            # Update the next task of the prev
            if prev is not None:
                prev.next = parent
            parent.prev = prev
            # Format the task to ToDoTaskView
            view = ToDoTaskView(task=parent).format()
            # Return the user message
            return UserMessage(content=f"【成功】：任务创建成功。任务ToDo视图：\n{view}"), False
        
        except AttributeError as e:
            # Log the error
            logger.error(f"Error creating task: {e}", traceback.format_exc())
            # Raise an error
            raise e
        
        except Exception as e:
            # Log the error
            logger.error(f"Error creating task: {e}")
            # Return the user message
            return UserMessage(content=f"【失败】：任务创建失败。错误信息：{e}"), True
    
    async def reason_act(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on. 
            sub_task_depth (int):
                The depth of the sub-task. 
            completion_config (CompletionConfig):
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
            completion_config = BaseCompletionConfig(format_json=True)
        else:
            # Update the format_json to True
            completion_config.update(format_json=True)
        
        # Initialize the error and tool call flag
        error_flag = False
        
        # === Thinking ===
        reason_act_prompt = self.prompt_group.get_prompt("reason_act_prompt").string()
        # Append layer limit to the reason act prompt
        reason_act_prompt = f"{reason_act_prompt}\n\n{PLAN_LAYER_LIMIT.format(detail_level=sub_task_depth)}"
        # Prompt the agent
        await self.agent.prompt(UserMessage(content=reason_act_prompt), target)
        # Observe the target
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reason_act_format"])
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Think about the target
        message = await self.agent.think(llm_name="reason_act", observe=observe, completion_config=completion_config)
        # Update the message to the target
        await self.agent.prompt(message, target)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")

        # === Create Task ===
        # Check if the target has sub-tasks
        if len(target.sub_tasks) > 0:
            # Set the prev of target to the prev of the first sub-task
            target.prev = target.sub_tasks[list(target.sub_tasks.keys())[0]].prev
            # Delete all the sub-tasks
            target.sub_tasks.clear()
        
        # Create new tasks based on the orchestration json
        message, error_flag = self.create_task(
            parent=target, 
            prev=target.prev, 
            orchestrate_json=message.content, 
            sub_task_depth=sub_task_depth, 
        )
        # Log the message
        logger.info(f"Create Task Message: \n{message.content}")
        # Update the target with the user message
        await self.agent.prompt(message, target)
        # Return the target, error flag and tool call flag
        return target, error_flag, True
    
    async def reflect(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool]:
        """Reflect on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reflect on.
            sub_task_depth (int):
                The depth of the sub-task. 
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs: 
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[TreeTaskNode, bool]:
                The target and the finish flag.
        """
        if completion_config is not None:
            # Cancel the json format
            completion_config.update(format_json=False)
        
        # Create a new message for layer limit announcement
        layer_limit_message = UserMessage(content=f"## 拆解层次限制\n\n【注意】：你最多只能拆解 {sub_task_depth} 层子任务。")
        # Update the layer limit message to the target
        await self.agent.prompt(layer_limit_message, target)
        
        # Call the parent reflect method
        return await super().reflect(
            target=target, 
            completion_config=completion_config, 
            **kwargs,
        )
        
    async def schedule(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        blueprint: str = "", 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Override the schedule method of the react workflow.
        
        Args:
            target (TreeTaskNode):
                The target to schedule.
            sub_task_depth (int):
                The depth of the sub-task. 
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            blueprint (str):
                The blueprint of the task.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for scheduling the workflow.
                
        Returns:
            TreeTaskNode:
                The target after scheduling.
                
        Raises:
            RuntimeError:
                If the target is not in the valid statuses.
        """
        if not target.is_created():
            # Log the error
            logger.error(f"Plan workflow requires the target status to be created, but the target status is {target.get_status().value}.")
            # Raise an error
            raise RuntimeError(f"Plan workflow requires the target status to be created, but the target status is {target.get_status().value}.")
            
        # Check if the target has history
        if len(target.get_history()) == 0:
            # Get the system prompt from the workflow
            system_prompt = self.prompt_group.get_prompt("system_prompt").string()
            # Update the system prompt to the history
            await self.agent.prompt(SystemMessage(content=system_prompt), target)
            # Create a UserMessage for the blueprint
            blueprint_message = UserMessage(content=BLUEPRINT_FORMAT.format(blueprint=blueprint))
            # Update the blueprint message to the history
            await self.agent.prompt(blueprint_message, target)
            # Create a new message for the current task results
            task_results = DocumentTaskView(task=target).format()
            # Create a UserMessage for the task results
            task_results_message = UserMessage(content=f"## 任务目前结果进度\n\n{task_results}")
            # Update the task results message to the history
            await self.agent.prompt(task_results_message, target)
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        # Run the workflow
        while target.is_created():
        
            # === Reason Stage ===
            # Reason and act on the target
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                sub_task_depth=sub_task_depth, 
                completion_config=completion_config, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
                # Increment the error counter
                current_error += 1
                # Notify the error limit to Agent
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the error message
                logger.info(f"Error Message: \n{message}")
                # Check if the error counter is greater than the max error retry
                if current_error >= max_error_retry:
                    # Set the task status to error
                    target.to_error()
                    # Log the error message
                    logger.critical(f"错误次数限制已达上限: {current_error}/{max_error_retry}，进入错误状态。")
                    # Force the react loop to finish
                    break
            
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await self.reflect(
                target=target, 
                sub_task_depth=sub_task_depth, 
                completion_config=completion_config, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag:
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Log the error message
                    logger.critical(f"连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。")
                    # Force the loop to break
                    break
            
        return target

    async def run(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        blueprint: str = "", 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Run the workflow.
        
        Args:
            target (TreeTaskNode):
                The target to run the workflow on.
            sub_task_depth (int):
                The depth of the sub-task. 
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            blueprint (str):
                The blueprint of the task.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            TreeTaskNode:
                The target after running the workflow.
        """
        # Check if the sub-task depth is greater than or equal to 1
        if not target.sub_task_depth >= 1:
            # Log the error
            logger.error(f"目标的子任务深度小于 1，无法继续规划。")
            # This target can not be planned
            return target
        
        # Run the workflow
        return await self.schedule(
            target=target, 
            sub_task_depth=sub_task_depth, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            blueprint=blueprint, 
            completion_config=completion_config, 
            **kwargs,
        )


class MemoryPlanWorkflow(PlanWorkflow):
    """MemoryPlanWorkflow is a workflow for planning the task with memory.
    """
    agent: MemoryAgent
        
    def get_memory_agent(self) -> MemoryAgent:
        """Get the memory agent.
        """
        return self.agent
    
    async def extract_memory(
        self, 
        target: TreeTaskNode, 
        **kwargs,
    ) -> str:
        """从目标中提取记忆，将临时记忆清空，返回压缩后的记忆
        
        参数:
            target (TreeTaskNode):
                目标
            **kwargs:
                额外参数
                
        返回:
            str:
                压缩后的记忆
        """
        # Get the memory agent
        memory_agent = self.get_memory_agent()
        # Extract the memory from the target
        return await memory_agent.extract_memory(target, **kwargs)
        
    async def schedule(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        blueprint: str = "", 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Override the schedule method of the react workflow.
        
        Args:
            target (TreeTaskNode):
                The target to schedule.
            sub_task_depth (int):
                The depth of the sub-task. 
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            blueprint (str):
                The blueprint of the task.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for scheduling the workflow.
                
        Returns:
            TreeTaskNode:
                The target after scheduling.
                
        Raises:
            RuntimeError:
                If the target is not in the valid statuses.
        """
        if not target.is_created():
            # Log the error
            logger.error(f"Plan workflow requires the target status to be created, but the target status is {target.get_status().value}.")
            # Raise an error
            raise RuntimeError(f"Plan workflow requires the target status to be created, but the target status is {target.get_status().value}.")
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        # Continue flag
        should_continue = True
        
        # Run the workflow
        while target.is_created() and should_continue:
        
            # === Prepare System Instruction ===
            # Get the system prompt from the workflow
            system_prompt = self.prompt_group.get_prompt("system_prompt").string()
            # Update the system prompt to the history
            await self.agent.prompt(SystemMessage(content=system_prompt), target)
            # Create a UserMessage for the blueprint
            blueprint_message = UserMessage(content=BLUEPRINT_FORMAT.format(blueprint=blueprint))
            # Update the blueprint message to the history
            await self.agent.prompt(blueprint_message, target)
            # Create a new message for the current task results
            task_results = DocumentTaskView(task=target).format()
            # Create a UserMessage for the task results
            task_results_message = UserMessage(content=f"## 任务目前结果进度\n\n{task_results}")
            # Update the task results message to the history
            await self.agent.prompt(task_results_message, target)
        
            # === Reason Stage ===
            # Reason and act on the target
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                sub_task_depth=sub_task_depth, 
                completion_config=completion_config, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
                # Increment the error counter
                current_error += 1
                # Notify the error limit to Agent
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the error message
                logger.info(f"Error Message: \n{message}")
                # Check if the error counter is greater than the max error retry
                if current_error >= max_error_retry:
                    # Set the task status to error
                    target.to_error()
                    # Force the react loop to finish
                    break
            
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await self.reflect(
                target=target, 
                sub_task_depth=sub_task_depth, 
                completion_config=completion_config, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag:
                # Force the loop to break
                should_continue = False
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Log the error message
                    logger.critical(f"连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。")
                    # Force the loop to break
                    should_continue = False
            
            # === Extract Memory ===
            # Extract the memory from the target
            compressed_memory = await self.extract_memory(target, **kwargs)
            # Update the compressed memory to the history
            await self.agent.update_temp_memory(temp_memory=compressed_memory, target=target)
            
        return target
