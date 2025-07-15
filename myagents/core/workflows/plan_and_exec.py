import json
from enum import Enum
from typing import Callable, Any

from json_repair import repair_json
from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import SystemMessage, UserMessage
from myagents.core.interface import Agent, TaskStatus, Context, Stateful, TreeTaskNode
from myagents.core.workflows.react import ReActFlow
from myagents.core.workflows.orchestrate import OrchestrateFlow
from myagents.core.tasks import DocumentTaskView, ToDoTaskView, BaseTreeTaskNode
from myagents.core.utils.extractor import extract_by_label
from myagents.core.utils.strings import normalize_string
from myagents.prompts.workflows.plan_and_exec import PROFILE


class PlanAndExecStage(Enum):
    """The stage of the plan and exec workflow.
    
    Attributes:
        PLAN_INIT (int):
            The init stage of the plan stage.
        PLAN_REASON_ACT (int):
            The reason and act stage of the plan stage.
        PLAN_REFLECT (int):
            The reflect stage of the plan stage.
        EXEC_INIT (int):
            The init stage of the exec stage.
        EXEC_REASON_ACT (int):
            The reason and act stage of the exec stage.
        EXEC_REFLECT (int):
            The reflect stage of the exec stage.
        ERROR (int):
            The error stage of the workflow.
    """
    PLAN_INIT = 0
    PLAN_REASON_ACT = 1
    PLAN_REFLECT = 2
    EXEC_INIT = 3
    EXEC_REASON_ACT = 4
    EXEC_REFLECT = 5
    ERROR = 6


class PlanAndExecFlow(OrchestrateFlow):
    """
    PlanAndExecFlow is a workflow for splitting a task into sub-tasks and executing the sub-tasks.
    
        
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent): 
            The agent that is used to orchestrate the task.
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
        stage (Enum):
            The stage of the workflow.
    """
    # Basic information
    profile: str
    agent: Agent
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]
    # Workflow stage
    stage: Enum
    
    def __init__(
        self, 
        profile: str = PROFILE, 
        *args, 
        **kwargs, 
    ) -> None:
        """Initialize the PlanAndExecFlow workflow.
        
        Args:
            profile (str, optional, defaults to PROFILE):
                The profile of the workflow.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        # Initialize the basic information
        self.profile = profile
        # Post initialize to initialize the tools
        self.post_init()
    
    def create_task(
        self, 
        parent: TreeTaskNode, 
        orchestration: str, 
        current_error: int = 0, 
    ) -> tuple[UserMessage, int]:
        """Create a new task based on the orchestration blueprint.
        
        Args:
            parent (TreeTaskNode):
                The parent task to create the new task.
            orchestration (str):
                The orchestration blueprint to create the new task.
            current_error (int, optional, defaults to 0):
                The current error counter. 
                
        Returns:
            UserMessage:
                The user message after creating the new task.
            int:
                The current error counter.
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
            for question, value in orchestration.items():
                # Convert the value to string
                key_outputs = ""
                for k, output in value['问题描述'].items():
                    key_outputs += f"{k}: {output}; "
                    
                # Create a new task
                new_task = BaseTreeTaskNode(
                    question=normalize_string(question), 
                    description=key_outputs, 
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
                parent.sub_tasks[question] = new_task
        
        try:
            # Repair the json
            orchestration = repair_json(orchestration)
            # Parse the orchestration
            orchestration: dict[str, dict[str, str]] = json.loads(orchestration)
            
            # Create the task
            dfs_create_task(
                parent=parent, 
                orchestration=orchestration, 
                sub_task_depth=parent.sub_task_depth - 1, 
            )
            # Format the task to ToDoTaskView
            view = ToDoTaskView(task=parent).format()
            # Return the user message
            return UserMessage(content=f"【成功】：任务创建成功。任务ToDo视图：\n{view}"), current_error
        
        except Exception as e:
            # Log the error
            logger.error(f"Error creating task: {e}")
            # Return the user message
            return UserMessage(content=f"【失败】：任务创建失败。错误信息：{e}"), current_error + 1
        
    async def run(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: dict[str, Any] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Plan and execute the target. This workflow will plan the task and execute the task. 

        Args:
            target (TreeTaskNode):
                The target to plan and execute.
            max_error_retry (int, optiona):
                The maximum number of error retries.
            max_idle_thinking (int, optional):
                The maximum number of idle thinking.
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            TreeTaskNode:
                The target after planning and executing.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default checker
            running_checker = lambda target: not target.is_finished()

        # Get the error prompt from the agent
        error_prompt = self.agent.prompts[PlanAndExecStage.ERROR]
        
        # Record the current error retry count
        current_error_retry = 0
        
        while running_checker(target):
            
            # Check if the target is created, if True, then plan the task
            if target.is_created():
                # Plan the task
                target = await self.plan(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                )
            
            elif target.is_running():
                # Execute the task
                target = await self.execute(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                )
                # Check if the target is created, if True, then process the error
                if target.is_created():
                    # Convert to cancelled status
                    target.to_cancelled()
                
            elif target.is_finished():
                # Break the loop
                break
            
            elif target.is_error():
                # Clean up all the cancelled sub-tasks
                cancelled_sub_tasks = [sub_task for sub_task in target.sub_tasks.values() if sub_task.is_cancelled()]
                # Delete the cancelled sub-tasks
                for sub_task in cancelled_sub_tasks:
                    del target.sub_tasks[sub_task.question]
                
                # Convert to created status and call for re-planning
                target.to_created()
                # Get the current result
                current_result = DocumentTaskView(target).format()
                # Create a new user message to record the error and the current result
                message = UserMessage(content=error_prompt.format(
                    error_retry=current_error_retry, 
                    max_error_retry=max_error_retry, 
                    error_reason=target.answer,
                    current_result=current_result,
                ))
                target.update(message)
                # Clean up the error information
                target.answer = ""
                
            elif target.is_cancelled():
                # Increment the error retry count
                current_error_retry += 1
                # Log the error
                logger.error(f"任务 {target.question} 处理失败，重试次数: {current_error_retry} / {max_error_retry}。")
                
                # Check if the error retry count is greater than the max error retry
                if current_error_retry > max_error_retry:
                    # Log the error
                    logger.error(f"错误次数 {current_error_retry} 超过最大错误次数 {max_error_retry}，任务终止。")
                    # Record the error information to the answer of the parent task
                    target.parent.answer = target.answer
                    # Break the loop
                    break
                
                # Convert to error status and call for error handling
                target.to_error()
            
            else:
                # Log the error
                logger.critical(f"Invalid target status in plan and exec workflow: {target.get_status()}")
                # Raise the error
                raise RuntimeError(f"Invalid target status in plan and exec workflow: {target.get_status()}")
            
        # Return the target
        return target
        
    async def plan(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: dict[str, Any] = {}, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Plan the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (TreeTaskNode):
                The task to plan.
            max_error_retry (int, optional):
                The maximum number of error retries.
            max_idle_thinking (int, optional):
                The maximum number of idle thinking. 
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 

        Returns:
            TreeTaskNode: 
                The target after planning.
        """
        # Create a new running checker
        running_checker = lambda target: target.is_created()
        # Prepare valid stages
        valid_stages = {
            "init": PlanAndExecStage.PLAN_INIT, 
            "reason_act": PlanAndExecStage.PLAN_REASON_ACT, 
            "reflect": PlanAndExecStage.PLAN_REFLECT, 
        }
        
        # Call the parent class to reason and act
        target = await super().reason_act_reflect(
            target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            running_checker=running_checker, 
            valid_stages=valid_stages,
        )
        return target
    
    async def execute(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: dict[str, Any] = {}, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Deep first execute the task. This is the post step of the planning in order to execute the task. 
        
        Args:
            target (TreeTaskNode):
                The task to deep first execute.
            max_error_retry (int, optional):
                The maximum number of error retries.
            max_idle_thinking (int, optional):
                The maximum number of idle thinking. 
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            TreeTaskNode:
                The target after deep first executing.
        """
        
        # Unfinished sub-tasks
        unfinished_sub_tasks = iter(target.sub_tasks.values())
        # Get the first unfinished sub-task
        sub_task = next(unfinished_sub_tasks, None)
        
        # Traverse all the sub-tasks
        while sub_task is not None:
            
            # Check if the sub-task is created, if True, there must be some error in the sub-task
            if sub_task.is_created():
                # Log the error
                logger.error(f"子任务执行中出现错误: \n{ToDoTaskView(sub_task).format()}")
                # Record the error information to the answer of the parent task
                target.answer = sub_task.answer
                # Cancel all the unfinished sub-tasks
                for sub_task in target.sub_tasks.values():
                    if not sub_task.is_finished():
                        sub_task.to_cancelled()
                # IF all the sub-tasks are cancelled, then set the parent task status to cancelled
                if all(sub_task.is_cancelled() for sub_task in target.sub_tasks.values()):
                    target.to_cancelled()
                else:
                    # Set the parent task status to error
                    target.to_created()
                # Return the target
                return target
                
            # Check if the sub-task is running, if True, then act the sub-task
            elif sub_task.is_running():
                # Log the sub-task
                logger.info(f"执行子任务: \n{ToDoTaskView(sub_task).format()}")
                # Act the sub-task
                sub_task = await self.execute(
                    sub_task, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                )

            # Check if the sub-task is failed, if True, then retry the sub-task
            elif sub_task.is_error():
                # Log the error sub-task
                logger.error(f"子任务执行中出现错误: \n{ToDoTaskView(sub_task).format()}")
                # Set the sub-task status to error
                sub_task.to_cancelled()
                # Record the error information to the answer of the parent task
                target.answer += sub_task.answer
            
            # Check if the sub-task is cancelled, if True, set the parent task status to created and stop the traverse
            elif sub_task.is_cancelled():
                # Log the cancelled sub-task
                logger.error(f"取消所有未执行或执行失败的子任务: \n{ToDoTaskView(target).format()}")
                # Record the error information to the answer of the parent task
                target.answer = sub_task.answer
                
                # Cancel all the sub-tasks
                for sub_task in target.sub_tasks.values():
                    if not sub_task.is_finished():
                        sub_task.to_cancelled()
                # IF all the sub-tasks are cancelled, then set the parent task status to cancelled
                if all(sub_task.is_cancelled() for sub_task in target.sub_tasks.values()):
                    target.to_cancelled()
                else:
                    # Set the parent task status to error
                    target.to_created()
                
                # Return the target
                return target 
            
            # Check if the sub-task is finished, if True, then summarize the result of the sub-task
            elif sub_task.is_finished():
                # Log the finished sub-task
                logger.info(f"子任务执行完成: \n{ToDoTaskView(sub_task).format()}")
                # Get the next unfinished sub-task
                sub_task = next(unfinished_sub_tasks, None)
            
            # ELSE, the sub-task is not created, running, failed, cancelled, or finished, it is a critical error
            else:
                # The sub-task is not created, running, failed, cancelled, or finished, then raise an error
                raise ValueError(f"The status of the sub-task is invalid in action flow: {sub_task.get_status()}")
            
        # Post traverse the task
        # All the sub-tasks are finished, then reason, act and reflect on the task
        if all(sub_task.is_finished() for sub_task in target.sub_tasks.values()):
            # Call the parent class to reason, act and reflect
            target = await self.execute_one(
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                *args, 
                **kwargs,
            )
            
        return target 
    
    async def execute_one(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 2, 
        completion_config: dict[str, Any] = {}, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Reason, act and reflect on the target. This is the post step of the planning in order to execute the task. 
        
        Args:
            target (TreeTaskNode):
                The task to reason, act and reflect.
            max_error_retry (int, optional):
                The maximum number of error retries.
            max_idle_thinking (int, optional):
                The maximum number of idle thinking. 
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            *args: 
                The additional arguments for running the agent.
            **kwargs: 
                The additional keyword arguments for running the agent.
                
        Returns:
            TreeTaskNode:
                The target after reasoning, acting and reflecting.
        """
        # Check if the target is running
        if not target.is_running():
            # The target is not running, return the target
            logger.warning(f"任务 {target.question} 不是运行状态。")
            return target
        
        # Get the prompts from the agent
        exec_system = self.agent.prompts[PlanAndExecStage.EXEC_INIT]
        
        # Get the blueprint from the context
        blueprint = self.agent.env.context.get("blueprint")
        # Get the task from the context
        task = self.agent.env.context.get("task")
        # Convert to task answer view
        task_result = DocumentTaskView(task).format()
        
        # Append the system prompt to the history
        message = SystemMessage(content=exec_system.format(
            blueprint=blueprint, 
            task_result=task_result,
        ))
        target.update(message)
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        while target.is_running():
            # === Reason and Act Stage ===
            target, error_flag, tool_call_flag = await ReActFlow.reason_act(
                self, 
                target, 
                to_stage=PlanAndExecStage.EXEC_REASON_ACT, 
                completion_config=completion_config, 
            )
            
            # Check if the error flag is set
            if error_flag:
                # Increment the error counter
                current_error += 1
                # Notify the error limit to Agent
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                target.update(message)
                # Log the error message
                logger.info(f"Error Message: \n{message}")
                # Check if the error counter is greater than the max error retry
                if current_error >= max_error_retry:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    target.answer += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    # Force the react loop to finish
                    break
            
            # Get the last message
            message = target.get_history()[-1]
            # Extract the final output from the message
            final_output = extract_by_label(message.content, "final_output", "final answer", "output", "answer")
            if final_output != "":
                # Set the answer of the task
                target.answer = final_output
            
            # Check if the task is cancelled
            if target.status == TaskStatus.CANCELED:
                # The task is cancelled, end the workflow
                break
            
            # === Reflect Stage ===
            target, finish_flag = await self.reflect(
                target, 
                to_stage=PlanAndExecStage.EXEC_REFLECT, 
            )
            # Check if the target is finished
            if finish_flag:
                # Set the task status to finished
                target.to_finished()
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
                target.update(message)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    target.answer += f"\n连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。"
        
        # Set the answer of the task
        if not target.answer: 
            target.answer = "任务执行结束，但未提供答案，执行可能存在未知错误。"
            
        # Log the answer
        logger.info(f"任务执行结束: \n{DocumentTaskView(target).format()}")
        return target
