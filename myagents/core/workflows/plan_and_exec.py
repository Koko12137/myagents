from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import SystemMessage, UserMessage
from myagents.core.interface import Agent, Context, TreeTaskNode, ReActFlow, CompletionConfig
from myagents.core.workflows.react import BaseReActFlow
from myagents.core.workflows.plan import PlanWorkflow
from myagents.core.tasks import DocumentTaskView, ToDoTaskView
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.plan_and_exec import PROFILE


class PlanAndExecFlow(BaseReActFlow):
    """
    PlanAndExecFlow is a workflow for splitting a task into sub-tasks and executing the sub-tasks.
    
        
    Attributes:
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
        
        profile (str):
            The profile of the workflow.
        agent (Agent): 
            The agent that is used to orchestrate the task.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        observe_format (str):
            The format of the observation.
        sub_workflows (dict[str, ReActFlow]):
            The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
            sub-workflow instance. 
    """
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_format: str
    # Sub-worflows
    sub_workflows: dict[str, ReActFlow]
    
    def __init__(
        self, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        **kwargs,
    ) -> None:
        """Initialize the OrchestrateFlow.

        Args:
            prompts (dict[str, str], optional):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
                The following prompts are required:
                    "plan_system_prompt": The system prompt for the plan workflow.
                    "plan_reason_act_prompt": The reason act prompt for the plan workflow.
                    "plan_reflect_prompt": The reflect prompt for the plan workflow.
                    "exec_system_prompt": The system prompt for the execute workflow.
                    "exec_reason_act_prompt": The reason act prompt for the execute workflow.
                    "exec_reflect_prompt": The reflect prompt for the execute workflow.
                    "error_prompt": The error prompt for the workflow.
            observe_formats (dict[str, str], optional):
                The formats of the observation. The key is the observation name and the value is the format method name. 
                The following observe formats are required:
                    "plan_reason_act": The reason act format for the plan workflow.
                    "plan_reflect": The reflect format for the plan workflow.
                    "exec_reason_act": The reason act format for the execute workflow.
                    "exec_reflect": The reflect format for the execute workflow.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Create the sub-workflows
        sub_workflows = {
            "plan": PlanWorkflow(
                prompts={
                    "system_prompt": prompts["plan_system_prompt"], 
                    "reason_act_prompt": prompts["plan_reason_act_prompt"], 
                    "reflect_prompt": prompts["plan_reflect_prompt"], 
                }, 
                observe_formats={
                    "reason_act_format": observe_formats['plan_reason_act_format'], 
                    "reflect_format": observe_formats['plan_reflect_format'], 
                }, 
            ), 
        }
        
        # Prepare the prompts and observe formats for the exec workflow
        prompts = {
            "system_prompt": prompts["exec_system_prompt"], 
            "reason_act_prompt": prompts["exec_reason_act_prompt"], 
            "reflect_prompt": prompts["exec_reflect_prompt"], 
            "error_prompt": prompts["error_prompt"], 
        }
        observe_formats = {
            "reason_act_format": observe_formats["exec_reason_act_format"], 
            "reflect_format": observe_formats["exec_reflect_format"], 
        }
        
        super().__init__(
            profile=PROFILE, 
            prompts=prompts, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
            **kwargs,
        )
        
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
        """Plan and execute the target. This workflow will plan the task and execute the task. 

        Args:
            target (TreeTaskNode):
                The target to plan and execute. 
            sub_task_depth (int, optional, defaults to -1): 
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                inferred from the target. 
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            running_checker (Callable[[TreeTaskNode], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            TreeTaskNode:
                The target after planning and executing.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default running checker
            running_checker = lambda target: not target.is_finished()
        
        # Get the error prompt from the agent
        error_prompt = self.prompts["error_prompt"]
        # Record the current error retry count
        current_error_retry = 0
        
        while running_checker(target):
            
            # Check if the target is created, if True, then plan the task
            if target.is_created():
                # Plan the task
                target = await self.plan(
                    target=target, 
                    sub_task_depth=sub_task_depth, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    **kwargs,
                )
            
            elif target.is_running():
                # Execute the task
                target = await self.execute(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    running_checker=running_checker, 
                    **kwargs,
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
                    del target.sub_tasks[sub_task.uid]
                
                if target.sub_task_depth > 0:
                    # Convert to created status and call for re-planning
                    target.to_created()
                else:
                    # Convert to running status
                    target.to_running()
                    
                # Get the current result
                current_result = DocumentTaskView(target).format()
                # Create a new user message to record the error and the current result
                message = UserMessage(content=error_prompt.format(
                    error_retry=current_error_retry, 
                    max_error_retry=max_error_retry, 
                    error_reason=target.results,
                    current_result=current_result,
                ))
                target.update(message)
                # Clean up the error information
                target.results = ""
                
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
                    target.parent.outputs = target.results
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
        sub_task_depth: int = -1, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Plan the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (TreeTaskNode):
                The task to plan.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                inferred from the target.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking. 
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.

        Returns:
            TreeTaskNode: 
                The target after planning.
        """
        # Create a new running checker
        running_checker = lambda target: target.is_created()
        
        # Call the parent class to reason and act
        target = await self.sub_workflows["plan"].run(
            target=target, 
            sub_task_depth=sub_task_depth, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            running_checker=running_checker, 
            **kwargs,
        )
        return target
    
    async def execute(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Deep first execute the task. This is the post step of the planning in order to execute the task. 
        
        Args:
            target (TreeTaskNode):
                The task to deep first execute.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking. 
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
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
                target.results = sub_task.outputs
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
                target.results += sub_task.outputs
            
            # Check if the sub-task is cancelled, if True, set the parent task status to created and stop the traverse
            elif sub_task.is_cancelled():
                # Log the cancelled sub-task
                logger.error(f"取消所有未执行或执行失败的子任务: \n{ToDoTaskView(target).format()}")
                # Record the error information to the answer of the parent task
                target.results = sub_task.outputs
                
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
            target = await self.reason_act_reflect(
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                **kwargs,
            )
            
        return target 
    
    async def reason_act_reflect(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Reason, act and reflect on the target. This is the post step of the planning in order to execute the task. 
        
        Args:
            target (TreeTaskNode):
                The task to reason, act and reflect.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking. 
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
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
        
        # Check if the target has history
        if len(target.get_history()) == 0:
            # === Prepare System Instruction ===
            # Get the prompts from the agent
            exec_system = self.prompts["system_prompt"]
            # Append the system prompt to the history
            message = SystemMessage(content=exec_system)
            # Update the system message to the history
            target.update(message)
            
            # Get the blueprint from the context
            blueprint = self.agent.env.context.get("blueprint")
            # Create a UserMessage for the blueprint
            blueprint_message = UserMessage(content=f"## 任务蓝图\n\n{blueprint}")
            # Update the blueprint message to the history
            target.update(blueprint_message)
            
            # Get the task from the context
            task = self.agent.env.context.get("task")
            # Create a UserMessage for the task results
            task_message = UserMessage(content=f"## 任务目前结果进度\n\n{DocumentTaskView(task=task).format()}")
            # Update the task message to the history
            target.update(task_message)
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        while target.is_running():
            # === Reason and Act ===
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
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
                    target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    # Force the react loop to finish
                    break
            
            # Check if the tool call flag is not set
            if not tool_call_flag:
                # Get the last message
                message = target.get_history()[-1]
                # Extract the final output from the message
                final_output = extract_by_label(message.content, "final_output", "final answer", "output", "answer")
                if final_output != "":
                    # Set the answer of the task
                    target.results = final_output
                else:
                    # Announce the empty final output
                    logger.warning(f"Empty final output: \n{message.content}")
                    # Create a new user message to record the empty final output
                    message = UserMessage(content=f"【警告】：没有在<final_output>标签中找到任何内容，你必须将最终输出放在<final_output>标签中。")
                    target.update(message)
            
            # === Reflect ===
            target, finish_flag = await self.reflect(
                target=target, 
                completion_config=completion_config, 
            )
            # Check if the target is finished
            if finish_flag:
                # Set the task status to finished
                target.to_finished()
            
            # Check if the tool call flag is not set
            elif not tool_call_flag and not target.results:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请遵守反思结果，尽快输出最终输出。")
                target.update(message)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    target.results += f"\n连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。"
        
        # Set the answer of the task
        if not target.results and target.is_finished(): 
            target.results = "任务执行结束，但未提供答案，执行可能存在未知错误。"
            
        # Log the answer
        logger.info(f"任务执行结束: \n{DocumentTaskView(target).format()}")
        return target
