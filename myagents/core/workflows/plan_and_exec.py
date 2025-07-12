from typing import Callable, Any

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import SystemMessage, UserMessage
from myagents.core.interface import Agent, TaskStatus, Context, Stateful, TreeTaskNode
from myagents.core.workflows.react import ReActFlow
from myagents.core.tasks import DocumentTaskView, ToDoTaskView
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.plan_and_exec import (
    PROFILE, 
    PLAN_SYSTEM_PROMPT, 
    PLAN_THINK_PROMPT, 
    EXEC_SYSTEM_PROMPT, 
    EXEC_THINK_PROMPT, 
    ERROR_PROMPT
)


class PlanAndExecFlow(ReActFlow):
    """
    PlanFlow is a workflow for splitting a task into sub-tasks.
    
        
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent): 
            The agent that is used to orchestrate the task.
        prompts (dict[str, str]):
            The prompts for running specific workflow of the workflow. 
            The following prompts are supported:
            - "plan_system": The system prompt of the workflow.
            - "plan_think": The think prompt of the workflow.
            - "exec_system": The system prompt of the workflow.
            - "exec_think": The think prompt of the workflow.
            - "exec_reflect": The reflect prompt of the workflow.
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
    """
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]
    
    def __init__(
        self, 
        profile: str = "", 
        plan_system_prompt: str = "", 
        plan_think_prompt: str = "", 
        exec_system_prompt: str = "", 
        exec_think_prompt: str = "", 
        error_prompt: str = "", 
        *args, 
        **kwargs, 
    ) -> None:
        """Initialize the PlanAndExecFlow workflow.
        
        Args:
            profile (str, optional, defaults to ""):
                The profile of the workflow.
            plan_system_prompt (str, optional, defaults to ""):
                The system prompt of the workflow.
            plan_think_prompt (str, optional, defaults to ""):
                The think prompt of the workflow.
            plan_reflect_prompt (str, optional, defaults to ""):
                The reflect prompt of the workflow.
            exec_system_prompt (str, optional, defaults to ""):
                The system prompt of the workflow.
            exec_think_prompt (str, optional, defaults to ""):
                The think prompt of the workflow.
            exec_reflect_prompt (str, optional, defaults to ""):
                The reflect prompt of the workflow.
            error_prompt (str, optional, defaults to ""):
                The error prompt of the workflow.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the basic information
        self.profile = profile if profile != "" else PROFILE
        # Update the prompts
        self.prompts.update({
            "plan_system": plan_system_prompt if plan_system_prompt != "" else PLAN_SYSTEM_PROMPT.format(profile=self.profile),
            "plan_think": plan_think_prompt if plan_think_prompt != "" else PLAN_THINK_PROMPT,
            "exec_system": exec_system_prompt if exec_system_prompt != "" else EXEC_SYSTEM_PROMPT,
            "exec_think": exec_think_prompt if exec_think_prompt != "" else EXEC_THINK_PROMPT,
            "error_prompt": error_prompt if error_prompt != "" else ERROR_PROMPT,
        })
        
        # Post initialize to initialize the tools
        self.post_init()
        
    async def run(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Plan and execute the target. This workflow will plan the task and execute the task. 

        Args:
            target (TreeTaskNode):
                The target to plan and execute.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking.
            prompts (dict[str, str], optional, defaults to {}):
                The prompts of the workflow. The following prompts are supported:
                - "plan_system": The system prompt of the workflow.
                - "plan_think": The think prompt of the workflow.
                - "exec_system": The system prompt of the workflow.
                - "exec_think": The think prompt of the workflow.
                - "exec_reflect": The reflect prompt of the workflow.
            completion_config (dict[str, Any], optional, defaults to {}):
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

        error_prompt = prompts.pop("error_prompt", self.prompts["error_prompt"])
        
        # Record the current error retry count
        current_error_retry = 0
        
        while running_checker(target):
            
            # Check if the target is created, if True, then plan the task
            if target.is_created():
                # Plan the task
                target = await self.plan(
                    target, 
                    max_error_retry, 
                    max_idle_thinking, 
                    prompts, 
                    completion_config, 
                    running_checker,
                )
            
            elif target.is_running():
                # Execute the task
                target = await self.execute(
                    target, 
                    max_error_retry, 
                    max_idle_thinking, 
                    prompts, 
                    completion_config, 
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
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Plan the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (TreeTaskNode):
                The task to plan.
            max_idle_thinking (int):
                The maximum number of idle thinking.

        Returns:
            TreeTaskNode: 
                The target after planning.
        """
        # Prepare the prompts 
        plan_system = prompts.pop("plan_system", self.prompts["plan_system"])
        plan_think = prompts.pop("plan_think", self.prompts["plan_think"])
        # Update the prompts
        prompts = {
            "react_system": plan_system,
            "react_think": plan_think,
        }
        
        # Call the parent class to reason and act
        await super().reason_act_reflect(
            target, 
            max_error_retry, 
            max_idle_thinking, 
            prompts, 
            completion_config, 
            running_checker, 
        )
        return target
    
    async def execute(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Deep first execute the task. This is the post step of the planning in order to execute the task. 
        
        Args:
            target (TreeTaskNode):
                The task to deep first execute.
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking. 
            prompts (dict[str, str]):
                The prompts of the workflow. The following prompts are supported:
                - "exec_system": The system prompt of the workflow.
                - "exec_think": The think prompt of the workflow.
                - "exec_reflect": The reflect prompt of the workflow.
            completion_config (dict[str, Any]):
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
                    max_error_retry, 
                    max_idle_thinking, 
                    prompts, 
                    completion_config, 
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
            # Prepare the prompts
            exec_system = prompts.pop("exec_system", self.prompts["exec_system"])
            exec_think = prompts.pop("exec_think", self.prompts["exec_think"])
            # Update the prompts
            prompts = {
                "react_system": exec_system,
                "react_think": exec_think,
            }
            
            # Call the parent class to reason, act and reflect
            target = await self.reason_act_reflect(
                target, 
                max_error_retry, 
                max_idle_thinking, 
                prompts, 
                completion_config, 
                *args, 
                **kwargs,
            )
            
        return target 
    
    async def reason_act_reflect(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Reason, act and reflect on the target. This is the post step of the planning in order to execute the task. 
        
        Args:
            target (TreeTaskNode):
                The task to reason, act and reflect.
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking. 
            prompts (dict[str, str]):
                The prompts of the workflow. The following prompts are supported:
                - "exec_system": The system prompt of the workflow.
                - "exec_think": The think prompt of the workflow.
                - "exec_reflect": The reflect prompt of the workflow.
            completion_config (dict[str, Any]):
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
        
        # Prepare the prompts
        react_system = prompts.pop("react_system", self.prompts["react_system"])
        react_think = prompts.pop("react_think", self.prompts["react_think"])
        react_reflect = prompts.pop("react_reflect", self.prompts["react_reflect"])
        
        # Get the blueprint from the context
        blueprint = self.agent.env.context.get("blueprint")
        # Get the task from the context
        task = self.agent.env.context.get("task")
        # Convert to task answer view
        task_result = DocumentTaskView(task).format()
        
        # Append the system prompt to the history
        message = SystemMessage(content=react_system.format(
            blueprint=blueprint, 
            task_result=task_result,
        ))
        target.update(message)
            
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        while target.is_running():
            # === Reason and Act Stage ===
            target, current_error, current_thinking = await self.reason_act(
                target, 
                react_think=react_think,
                max_error_retry=max_error_retry, 
                current_error=current_error, 
                max_idle_thinking=max_idle_thinking, 
                current_thinking=current_thinking, 
                prompts=prompts, 
                completion_config=completion_config, 
            )
            # Get the last message
            message = target.get_history()[-1]
            # Extract the final output from the message
            final_output = extract_by_label(message.content, "final_output", "final answer", "output", "answer")
            if final_output != "":
                # Set the answer of the task
                target.answer = final_output
            
            # Check if the task is cancelled
            if target.status == TaskStatus.CANCELLED:
                # The task is cancelled, end the workflow
                break
            
            # === Reflect Stage ===
            target, finish_flag = await self.reflect(
                target, 
                react_reflect=react_reflect,
            )
            if finish_flag:
                # Set the task status to finished
                target.to_finished()
        
        # Set the answer of the task
        if not target.answer: 
            target.answer = "任务执行结束，但未提供答案，执行可能存在未知错误。"
            
        # Log the answer
        logger.info(f"任务执行结束: \n{ToDoTaskView(target).format()}")
        return target
