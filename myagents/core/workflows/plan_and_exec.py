from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import UserMessage
from myagents.core.interface import Agent, Context, TreeTaskNode, ReActFlow, CompletionConfig
from myagents.core.workflows.react import TreeTaskReActFlow
from myagents.core.workflows.orchestrate import OrchestrateFlow
from myagents.core.tasks import DocumentTaskView, ToDoTaskView
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.plan_and_exec import PROFILE


class PlanAndExecFlow(TreeTaskReActFlow):
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
        prompts: dict[str, str], 
        observe_formats: dict[str, str], 
        **kwargs,
    ) -> None:
        """Initialize the OrchestrateFlow.

        Args:
            prompts (dict[str, str]:
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
                The following prompts are required:
                    "orch_plan_system_prompt": The system prompt for the plan workflow of the orchestrate workflow.
                    "orch_plan_reason_act_prompt": The reason act prompt for the plan workflow of the orchestrate workflow.
                    "orch_plan_reflect_prompt": The reflect prompt for the plan workflow of the orchestrate workflow.
                    "orch_exec_system_prompt": The system prompt for the execute workflow of the orchestrate workflow.
                    "orch_exec_reason_act_prompt": The reason act prompt for the execute workflow of the orchestrate workflow.
                    "orch_exec_reflect_prompt": The reflect prompt for the execute workflow of the orchestrate workflow.
                    "exec_system_prompt": The system prompt for the execute workflow.
                    "exec_reason_act_prompt": The reason act prompt for the execute workflow.
                    "exec_reflect_prompt": The reflect prompt for the execute workflow.
                    "error_prompt": The error prompt for the workflow.
            observe_formats (dict[str, str]):
                The formats of the observation. The key is the observation name and the value is the format method name. 
                The following observe formats are required:
                    "orch_plan_reason_act_format": The reason act format for the plan workflow of the orchestrate workflow.
                    "orch_plan_reflect_format": The reflect format for the plan workflow of the orchestrate workflow.
                    "orch_exec_reason_act_format": The reason act format for the execute workflow of the orchestrate workflow.
                    "orch_exec_reflect_format": The reflect format for the execute workflow of the orchestrate workflow.
                    "exec_reason_act_format": The reason act format for the execute workflow.
                    "exec_reflect_format": The reflect format for the execute workflow.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Create the sub-workflows
        sub_workflows = {
            "plan": OrchestrateFlow(
                prompts={
                    "plan_system_prompt": prompts["orch_plan_system_prompt"], 
                    "plan_reason_act_prompt": prompts["orch_plan_reason_act_prompt"], 
                    "plan_reflect_prompt": prompts["orch_plan_reflect_prompt"], 
                    "exec_system_prompt": prompts["orch_exec_system_prompt"], 
                    "exec_reason_act_prompt": prompts["orch_exec_reason_act_prompt"], 
                    "exec_reflect_prompt": prompts["orch_exec_reflect_prompt"], 
                }, 
                observe_formats={
                    "plan_reason_act_format": observe_formats['orch_plan_reason_act_format'], 
                    "plan_reflect_format": observe_formats['orch_plan_reflect_format'], 
                    "exec_reason_act_format": observe_formats['orch_exec_reason_act_format'], 
                    "exec_reflect_format": observe_formats['orch_exec_reflect_format'], 
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
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Run the workflow.
        
        Args:
            target (TreeTaskNode):
                The target to run.
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            TreeTaskNode:
                The target after running.
        """
        return await self.schedule(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            **kwargs,
        )
        
    async def schedule(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Override the schedule method of the react workflow.

        Args:
            target (TreeTaskNode):
                The target to plan and execute. 
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            TreeTaskNode:
                The target after scheduling. 
        
        Raises:
            ValueError:
                If the target is not created.
        """
        # Get the error prompt from the agent
        error_prompt = self.prompts["error_prompt"]
        # Record the current error retry count
        current_error_retry = 0
        
        while not target.is_finished():
            
            # Check if the target is created, if True, then plan the task
            if target.is_created() and (
                # Check if the target has sub-tasks and any of the sub-tasks are cancelled
                (len(target.sub_tasks) != 0 and any(sub_task.is_cancelled() for sub_task in target.sub_tasks.values())) or
                # Check if the target has no sub-tasks
                (len(target.sub_tasks) == 0)
            ):
                # Check if any of the sub-tasks are cancelled
                if any(sub_task.is_cancelled() for sub_task in target.sub_tasks.values()):
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
                
                    # Get all the unfinished sub-tasks
                    unfinished_sub_tasks = [sub_task for sub_task in target.sub_tasks.values() if sub_task.is_cancelled()]
                    # Delete the unfinished sub-tasks
                    for sub_task in unfinished_sub_tasks:
                        del target.sub_tasks[sub_task.name]
                        # Log the deleted sub-task
                        logger.error(f"删除已取消的子任务: \n{ToDoTaskView(sub_task).format()}")
                
                # Plan the task
                target = await self.plan(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    **kwargs,
                )
            
            elif (target.is_created() and len(target.sub_tasks) > 0) or target.is_running():
                # Execute the task
                target = await self.execute(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    **kwargs,
                )
                
            elif target.is_finished():
                # Break the loop
                break
            
            elif target.is_error():
                # Rollback the target to created
                target.to_created()
                # Increment the error retry count
                current_error_retry += 1

                # Check if the error retry count is greater than the max error retry
                if current_error_retry > max_error_retry:
                    # Log the error
                    logger.error(f"错误次数 {current_error_retry} 超过最大错误次数 {max_error_retry}，任务终止。")
                    # Set the target status to cancelled
                    target.to_cancelled()
                    # Break the loop
                    break
                
                # Set the sub task status to cancelled if not finished
                for sub_task in target.sub_tasks.values():
                    if not sub_task.is_finished():
                        sub_task.to_cancelled()
                
                # Log the cancelled sub-task
                logger.error(f"取消所有未执行或执行失败的子任务: \n{ToDoTaskView(target).format()}")
                
            elif target.is_cancelled():
                # Log the cancelled target
                logger.error(f"目标已被取消: \n{ToDoTaskView(target).format()}")
                # Break the loop
                break
            
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
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Plan the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (TreeTaskNode):
                The task to plan.
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking. 
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.

        Returns:
            TreeTaskNode: 
                The target after planning.
        """
        # Call the parent class to reason and act
        target = await self.sub_workflows["plan"].run(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            **kwargs,
        )
        return target
    
    async def execute(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
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
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            TreeTaskNode:
                The target after deep first executing.
        """
        if len(target.sub_tasks) > 0:
            sub_task = target.sub_tasks[list(target.sub_tasks)[0]]
        else:
            sub_task = target
        
        # Traverse all the sub-tasks
        while sub_task != target:
            
            # Check if the sub-task is created, if True, there must be some error in the sub-task
            if sub_task.is_created():
                sub_task = await self.schedule(
                    target=sub_task, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                )
                
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
                # Log the error
                logger.error(f"子任务执行失败: \n{ToDoTaskView(sub_task).format()}")
                # Rollback the target to error
                target.to_error()
                # Return the target
                return target
            
            # Check if the sub-task is cancelled, if True, set the parent task status to created and stop the traverse
            elif sub_task.is_cancelled():
                # Log the error
                logger.error(f"子任务已被取消: \n{ToDoTaskView(sub_task).format()}")
                # Rollback the target to error
                target.to_error()
                # Return the target
                return target
            
            # Check if the sub-task is finished, if True, then summarize the result of the sub-task
            elif sub_task.is_finished():
                # Log the finished sub-task
                logger.info(f"子任务执行完成: \n{ToDoTaskView(sub_task).format()}")
                # Get the next unfinished sub-task
                sub_task = sub_task.next
            
            # ELSE, the sub-task is not created, running, failed, cancelled, or finished, it is a critical error
            else:
                # The sub-task is not created, running, failed, cancelled, or finished, then raise an error
                raise RuntimeError(f"The status of the sub-task is invalid in action flow: {sub_task.get_status()}")
            
        # Post traverse the task
        # All the sub-tasks are finished, then reason, act and reflect on the task
        if all(sub_task.is_finished() for sub_task in target.sub_tasks.values()):
            # Call the parent class to reason, act and reflect
            target = await self.run(
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                **kwargs,
            )
            
        return target 
