from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import SystemMessage
from myagents.core.interface import Agent, TaskStatus, Context, Stateful, TreeTaskNode
from myagents.core.workflows.react import ReActFlow
from myagents.core.tasks import DocumentTaskView, ToDoTaskView
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.plan import PROFILE, PLAN_ATTENTION_PROMPT, EXEC_PLAN_PROMPT, PLAN_SYSTEM_PROMPT


class PlanAndExecFlow(ReActFlow):
    """
    PlanFlow is a workflow for splitting a task into sub-tasks.
    
        
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent): 
            The agent that is used to orchestrate the task.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
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
        system_prompt: str = "", 
        attention_prompt: str = "", 
        exec_prompt: str = "", 
        *args, 
        **kwargs, 
    ) -> None:
        """Initialize the PlanAndExecFlow workflow.
        
        Args:
            profile (str, optional, defaults to ""):
                The profile of the workflow.
            system_prompt (str, optional, defaults to ""):
                The system prompt of the workflow.
            attention_prompt (str, optional, defaults to ""):
                The attention prompt of the workflow.
            exec_prompt (str, optional, defaults to ""):
                The execution prompt of the workflow.
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
            "system": system_prompt if system_prompt != "" else PLAN_SYSTEM_PROMPT.format(profile=self.profile),
            "attention": attention_prompt if attention_prompt != "" else PLAN_ATTENTION_PROMPT,
            "exec": exec_prompt if exec_prompt != "" else EXEC_PLAN_PROMPT,
        })
        
        # Post initialize to initialize the tools
        self.post_init()
        
    async def run(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
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
            tool_choice (str, optional, defaults to None):
                The tool choice of the agent.
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the agent.
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

        # Record the current error retry count
        current_error_retry = 0
        
        # Update system prompt to history
        message = SystemMessage(content=self.prompts["system"])
        target.update(message)
        
        while running_checker(target):
            
            if target.is_created():
                # Plan the task
                await self.plan(
                    target, 
                    max_error_retry, 
                    max_idle_thinking, 
                    tool_choice, 
                    exclude_tools, 
                    running_checker,
                )
            
            elif target.is_running():
                # Execute the task
                await self.reason_act_reflect(
                    target, 
                    max_error_retry, 
                    max_idle_thinking, 
                    tool_choice, 
                    exclude_tools, 
                    running_checker,
                )
            
            elif target.is_finished():
                # Break the loop
                break
            
            elif target.is_error():
                # Increment the error retry count
                current_error_retry += 1
                # Process the error
                await self.__process_error(target, max_error_retry, max_idle_thinking, tool_choice, exclude_tools, running_checker)
                
                # Check if the error retry count is greater than the max error retry
                if current_error_retry > max_error_retry:
                    # Log the error
                    logger.error(f"Error retry count {current_error_retry} is greater than the max error retry {max_error_retry}.")
                    # Cancel the target 
                    target.to_cancelled()
            
            else:
                # Log the error
                logger.critical(f"Invalid target status in plan and exec workflow: {target.get_status()}")
                # Raise the error
                raise RuntimeError(f"Invalid target status in plan and exec workflow: {target.get_status()}")
            
        # Return the target
        return target
    
    async def __process_error(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Process the error of the target.
        """
        pass
        
    async def plan(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
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
        plan_system = kwargs.pop("plan_system", self.prompts["plan_system"])
        plan_think = kwargs.pop("plan_think", self.prompts["plan_think"])
        
        # Call the parent class to reason and act
        await super().reason_act_reflect(
            target, 
            max_error_retry, 
            max_idle_thinking, 
            tool_choice, 
            exclude_tools, 
            running_checker, 
            plan_system=plan_system,
            plan_think=plan_think,
        )
        return target
    
    async def execute(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Execute the task. This is the post step of the planning in order to execute the task. 
        """
        # Prepare the prompts 
        exec_system = kwargs.pop("exec_system", self.prompts["exec_system"])
        exec_think = kwargs.pop("exec_think", self.prompts["exec_think"])
        exec_reflect = kwargs.pop("exec_reflect", self.prompts["exec_reflect"])
        
        # Unfinished sub-tasks
        unfinished_sub_tasks = iter(target.sub_tasks.values())
        # Get the first unfinished sub-task
        sub_task = next(unfinished_sub_tasks, None)
        
        # Traverse all the sub-tasks
        while sub_task is not None:
                
            # Check if the sub-task is running, if True, then act the sub-task
            if sub_task.is_running():
                # Log the sub-task
                logger.info(f"执行子任务: \n{ToDoTaskView(sub_task).format()}")
                # Act the sub-task
                sub_task = await self.execute(
                    sub_task, 
                    max_error_retry, 
                    max_idle_thinking, 
                    tool_choice, 
                    exclude_tools, 
                    running_checker,
                )

            # Check if the sub-task is failed, if True, then retry the sub-task
            elif sub_task.is_error():
                # Cancel the sub-task
                sub_task.to_cancelled()
            
            # Check if the sub-task is cancelled, if True, set the parent task status to created and stop the traverse
            elif sub_task.is_cancelled():
                # Set the parent task status to created
                target.to_created()
                # Log the cancelled sub-task
                logger.error(f"取消所有未执行或执行失败的子任务: \n{ToDoTaskView(target).format()}")
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
        # All the sub-tasks are finished, then call the action flow to act the task
        target = await self.reason_act_reflect(
            target, 
            max_error_retry, 
            max_idle_thinking, 
            tool_choice, 
            exclude_tools, 
            running_checker, 
            *args, 
            **kwargs,
        )
        
        return target 
    
    async def reason_act_reflect(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Execute the task. This is the post step of the planning in order to execute the task. 
        """
        # Check if the target is running
        if not target.is_running():
            # The target is not running, return the target
            logger.warning(f"任务 {target.question} 不是运行状态。")
            return target
        
        # Prepare the prompts
        exec_system = kwargs.pop("exec_system", self.prompts["exec_system"])
        exec_think = kwargs.pop("exec_think", self.prompts["exec_think"])
        exec_reflect = kwargs.pop("exec_reflect", self.prompts["exec_reflect"])
        
        # Get the blueprint from the context
        blueprint = self.agent.env.context.get("blueprint")
        # Get the task from the context
        task = self.agent.env.context.get("task")
        # Convert to task answer view
        task_result = DocumentTaskView(task).format()
        
        # Append the system prompt to the history
        message = SystemMessage(content=exec_system.format(blueprint=blueprint, task_result=task_result))
        target.update(message)
            
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        while target.is_running():
            # === Reason and Act Stage ===
            target, current_error, current_thinking = await self.reason_act(
                target, 
                react_think=exec_think,
                max_error_retry=max_error_retry, 
                current_error=current_error, 
                max_idle_thinking=max_idle_thinking, 
                current_thinking=current_thinking, 
                tool_choice=tool_choice, 
                exclude_tools=exclude_tools,
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
                react_reflect=exec_reflect,
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
