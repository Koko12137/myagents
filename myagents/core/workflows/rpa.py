import re
from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.envs.task import TaskContextView
from myagents.core.interface import Agent, Task, TaskStatus, Logger, Workflow, Context
from myagents.core.message import ToolCallRequest, ToolCallResult, MessageRole, CompletionMessage, StopReason
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.workflows.act import TaskCancelledError
from myagents.core.utils.tools import ToolView
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.rpa import RPA_CHECK_PROMPT, RPA_PLAN_PROMPT


class ReasonPlanActFlow(BaseWorkflow):
    """This is use for Reasoning, Planning and Acting about the task. This flow mainly controls the loop 
    with following steps:
    
    - Execute the current context directly if the plans do not need to update. 
        1. Check the strategy of the task and the status of the sub-tasks.
            - If the strategy is ALL, continue step 1 until all the sub-tasks are finished.
            - If the strategy is ANY, continue step 1 until one of the sub-tasks is finished.
        2. Execute the task that without sub-tasks and the status is RUNNING. 
        3. Update the current context with the result of the execution. 
        4. Go back to previous call, then turn to step 1.
        5. If there is no parent_task, end the recursive loop.
        
    - Split the task into sub-tasks if the plans need to update.
        1. Decide to split the current task into sub-tasks or modify the current task.
        2. Split the current task into sub-tasks or modify the current task.
        3. Set the sub-tasks or modified task as the current task and go to step 1. 
        
    Attributes:
        agent (Agent): 
            The agent that is used to orchestrate the task.
        debug (bool): 
            Whether to enable the debug mode.
        custom_logger (Logger, defaults to logger): 
            The custom logger. If not provided, the default loguru logger will be used. 
        context (ToolCallContext):
            The context of the tool call.
            
        tools (dict[str, FastMCPTool]): 
            The tools can be used for the agent. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the workflow. These functions can be used to control the workflow. 
        workflows (dict[str, Workflow]):
            The workflows that will be orchestrated to process the task. 
    """
    agent: Agent
    debug: bool
    custom_logger: Logger
    context: Context
    
    tools: dict[str, FastMCPTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]

    def __init__(
        self, 
        agent: Agent, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
        workflows: dict[str, Workflow] = {}, 
        *args: tuple, 
        **kwargs: dict, 
    ) -> None:
        """Initialize the ReActFlow.
        
        Args:
            agent (Agent): 
                The agent that is used to orchestrate the task.
            custom_logger (Logger, optional): 
                The custom logger. If not provided, the default loguru logger will be used. 
            debug (bool, optional): 
                Whether to enable the debug mode.
            workflows (dict[str, Workflow], optional):
                The workflows that will be orchestrated to process the task.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(agent=agent, custom_logger=custom_logger, debug=debug, workflows=workflows, *args, **kwargs)
        
        # Post initialize to initialize the tools
        self.post_init()
        
    def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        
        This method will be called after the initialization of the workflow.
        """
        @self.register_tool("retry_task")
        async def retry_task() -> None:
            """
            如果当前任务在执行阶段中出现任何你认为重试的任务，你可以调用这个工具来重试该任务。
            
            Args:
                None
            
            Returns: 
                None
            """
            task = self.context.get("task")
            
            # Find the first one of the sub-tasks that is failed
            for sub_task in task.sub_tasks.values():
                if sub_task.status == TaskStatus.FAILED:
                    # Retry the sub-task
                    sub_task = await self.__act(sub_task)
                    return
            
            # No failed sub-task, then update the task status
            task.status = TaskStatus.RUNNING
            return
        
        @self.register_tool("finish_plan")
        async def finish_plan() -> None:
            """
            完成当前任务的规划拆解。你仅需要在完成所有子任务后，才需要调用这个工具。你仅需要选择以下选项之一来完成当前任务的规划：
            
            - 调用这个工具来完成当前任务的规划。
            - 在消息中设置完成标志为 True，并不要调用这个工具。
            
            Args:
                None
            
            Returns:
                None
            """
            task = self.context.get("task")
            task.status = TaskStatus.RUNNING
            return
        
        @self.register_tool("cancel_task")
        async def cancel_task(question_path: list[str], reason: str) -> None:
            """
            取消当前任务特定路径指向的子任务。你仅需要在发现任何子任务的规划存在问题时，才需要调用这个工具。
            
            Args:
                question_path (list[str]):
                    错误任务的问答路径。这个路径将会被用于找到具有相同问题的子任务。第一个元素应该是根任务的问题，下一个元素应该是导航到子任务的下一个层级。
                reason (str):
                    取消任务的原因。
            
            Returns:
                None
                
            Raises:
                KeyError:
                    如果问答路径没有在任务中找到。
            """
            task = self.context.get("task")
            # Check if the length of the question path is greater than the detail level
            if len(question_path) > task.detail_level:
                raise ValueError(f"The length of the question path is greater than the detail level: {len(question_path)} > {task.detail_level}")
            
            if question_path[0] == task.question:
                question_path = question_path[1:]

            # Traverse the question path and find the sub-task that has the same question
            for question in question_path:
                task = task.sub_tasks[question]
            
            # Get the parent task
            parent_task = task.parent
            # Cancel the task
            task.status = TaskStatus.CANCELLED
            task.answer = reason
            
            """ [[ ## Announce the cancellation to the parent task ## ]] """
            # Append the cancellation message to the task history
            parent_task.update(TaskStatus.PLANNING, CompletionMessage(
                role=MessageRole.USER, 
                content=f"Error in the planning: {reason}, the sub-task {task.question} is cancelled.", 
                stop_reason=StopReason.NONE, 
            ))
            return
        
        # Check the registered tools count
        if len(self.tools) == 0:
            self.custom_logger.error(f"ReasonPlanActFlow 注册的工具为空: {format_exc()}")
            raise RuntimeError("No tools registered for the rpa flow.")
        
        # Check the tools
        tool_str = ""
        for tool in self.tools.values():
            tool_str += f"{ToolView(tool).format()}\n"
        self.custom_logger.debug(f"Tools: \n{tool_str}")
        
    async def __reason(self, env: Task) -> Task:
        """Reason about the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            env (Task):
                The task to reason about.
        
        Args:
            task (Task): The task to reason about.

        Returns:
            Task: The task with the orchestration plan.
        """
        # This is used for no blueprint found limit.
        # If the agent is thinking more than max_thinking times, the loop will be finished.
        max_thinking = 3
        current_thinking = 0
        
        # Observe the task
        observe = await self.agent.observe(env)
        # Log the observation
        self.custom_logger.info(f"当前观察: \n{observe}")
        # Create a new message for the current observation
        message = CompletionMessage(
            role=MessageRole.USER, 
            content=RPA_PLAN_PROMPT.format(
                task_context=observe, 
                detail_level=env.detail_level, 
            ), 
            stop_reason=StopReason.NONE, 
        )
        # Append the reason prompt to the task history
        env.update(TaskStatus.CREATED, message)
    
        while env.status == TaskStatus.CREATED:
            # Call for completion
            message: CompletionMessage = await self.agent.think(
                env.history[TaskStatus.CREATED], 
                allow_tools=False, 
                external_tools=self.tools, 
            )
            # Log the message
            self.custom_logger.info(f"模型回复: \n{message.content}")
            # Record the completion message
            env.update(TaskStatus.CREATED, message)
            
            # Extract the orchestration blueprint from the task by regular expression
            blueprint = extract_by_label(message.content, "orchestration")
            if blueprint is not None:
                # Log the blueprint
                self.custom_logger.info(f"规划蓝图: \n{blueprint}")
                # Update the blueprint to the global context of react flow and all the sub-flows
                self.context = self.context.create_next(blueprint=blueprint)
                for workflow in self.workflows.values():
                    workflow.context = workflow.context.create_next(blueprint=blueprint)
                break
            else:
                # Update the current thinking
                current_thinking += 1
                # Check if the current thinking is greater than the max thinking
                if current_thinking > max_thinking:
                    # Announce the idle thinking
                    message = CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"【注意】：你已经达到了 {max_thinking} 次思考上限，你将会被强制退出循环。", 
                        stop_reason=StopReason.NONE, 
                    )
                    # Append the message to the task history
                    env.update(TaskStatus.CREATED, message)
                    # No more thinking is allowed, raise an error
                    raise RuntimeError("No orchestration blueprint was found in <orchestration> tags for 3 times thinking.")
                
                # No blueprint is found, create an error message
                message = CompletionMessage(
                    role=MessageRole.USER, 
                    content=f"没有在<orchestration>标签中找到规划蓝图。请重新规划。你已经思考了 {current_thinking} 次，" \
                        f"在思考 {max_thinking} 次后，你将会被强制退出循环。下一步你必须给出规划蓝图，否则你将会被惩罚。", 
                    stop_reason=StopReason.NONE, 
                )
                # Append the error message to the task history
                env.update(TaskStatus.CREATED, message)
                # Log the message
                self.custom_logger.warning(f"模型回复中没有找到规划蓝图，提醒模型重新思考。")
        
        return env
        
    async def __plan(self, env: Task) -> Task:
        """Plan the task.
        
        Args:
            env (Task):
                The task to plan.
        
        Returns:
            Task:
                The task after planning.
        """
        
        # Plan the current task
        while env.status == TaskStatus.CREATED:
            env = await self.workflows["plan"].run(env)
            
            """ [[ ## Check if the planning flow is finished properly ## ]] """
            # Observe the task
            observe = await self.agent.observe(env)
            # Log the observation
            self.custom_logger.info(f"当前观察: \n{observe}")
            # Think about the env task
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=RPA_CHECK_PROMPT.format(task_context=observe), 
                stop_reason=StopReason.NONE, 
            )
            # Append the message to the task history
            env.update(TaskStatus.CREATED, message)
            
            # Call for completion
            message: CompletionMessage = await self.agent.think(
                env.history[TaskStatus.CREATED], 
                allow_tools=False, 
                external_tools=self.tools, 
            )
            # Log the message
            self.custom_logger.info(f"模型回复: \n{message.content}")
            # Record the completion message
            env.update(TaskStatus.CREATED, message)
            
            # Reset the current thinking
            current_thinking = 0
            max_thinking = 3
            
            # Extract the finish flag from the message, this will not be used for tool calling.
            finish_flag = extract_by_label(message.content, "finish_flag", "finish flag", "finish")
            if finish_flag is not None:
                # Check if the finish flag is True
                if finish_flag == "True":
                    finish_flag = True
                else:
                    finish_flag = False
            else:
                finish_flag = False
            
            # Check the stop reason
            if message.stop_reason == StopReason.TOOL_CALL:
                # Traverse all the tool calls
                for tool_call in message.tool_calls:
                    # Reset the current thinking due to tool calling
                    current_thinking = 0
                    
                    try:
                        # Call the tool
                        result = await self.call_tool(env, tool_call)
                    except Exception as e:
                        # Handle the unexpected error
                        self.custom_logger.error(f"工具调用中出现了未知异常: \n{e}")
                        raise e
                    # Append the result to the task history
                    env.update(TaskStatus.CREATED, result)
                    
            elif finish_flag:
                # Set the task status to running
                env.status = TaskStatus.RUNNING
                # Break the loop
                break
            else:
                # Update the current thinking due to no tool calling
                current_thinking += 1
                
                # Check if the current thinking is greater than the max thinking
                if current_thinking > max_thinking:
                    # Announce the idle thinking
                    message = CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"【注意】：你已经达到了 {max_thinking} 次思考上限，你将会被强制退出循环。", 
                        stop_reason=StopReason.NONE, 
                    )
                    # Append the message to the task history
                    env.update(TaskStatus.CREATED, message)
                    # No more thinking is allowed, break the loop
                    self.custom_logger.error(f"连续思考上限达到，强制退出循环: \n{TaskContextView(env).format()}")
                    break
                
                # Announce the idle thinking
                message = CompletionMessage(
                    role=MessageRole.USER, 
                    content=f"【注意】：你已经思考了 {current_thinking} 次，但是没有找到任何工具调用。在思考 {max_thinking} 次后，你将会被强制退出循环。", 
                    stop_reason=StopReason.NONE, 
                )
                # Append the message to the task history
                env.update(TaskStatus.CREATED, message)

        return env
    
    async def __reason_and_plan(self, env: Task) -> Task:
        """Reason about the task and plan the task.
        
        Args:
            env (Task):
                The task to reason about and plan.
        """
        # Check the max detail level
        if env.detail_level == 0: 
            # This is a leaf task, no need to reason and plan
            return env
            
        # Reason about the task
        env = await self.__reason(env)
        # Plan the task
        env = await self.__plan(env)
        return env
        
    async def run(self, env: Task) -> Task:
        """Orchestrate the task and act the task.

        Args:
            env (Task): 
                The task to orchestrate and act.

        Returns:
            Task: 
                The task after orchestrating and acting.
        """
        if env.status == TaskStatus.CREATED:
            # Reason about the task and create the blueprint
            env = await self.__reason_and_plan(env)
        
        while env.status not in [TaskStatus.FINISHED, TaskStatus.FAILED, TaskStatus.CANCELLED]: 
            # Log the current task
            self.custom_logger.info(f"\nReAct 正在执行任务: {env.question}")
            
            try:
                # Act the task from root
                await self.workflows["action"].run(env)
            except TaskCancelledError as e:
                # The re-planning is needed, continue and re-plan the task
                current = self.workflows["action"].context.get("task")
                # Resume the temporary task context of the action flow
                self.workflows["action"].context = self.workflows["action"].context.done()
                current = await self.__plan(current)
                # Resume the temporary blueprint context of the action flow
                self.workflows["action"].context = self.workflows["action"].context.done()
                continue
            except Exception as e:
                # Unexpected error
                self.custom_logger.error(f"Unexpected error: {e}")
                raise e
            
        return env
    
    async def call_tool(self, ctx: Task, tool_call: ToolCallRequest, **kwargs: dict) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            ctx (Task):
                The task to be executed.
            tool_call (ToolCallRequest):
                The tool call request.
            **kwargs (dict):
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult:
                The tool call result.
                
        Raises:
            ValueError:
                If the tool call name is unknown. 
        """
        # Log the tool call
        self.custom_logger.info(f"Tool calling {tool_call.name} with arguments: {tool_call.args}.")
        
        # Check if the tool is maintained by the workflow
        if tool_call.name not in self.tools:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, react flow allow only react tools.")
        
        # Create a new context
        self.context = self.context.create_next(task=ctx, **kwargs)
        
        if tool_call.name == "retry_task":
            await self.tool_functions["retry_task"]()
        
            # Create a new result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content="The task is retried.",
            )
            
        elif tool_call.name == "finish_plan":
            await self.tool_functions["finish_plan"]()
            
            # Create a new result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content="The plan is finished.",
            )
            
        elif tool_call.name == "error_in_plan":
            try:
                await self.tool_functions["error_in_plan"](**tool_call.args)
                content = "The error is raised in the plan. The task will be re-planned."
                is_error = False
            except Exception as e:
                # Unexpected error
                self.custom_logger.error(f"Unexpected error: {e}")
                content = f"There is an unexpected error while calling the error_in_plan tool: {e}"
                is_error = True
                
            # Create a new result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=is_error, 
                content=content,
            )
            
        # Done the current context
        self.context = self.context.done()
        
        # Return the result
        return result
