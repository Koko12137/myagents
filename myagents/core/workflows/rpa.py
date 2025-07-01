from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.envs.task import TaskContextView
from myagents.core.interface import Agent, Task, TaskStatus, Logger, Workflow, Context
from myagents.core.message import ToolCallRequest, ToolCallResult, MessageRole, CompletionMessage, StopReason
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.utils.tools import ToolView
from myagents.core.utils.extractor import extract_by_label
from myagents.core.utils.strings import find_best_match, levenshtein_distance, safe_string_compare
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
        system_prompt (str):
            The system prompt of the workflow. This is used to set the system prompt of the workflow. 
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
    system_prompt: str
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
                if sub_task.status == TaskStatus.ERROR:
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
            # Set the task status to running recursively
            self.__set_running(task)
            return
        
        @self.register_tool("remove_task")
        async def remove_task(question_path: list[str], reason: str, fuzzy_match: bool = True) -> None:
            """
            删除当前任务特定路径指向的子任务。你仅需要在发现任何子任务的规划存在问题时，才需要调用这个工具。
            
            Args:
                question_path (list[str]):
                    需要删除的子任务的问答路径。这个路径将会被用于找到具有相同问题的子任务。比如：
                    ["目标Objective 1", "目标Objective 1.1", "目标Objective 1.1.1"]，这里"目标Objective 1.1.1"会被删除。
                reason (str):
                    删除子任务的原因。
                fuzzy_match (bool, optional):
                    是否使用模糊匹配。默认为True。如果为True，会使用编辑距离计算每个子任务问题和question_path当前值的差异，
                    选取编辑距离最小的进行删除。如果question_path中含有非法字符，一定要使用模糊匹配，否则无法删除。
            
            Returns:
                None
                
            Raises:
                KeyError:
                    如果问答路径没有在任务中找到。
                ValueError:
                    如果问答路径的第一个问题与根任务的问题不一致。
                IndexError:
                    如果问答路径为空。
            """
            task = self.context.get("task")
            
            # Check if the first question is the root task question
            if question_path[0] == task.question:
                question_path = question_path[1:]
            elif len(question_path) == 0:
                raise IndexError("The question path is empty.")
            
            # Store the actual matched questions for deletion
            actual_matched_questions = []
            
            # Traverse the question path and find the sub-task that has the same question
            for i, question in enumerate(question_path):
                if fuzzy_match and task.sub_tasks:
                    # Use fuzzy matching to find the best match
                    try:
                        matched_question = find_best_match(question, task.sub_tasks.keys())
                        matched_task = task.sub_tasks[matched_question]
                        distance = levenshtein_distance(question, matched_question)
                        self.custom_logger.info(f"模糊匹配: 目标='{question}', 最佳匹配='{matched_question}', 编辑距离={distance}")
                        
                        # 如果编辑距离过大，给出警告
                        if distance > len(question) * 0.5:  # 如果编辑距离超过原字符串长度的一半
                            self.custom_logger.warning(f"编辑距离较大 ({distance})，请确认匹配是否正确")
                        
                        task = matched_task
                        # Store the actual matched question for later deletion
                        actual_matched_questions.append(matched_question)
                    except KeyError as e:
                        # If no sub-tasks found, fall back to exact matching
                        self.custom_logger.warning(f"模糊匹配失败: {e}, 尝试精确匹配")
                        if question not in task.sub_tasks:
                            raise KeyError(f"Question '{question}' not found in sub-tasks and no fuzzy match available.")
                        task = task.sub_tasks[question]
                        actual_matched_questions.append(question)
                else:
                    # Use exact matching with safe string comparison
                    found = False
                    for sub_question, sub_task in task.sub_tasks.items():
                        if safe_string_compare(question, sub_question):
                            task = sub_task
                            actual_matched_questions.append(sub_question)
                            found = True
                            break
                    
                    if not found:
                        raise KeyError(f"Question '{question}' not found in sub-tasks.")
                
                # Reset the status to created
                task.status = TaskStatus.CREATED
            
            # Get the parent task
            parent_task = task.parent
            # Remove the task using the actual matched question
            actual_question_to_remove = actual_matched_questions[-1]
            del task.parent.sub_tasks[actual_question_to_remove]
            
            """ [[ ## Announce the removal to the parent task ## ]] """
            # Append the removal message to the task history
            parent_task.update(TaskStatus.PLANNING, CompletionMessage(
                role=MessageRole.USER, 
                content=f"Error in the planning: {reason}, the sub-task {actual_question_to_remove} is removed.", 
                stop_reason=StopReason.NONE, 
            ))
            
            # Set the task to the current task
            task = self.context.get("task")
            # Reset the status to created
            task.status = TaskStatus.CREATED
            # Append the removal message to the task history
            task.update(TaskStatus.CREATED, CompletionMessage(
                role=MessageRole.USER, 
                content=f"The sub-task {actual_question_to_remove} is removed. Reason: {reason}", 
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
        
    def __set_running(self, task: Task) -> None:
        """Set the task status to running recursively.
        
        Args:
            task (Task):
                The task to set the status to running.
        """
        task.status = TaskStatus.RUNNING
        for sub_task in task.sub_tasks.values():
            self.__set_running(sub_task)
        
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
            if blueprint != "":
                # Log the blueprint
                self.custom_logger.info(f"规划蓝图: \n{blueprint}")
                # Update the blueprint to the global context of react flow and all the sub-flows
                self.context = self.context.create_next(blueprint=blueprint, task=env)
                for workflow in self.workflows.values():
                    workflow.context = workflow.context.create_next(blueprint=blueprint, task=env)
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
        # Record the current thinking and the max thinking
        current_thinking = 0
        max_thinking = 3
        # Record the current error and the max error
        current_error = 0
        max_error = 3
        
        # Tools description
        tool_str = ""
        for name, tool in self.tools.items():
            if name == "retry_task": # NOTE: retry_task is not a tool for planning, it is a tool for acting.
                continue
            tool_str += f"{ToolView(tool).format()}\n"
        special_tools = {k: v for k, v in self.tools.items() if k != "retry_task"}
        
        # Plan the current task and check whether the planning result is correct
        while env.status in [TaskStatus.CREATED, TaskStatus.CHECKING]: 
            if env.status == TaskStatus.CREATED:
                # Reset the current thinking
                current_thinking = 0
                # Plan the task
                env = await self.workflows["plan"].run(env)
            
            ## Check if the planning flow is finished properly ##
            elif env.status == TaskStatus.CHECKING:
                # Observe the task
                observe = await self.agent.observe(env)
                # Log the observation
                self.custom_logger.info(f"当前观察: \n{observe}")
                # Think about the env task
                message = CompletionMessage(
                    role=MessageRole.USER, 
                    content=RPA_CHECK_PROMPT.format(task_context=observe, tools=tool_str), 
                    stop_reason=StopReason.NONE, 
                )
                # Append the message to the task history
                env.update(TaskStatus.CREATED, message)
                
                # Call for completion
                message: CompletionMessage = await self.agent.think(
                    env.history[TaskStatus.CREATED], 
                    allow_tools=False, 
                    external_tools=special_tools, 
                )
                # Log the message
                self.custom_logger.info(f"模型回复: \n{message.content}")
                # Record the completion message
                env.update(TaskStatus.CREATED, message)
                
                # Extract the finish flag from the message, this will not be used for tool calling.
                finish_flag = extract_by_label(message.content, "finish_flag", "finish flag", "finish")
                if finish_flag != "":
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
                        
                        # Call the tool
                        result = await self.call_tool(env, tool_call)
                        
                        if result.is_error:
                            # Update the current error
                            current_error += 1
                        
                        # Append the result to the task history
                        env.update(TaskStatus.CREATED, result)
                        
                elif finish_flag:
                    # Set the task status to running recursively
                    self.__set_running(env)
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
                        self.custom_logger.warning(f"连续思考上限达到，强制设置任务状态为运行并退出循环: \n{TaskContextView(env).format()}")
                        # Force the task status to running
                        self.__set_running(env)
                        # Break the loop
                        break
                    
                    # Announce the idle thinking
                    message = CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"【注意】：你已经思考了 {current_thinking} 次，但是没有找到任何工具调用。在思考 {max_thinking} 次后，你将会被强制退出循环。", 
                        stop_reason=StopReason.NONE, 
                    )
                    # Append the message to the task history
                    env.update(TaskStatus.CREATED, message)
                    
                # Check if the current error is greater than the max error
                if current_error > max_error:
                    # Announce the error
                    message = CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"【注意】：你已经达到了 {max_error} 次错误上限，你将会被强制退出循环。", 
                        stop_reason=StopReason.NONE, 
                    )
                    # Append the message to the task history
                    env.update(TaskStatus.CREATED, message)
                    # No more error is allowed, break the loop
                    self.custom_logger.warning(f"连续错误上限达到，强制设置任务状态为运行并退出循环: \n{TaskContextView(env).format()}")
                    # Force the task status to running
                    self.__set_running(env)
                    # Break the loop
                    break

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
        
        if env.status == TaskStatus.CREATED:
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
        while env.status not in [TaskStatus.FINISHED, TaskStatus.ERROR, TaskStatus.CANCELLED]:  
            if env.status == TaskStatus.CREATED:
                # Reason about the task and create the blueprint
                env = await self.__reason_and_plan(env)
            
            # Log the current task
            self.custom_logger.info(f"\nReason Plan Act Flow 正在执行任务: {env.question}")
            
            try:
                # Act the task from root
                await self.workflows["action"].run(env)
            except Exception as e:
                # Unexpected error
                self.custom_logger.error(f"Unexpected error: {e}")
                raise e
            
            if env.status == TaskStatus.CREATED:
                # Clean up the answer of the env
                env.answer = ""
        
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
        
        # Create a new context
        self.context = self.context.create_next(task=ctx, **kwargs)
        
        try:
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
                
            elif tool_call.name == "remove_task":
                await self.tool_functions["remove_task"](**tool_call.args)
                    
                # Create a new result
                result = ToolCallResult(
                    tool_call_id=tool_call.id, 
                    is_error=False, 
                    content="The sub-task is removed.",
                )
        
        except Exception as e:
            # Unexpected error
            self.custom_logger.error(f"Unexpected error: {e}, traceback: \n{format_exc()}")
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=True, 
                content=f"Unexpected error: {e}",
            )
        finally:
            # Done the current context
            self.context = self.context.done()
        
        # Return the result
        return result
