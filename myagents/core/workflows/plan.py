import inspect
from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.message import CompletionMessage, MessageRole, StopReason, ToolCallRequest, ToolCallResult
from myagents.core.interface import Agent, Task, TaskStatus, Logger, Workflow
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.envs.task import BaseTask, TaskContextView
from myagents.core.utils.tools import ToolView
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.plan import PLAN_ATTENTION_PROMPT, EXEC_PLAN_PROMPT, PLAN_SYSTEM_PROMPT


class PlanFlow(BaseWorkflow):
    """
    PlanFlow is a workflow for splitting a task into sub-tasks.
    
    Attributes:
        system_prompt (str):
            The system prompt of the workflow.
        agent (Agent):
            The agent that will be used to plan the task. 
        context (BaseContext):
            The global context container of the workflow.
        debug (bool, defaults to False):
            The debug flag. 
        custom_logger (Logger, defaults to logger): 
            The custom logger. If not provided, the default loguru logger will be used. 
        tools (dict[str, FastMCPTool]):
            The orchestration tools. These tools can be used to control the workflow. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the workflow. These functions can be used to control the workflow. 
        workflows (dict[str, Workflow]):
            The workflows that will be orchestrated to process the task. 
    """
    system_prompt: str = PLAN_SYSTEM_PROMPT
    agent: Agent
    debug: bool
    custom_logger: Logger
    tools: dict[str, FastMCPTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]
    
    def __init__(
        self, 
        agent: Agent, 
        debug: bool = False, 
        custom_logger: Logger = logger, 
        workflows: dict[str, Workflow] = {}, 
        *args: tuple, 
        **kwargs: dict, 
    ) -> None:
        """Initialize the PlanFlow workflow.
        
        Args:
            agent (Agent):
                The agent that will be used to plan the task.
            debug (bool, optional):
                The debug flag. If not provided, the default value is False.
            custom_logger (Logger, optional):
                The custom logger. If not provided, the default loguru logger will be used.
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
        """Post init for the PlanFlow workflow.
        """
        @self.register_tool("create_task")
        async def create_task(
            question: str, 
            description: str, 
            is_leaf: bool, 
        ) -> Task:
            """
            创建一个新的任务，并将其添加到当前任务的子任务列表中。
            
            Args:
                question (str): 
                    当前任务需要回答或解决的问题。
                description (str): 
                    你需要用一段文字来描述当前问题应该达到什么目标，不能仅用一句话来描述。
                is_leaf (bool):
                    当前任务是否是叶子任务。如果当前任务是叶子任务，则当前任务将自动设置为运行状态。
                
            Returns:
                Task: 
                    已创建的子任务。 
            """
            
            # Get the parent task from the context
            parent = self.context.get("task")
            # Create a new task
            new_task = BaseTask(
                question=question, 
                description=description, 
                detail_level=parent.detail_level - 1, 
                is_leaf=is_leaf, 
            )
            
            # Check if the new task is a leaf task
            if not new_task.is_leaf:
                # Set the status of the new task
                new_task.status = TaskStatus.CREATED
            else:
                # Set the status of the new task as running
                new_task.status = TaskStatus.CHECKING
                
            # Add the new task to the task
            parent.sub_tasks[new_task.question] = new_task
            # Add the parent task to the new task
            new_task.parent = parent
            return new_task
            
        @self.register_tool("finish_planning")
        async def finish_planning() -> bool:
            """
            完成当前任务的规划。你仅需要在完成所有子任务后，才需要调用这个工具。你仅需要选择以下选项之一来完成当前任务的规划：
            
            - 调用这个工具来完成当前任务的规划。
            - 在消息中设置完成标志为 True，并不要调用这个工具。
            
            Args:
                None
                
            Returns:
                bool: 
                    True 表示当前任务的规划已经完成，False 表示当前任务的规划未完成。
            """
            # Set the task status to running
            task = self.context.get("task")
            task.status = TaskStatus.CHECKING
            return True
        
        @self.register_tool("set_as_leaf")
        async def set_as_leaf() -> None:
            """
            如果发现当前任务不是叶子任务，但是当前任务在蓝图中的规划中应该是叶子任务，你可以调用这个工具来将当前任务设置为叶子任务。
            
            Args:
                None
                
            Returns:
                None
            """
            # Set the task as a leaf task
            task = self.context.get("task")
            task.is_leaf = True
            task.status = TaskStatus.CHECKING
        
        # Additional tools information for tool calling
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
        
        if self.debug:
            # Check the tools
            self.custom_logger.debug(f"Tools: \n{tools_str}")
            # Check the registered tools count
            if len(self.tools) == 0:
                self.custom_logger.error(f"No tools registered for the plan flow. {format_exc()}")
                raise RuntimeError("No tools registered for the plan flow.")
            
    async def __layer_create(self, task: Task) -> Task:
        """Create the sub-tasks for the task.

        Args:
            task (Task):
                The task to create the sub-tasks.

        Raises:
            ValueError:
                - If the tool call name is unknown. 
                - If more than one tool calling is required. 

        Returns:
            Task:
                The task after created the sub-tasks.
        """
        # Additional tools information for tool calling
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
        
        # Check the history of the task, if the history is empty, then we need to set the system prompt
        if len(task.history[TaskStatus.PLANNING]) == 0:
            # Set the system prompt
            task.update(TaskStatus.PLANNING, CompletionMessage(
                role=MessageRole.SYSTEM, 
                content=self.system_prompt.format(blueprint=self.context.get("blueprint")), 
            ))
        
        # Set the task status to planning
        task.status = TaskStatus.PLANNING
        
        # Observe the task
        observe = await self.agent.observe(task)
        # Log the observation
        self.custom_logger.info(f"当前观察: \n{observe}")
        # Create a new message for the current observation
        message = CompletionMessage(
            role=MessageRole.USER, 
            content=PLAN_ATTENTION_PROMPT.format(
                task_context=observe, 
                tools=tools_str, 
            ), 
            stop_reason=StopReason.NONE, 
        )
        # Append Plan Prompt and Call for Completion
        # Append for current task recording
        task.update(TaskStatus.PLANNING, message)
        # Call for completion
        message: CompletionMessage = await self.agent.think(
            task.history[TaskStatus.PLANNING], 
            allow_tools=False, 
            external_tools=self.tools, 
            tool_choice="auto",
        )
        # Log the message
        self.custom_logger.info(f"模型回复: \n{message.content}")
        # Record the completion message
        task.update(TaskStatus.PLANNING, message)
        
        # This is used for no tool calling thinking limit.
        # If the agent is thinking more than max_thinking times, the loop will be finished.
        max_thinking = 3
        current_thinking = 0
        # This is used for error limit.
        # If the agent is thinking more than max_error times, the loop will be finished.
        max_error = 3
        current_error = 0
        
        # Modify the orchestration
        while task.status == TaskStatus.PLANNING:
            # Observe the task planning history
            observe = await self.agent.observe(task)
            # Log the observation
            self.custom_logger.info(f"当前观察: \n{observe}")

            # Create a new message for the current observation
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=EXEC_PLAN_PROMPT.format(
                    task_context=observe, 
                    task_status_description=inspect.getdoc(TaskStatus), 
                ), 
                stop_reason=StopReason.NONE, 
            )
            # Update the task status
            task.update(TaskStatus.PLANNING, message)
            # Call for completion
            message: CompletionMessage = await self.agent.think(
                task.history[TaskStatus.PLANNING], 
                allow_tools=False, 
                external_tools=self.tools, 
            )
            # Log the message
            self.custom_logger.info(f"模型回复: \n{message.content}")
            # Record the completion message
            task.update(TaskStatus.PLANNING, message)
            
            # Extract the finish flag from the message, this will not be used for tool calling.
            finish_flag = extract_by_label(message.content, "finish_flag", "finish flag", "finish")
            if finish_flag is not None:
                # Extract the finish flag
                # Check if the finish flag is True
                if finish_flag == "True":
                    finish_flag = True
                else:
                    finish_flag = False
            else:
                finish_flag = False

            # Check the stop reason
            if message.stop_reason == StopReason.TOOL_CALL:
                # Reset the current thinking
                current_thinking = 0
                
                # Traverse all the tool calls
                for tool_call in message.tool_calls:
                    try:
                        # Call the tool
                        tool_result = await self.call_tool(task, tool_call)
                        
                    except ValueError as e:
                        # Handle the error
                        tool_result = ToolCallResult(
                            tool_call_id=tool_call.id, 
                            is_error=True, 
                            content=f"工具调用 {tool_call.name} 失败: \n{e}", 
                        )
                        # Handle the error and update the task status
                        self.custom_logger.warning(f"工具调用 {tool_call.name} 失败: \n{e}")
                    
                    except Exception as e:
                        # Handle the unexpected error
                        self.custom_logger.error(f"工具调用中出现了未知异常: \n{e}")
                        raise e
                    
                    # Append for current task recording
                    task.update(TaskStatus.PLANNING, tool_result)
                    
            elif finish_flag:
                # Update the current thinking
                current_thinking += 1
                # Set the task status to running
                task.status = TaskStatus.CHECKING
            else:
                # Update the current thinking
                current_thinking += 1
            
                # Check if the current thinking is greater than the max thinking
                if current_thinking > max_thinking:
                    # No more tool calling is allowed, break the loop
                    self.custom_logger.error(f"连续思考上限达到，将任务状态设置为运行状态，并强制退出循环: \n{TaskContextView(task).format()}")
                    task.status = TaskStatus.CHECKING
                    # Announce the idle thinking
                    message = CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"【注意】：你已经达到了 {max_thinking} 次思考上限，你将会被强制退出循环。", 
                        stop_reason=StopReason.NONE, 
                    )
                    # Append the message to the task history
                    task.update(TaskStatus.PLANNING, message)
                    
                    # Check the sub tasks
                    if len(task.sub_tasks) == 0 and not task.is_leaf:
                        # Force the task to leaf
                        task.is_leaf = True
                        # Log the task
                        self.custom_logger.warning(f"强制将任务设置为叶子任务: \n{TaskContextView(task).format()}")
                    
                    # Force the loop to break
                    break
                
                # Announce the idle thinking
                message = CompletionMessage(
                    role=MessageRole.USER, 
                    content=f"【注意】：你已经思考了 {current_thinking} 次，但是没有找到任何工具调用。在思考 {max_thinking} 次后，你将会被强制退出循环。", 
                    stop_reason=StopReason.NONE, 
                )
                # Append the message to the task history
                task.update(TaskStatus.PLANNING, message)
                
            # Check if the planning is finished
            if task.status == TaskStatus.CHECKING:
                # Double check the planning result
                if len(task.sub_tasks) == 0 and not task.is_leaf:
                    current_error += 1
                    # Check if the current error is greater than the max error
                    if current_error >= max_error:
                        # The planning is error, roll back the task status to planning
                        self.custom_logger.error(f"任务规划执行错误，当前任务没有子任务，但是当前任务不是叶子任务。由于错误累计次数超过上限，将任务强制设为叶子任务: \n{TaskContextView(task).format()}")
                        task.is_leaf = True
                        task.status = TaskStatus.CHECKING
                        # Record the error to history and announce the penalty
                        task.update(TaskStatus.PLANNING, CompletionMessage(
                            role=MessageRole.USER, 
                            content=f"任务规划错误次数累计达到上限，将任务强制设为叶子任务，你将会被惩罚。", 
                            stop_reason=StopReason.NONE, 
                        ))
                        # Force the loop to break
                        break
                    
                    # The planning is error, roll back the task status to planning
                    self.custom_logger.error(f"任务规划执行错误，当前任务没有子任务，但是当前任务不是叶子任务，回滚到规划状态: \n{TaskContextView(task).format()}")
                    task.status = TaskStatus.PLANNING
                    # Record the error to history and announce the penalty
                    task.update(TaskStatus.PLANNING, CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"任务规划错误，当前的任务没有子任务，但是当前的任务不是叶子任务，请重新执行规划拆解。" \
                            f"如果蓝图规划该任务为叶子任务，请调用 `set_as_leaf` 工具来将当前任务设置为叶子任务。累计错误次数上限为 {max_error} 次。", 
                        stop_reason=StopReason.NONE, 
                    ))
        
        return task
    
    async def run(self, task: Task) -> Task:
        """Run the PlanFlow workflow. This workflow will create the sub-tasks for the task layer by layer. 
        This will only process three kinds of status:
        
        - TaskStatus.CREATED: The sub tasks will be created.
        - TaskStatus.RUNNING: The next task in the queue will be processed.
        - TaskStatus.CANCELLED: The task will be re-planning.
        
        Args:
            task (Task):
                The task to be executed.

        Returns:
            Task:
                The task after execution.
                
        Raises:
            ValueError:
                If the status of the current task is not valid for planning.
        """
        # Check if the task is a leaf task
        if task.is_leaf:
            # Set the task status to running
            task.status = TaskStatus.CHECKING
            return task
        
        # Layer by layer traverse the task and create the sub-tasks
        queue: list[Task] = [task]
        
        while queue:
            # Create a new queue
            new_queue = []
            # Get the first task from the queue
            current_task = queue.pop(0)
            
            # Traverse the queue
            while True:
                # Check if the current task is pending or failed
                if current_task.status == TaskStatus.CREATED: 
                    if not current_task.is_leaf:
                        # Check if the max detail level is reached
                        if current_task.detail_level == 1:
                            # Force the task to leaf
                            current_task.is_leaf = True
                            current_task.status = TaskStatus.CHECKING
                            # Log the task
                            self.custom_logger.warning(f"由于达到了最大拆解层级，强制将任务设置为叶子任务: \n{TaskContextView(current_task).format()}")
                        
                        elif len(current_task.sub_tasks) == 0:
                            # Log the current task
                            self.custom_logger.info(f"规划当前任务: \n{TaskContextView(current_task).format()}")
                            # Call the plan flow to plan the task
                            current_task = await self.__layer_create(current_task)
                            
                        else:
                            # Log the current task with warning announcement about an unexpected re-planning
                            self.custom_logger.warning(f"当前任务 {current_task.question} 的子任务数量为 {len(current_task.sub_tasks)}，但仍然进入了规划状态。")
                            # Call the plan flow to plan the task
                            current_task = await self.__layer_create(current_task)
                    else:
                        # Set the task status to checking
                        current_task.status = TaskStatus.CHECKING
                        # Log the current task with a warning announcement
                        self.custom_logger.warning(f"当前任务 {current_task.question} 是叶子任务，但没有进入检查状态，已强制设置为检查状态。")
                
                elif current_task.status == TaskStatus.CHECKING:
                    if not current_task.is_leaf:
                        # Check the detail level of the current task
                        if current_task.detail_level == 2:
                            # Traverse the sub-tasks
                            for sub_task in current_task.sub_tasks.values():
                                if sub_task.status == TaskStatus.CREATED and not sub_task.is_leaf:
                                    # Force the sub-task to leaf
                                    sub_task.is_leaf = True
                                    sub_task.status = TaskStatus.CHECKING
                                    # Log the sub-task
                                    self.custom_logger.warning(f"由于达到了最大拆解层级，强制将子任务设置为叶子任务: \n{TaskContextView(sub_task).format()}")
                        
                        # Check if there is any pending sub-task
                        elif len(current_task.sub_tasks) > 0:
                            # Add the sub-task to the new queue
                            for sub_task in current_task.sub_tasks.values():
                                if sub_task.status == TaskStatus.CREATED and not sub_task.is_leaf:
                                    new_queue.append(sub_task)
                                    # Log the new sub-task
                                    self.custom_logger.info(f"将子任务添加到队列: \n{TaskContextView(sub_task).format()}")
                        
                    # Get the next task from the queue
                    try:
                        current_task = queue.pop(0)
                    except Exception as e:
                        # Queue is empty, break the loop
                        self.custom_logger.info(f"队列已空，退出循环。")
                        break
                
                # Check if all the sub-tasks are cancelled
                elif all(sub_task.status == TaskStatus.CANCELLED for sub_task in current_task.sub_tasks.values()):
                    # Log the current task
                    self.custom_logger.info(f"重新规划当前任务: \n{TaskContextView(current_task).format()}")
                    # Call the plan flow to plan the task
                    current_task = await self.__layer_create(current_task)
                    
                else:
                    # The status is not valid for planning
                    raise ValueError(f"The status of the current task is not valid for planning: {current_task.status}")
            
            # Update the queue
            queue = new_queue
            # Log the new queue
            self.custom_logger.info(f"更新队列，当前队列中包含 {len(queue)} 个任务。")
            # Log the structure of the task
            self.custom_logger.warning(f"当前任务结构: \n{TaskContextView(task).format()}")
        
        return task
    
    async def call_tool(self, ctx: Task, tool_call: ToolCallRequest, **kwargs: dict) -> ToolCallResult:
        """Call a tool to control the workflow.

        Args:
            task (Task):
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
            if tool_call.name == "create_task":
                # Call from external tools
                new_task: Task = await self.tool_functions[tool_call.name](**tool_call.args)
                
                # Create ToolCallResult
                tool_result = ToolCallResult(
                    tool_call_id=tool_call.id, 
                    is_error=False, 
                    content=f"Task {new_task.question} created.",
                )
                # Log the tool call result
                self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
                
            elif tool_call.name == "finish_planning":
                # No more orchestration is required, so we update the task status to checking
                ctx.status = TaskStatus.CHECKING
                
                # Create ToolCallResult
                tool_result = ToolCallResult(
                    tool_call_id=tool_call.id, 
                    is_error=False, 
                    content=f"Planning finished by calling the finish_planning tool.",
                )
                # Log the tool call result
                self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
                
            elif tool_call.name == "set_as_leaf":
                # Call from external tools
                await self.tool_functions[tool_call.name](**tool_call.args)
                
                # Create ToolCallResult
                tool_result = ToolCallResult(
                    tool_call_id=tool_call.id, 
                    is_error=False, 
                    content=f"Task {ctx.question} set as leaf task.",
                )
                # Log the tool call result
                self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
                
            else:
                # Error
                raise ValueError(f"Unknown tool call name: {tool_call.name} in plan flow.")
        
        except Exception as e:
            # Handle the error
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=True, 
                content=f"Tool call {tool_call.name} failed with information: {e}", 
            )
            # Log the error
            self.custom_logger.error(f"Tool call {tool_call.name} failed with information: {e}")
        finally:
            # Done the current context
            self.context = self.context.done()
        
        return tool_result
