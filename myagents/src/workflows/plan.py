import re
from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.message import CompletionMessage, MessageRole, StopReason, ToolCallRequest, ToolCallResult
from myagents.src.interface import Agent, Task, TaskStatus, TaskStrategy, Logger
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.envs.task import BaseTask, TaskContextView
from myagents.src.utils.context import BaseContext
from myagents.src.utils.tools import ToolView
from myagents.prompts.workflows.plan import REASON_PROMPT, PLAN_ATTENTION_PROMPT, EXEC_PLAN_PROMPT, PLAN_SYSTEM_PROMPT


class PlanFlow(BaseWorkflow):
    """
    PlanFlow is a workflow for splitting a task into sub-tasks.
    
    Attributes:
        agent (Agent):
            The agent that will be used to plan the task. 
        debug (bool, defaults to False):
            The debug flag. 
        custom_logger (Logger, defaults to logger): 
            The custom logger. If not provided, the default loguru logger will be used. 
        tools (dict[str, FastMCPTool]):
            The orchestration tools. These tools can be used to control the workflow. 
        tool_functions (dict[str, Callable]):
            The functions of the tools provided by the workflow. These functions can be used to control the workflow. 
    """
    system_prompt: str = PLAN_SYSTEM_PROMPT
    
    agent: Agent
    debug: bool
    custom_logger: Logger
    tools: dict[str, FastMCPTool]
    tool_functions: dict[str, Callable]
    
    def __init__(
        self, 
        agent: Agent, 
        debug: bool = False, 
        custom_logger: Logger = logger, 
    ) -> None:
        super().__init__(agent, custom_logger, debug)
        
        # Create a new blueprint context
        self.context = BaseContext(key_values={})

        # Post initialize to initialize the tools
        self.post_init()
        
        # Additional tools information for tool calling
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
        
        if self.debug:
            # Check the tools
            self.custom_logger.info(f"Tools: \n{tools_str}")
            # Check the registered tools count
            if len(self.tools) == 0:
                self.custom_logger.error(f"No tools registered for the plan flow. {format_exc()}")
                raise RuntimeError("No tools registered for the plan flow.")
            
    def post_init(self) -> None:
        """Post init for the PlanFlow workflow.
        """
        @self.register_tool("create_task")
        async def create_task(question: str, description: str, is_leaf: bool) -> Task:
            """
            Create a task. This task will be added to the current task as a sub-task. 
            
            Args:
                question (str): 
                    The question of the task. 
                description (str): 
                    The description of the task. 
                is_leaf (bool):
                    Whether the task is a leaf task. If the task is a leaf task, the task will be set as running status
                    automatically. 

            Returns:
                Task: 
                    The created task. 
            """
            task = BaseTask(
                question=question, 
                description=description, 
                is_leaf=is_leaf, 
            )
            return task
        
        @self.register_tool("set_strategy")
        async def set_strategy(strategy: str) -> TaskStrategy:
            """
            Set the strategy of the parent task. If the strategy of the parent task is not correct, you can modify it.
            Otherwise, you can leave it as it is. 
            
            Args:
                strategy (str):
                    The strategy of the task. The available strategies are: 
                    - "all": The task is completed if all the sub-tasks are finished.
                    - "any": The task is completed if any of the sub-tasks are finished. 
                    
            Returns:
                TaskStrategy:
                    The strategy of the task. Then this strategy will be set as the strategy of the task. 
            """
            return TaskStrategy(strategy)
        
        @self.register_tool("set_as_leaf")
        async def set_as_leaf() -> None:
            """
            If you found that the current task is not a leaf task, but the task should be a leaf task in the blueprint, 
            you can call this tool to set the task as a leaf task.
            
            Args:
                None
                
            Returns:
                None
            """
            return None
            
        @self.register_tool("finish_planning")
        async def finish_planning() -> bool:
            """
            Finish the planning of the parent task. This will set the status of the parent task to running and the loop 
            of the workflow will be finished. Do not call this tool until you have finished all your needs correctly. 
            Once you call this tool, the loop of the workflow will be finished and you cannot modify your error.
            """
            return 
        
        @self.register_tool("raise_conflict")
        async def raise_conflict(reason: str) -> None:
            """
            Raise a conflict about the planning blueprint. This will set the task status to failed and call the planner for 
            re-planning the blueprint. 
            
            Args:
                reason (str):
                    The reason of the conflict you want to raise.
                
            Returns:
                None
            """
            self.custom_logger.error(f"Raise a conflict about the planning blueprint: {reason}")
            # Get the current task
            task = self.context.get("task")
            # Set the task status to failed
            task.status = TaskStatus.FAILED
            return 
            
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
        if len(task.history) == 0:
            # Set the system prompt
            task.history.append(CompletionMessage(
                role=MessageRole.SYSTEM, 
                content=self.system_prompt.format(blueprint=self.context.get("blueprint")), 
            ))
        
        # Set the task status to planning
        task.status = TaskStatus.PLANNING
        
        # Observe the task
        observe = await self.agent.observe(task)
        # Log the observation
        self.custom_logger.info(f"Observation: \n{observe}")
        
        # Check if the last message is a user message
        if task.history[-1].role == MessageRole.USER:
            # Append the observation to the last message
            task.history[-1].content += f"\n\nCurrent Observation: {observe}"
        else:
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
            task.history.append(message)
            
        # Call for completion
        message: CompletionMessage = await self.agent.think(task.history, allow_tools=False)
        
        # Record the completion message
        task.history.append(message)
        
        # This is used for no tool calling thinking limit.
        # If the agent is thinking more than max_thinking times, the loop will be finished.
        max_thinking = 3
        current_thinking = 0
        
        # Modify the orchestration
        while task.status == TaskStatus.PLANNING:
            # Observe the task planning history
            observe = await self.agent.observe(task)
            # Log the observation
            self.custom_logger.info(f"Observation: \n{observe}")

            # Create a new message for the current observation
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=EXEC_PLAN_PROMPT.format(
                    task_context=observe, 
                ), 
                stop_reason=StopReason.NONE, 
            )
            
            if task.history[-1].role != MessageRole.USER:
                # Append Plan Prompt and Call for Completion
                task.history.append(message)
            else:
                # Append the observation to the last content
                task.history[-1].content += f"\n\n{message.content}"
            
            # Call for completion
            message: CompletionMessage = await self.agent.think(
                task.history, 
                allow_tools=False, 
                external_tools=self.tools,
            )
            # Record the completion message
            task.history.append(message)
            
            # Extract the finish flag from the message, this will not be used for tool calling.
            finish_flag = re.search(r"<finish_flag>\n(.*)\n</finish_flag>", message.content, re.DOTALL)
            if finish_flag:
                # Extract the finish flag
                finish_flag = finish_flag.group(1)
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
                            content=f"Tool call {tool_call.name} failed with information: {e}", 
                        )
                        
                        # Handle the error and update the task status
                        self.custom_logger.error(f"Tool call {tool_call.name} failed with information: {e}")
                    except Exception as e:
                        # Handle the unexpected error
                        self.custom_logger.error(f"Tool call {tool_call.name} failed with information: {e}")
                        raise e
                    
                    # Update the messages
                    # Append for current task recording
                    task.history.append(tool_result)
            elif finish_flag:
                # Set the task status to running
                task.status = TaskStatus.RUNNING
            else:
                # Update the current thinking
                current_thinking += 1
                # Check if the current thinking is greater than the max thinking
                if current_thinking > max_thinking:
                    # No more tool calling is allowed, break the loop
                    self.custom_logger.error(f"No more tool calling is allowed, set the task status to running: \n{TaskContextView(task).format()}")
                    task.status = TaskStatus.RUNNING
                    
            # Check if the planning is finished
            if task.status == TaskStatus.RUNNING:
                # Double check the planning result
                if len(task.sub_tasks) == 0 and not task.is_leaf and current_thinking < max_thinking:
                    # The planning is error, re-plan the task
                    self.custom_logger.error(f"The planning is error, re-plan the task: \n{TaskContextView(task).format()}")
                    # Set the task status to planning
                    task.status = TaskStatus.PLANNING
                    # Record the error to history and announce the penalty
                    task.history.append(CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"任务规划错误，当前的任务没有子任务，但是当前的任务不是叶子任务，请重新执行规划拆解。以下是来自规划阶段的蓝图，你只需要执行：\n{self.context.get('blueprint')}", 
                        stop_reason=StopReason.NONE, 
                    ))
                elif len(task.sub_tasks) == 0 and not task.is_leaf and current_thinking >= max_thinking:
                    # The planning is error, re-plan the task
                    self.custom_logger.error(f"The planning is error and no more thinking is allowed, set the task status to failed: \n{TaskContextView(task).format()}")
                    # Set the task status to failed
                    task.status = TaskStatus.FAILED
        
        return task
        
    async def __reason(self, task: Task) -> Task:
        """Reason about the task and give a general orchestration plan.
        
        Args:
            task (Task): The task to reason about.

        Returns:
            Task: The task with the orchestration plan.
        """
        # Observe the task
        observe = await self.agent.observe(task)
        # Create a new message for the current observation
        message = CompletionMessage(
            role=MessageRole.USER, 
            content=REASON_PROMPT.format(task_context=observe), 
            stop_reason=StopReason.NONE, 
        )
        # Log the reason prompt
        self.custom_logger.info(f"Reason For General Orchestration and Action Plan: \n{message.content}")
        
        # Append Reason Prompt and Call for Completion
        # Append for current task recording
        task.history.append(message)
        
        # This is used for no blueprint found limit.
        # If the agent is thinking more than max_thinking times, the loop will be finished.
        max_thinking = 3
        current_thinking = 0
        
        while True:
            # Call for completion
            message: CompletionMessage = await self.agent.think(task.history, allow_tools=False)
            # Record the completion message
            # Append for current task recording
            task.history.append(message)
            
            # Extract the orchestration blueprint from the task by regular expression
            blueprint = re.search(r"<orchestration>\n(.*)\n</orchestration>", message.content, re.DOTALL)
            if blueprint:
                # Extract the blueprint from the task
                blueprint: str = blueprint.group(1)
                # Log the blueprint
                self.custom_logger.info(f"Orchestration Blueprint: \n{blueprint}")
                # Update the blueprint to the task
                self.context = self.context.create_next(blueprint=blueprint)
                break
            else:
                # Update the current thinking
                current_thinking += 1
                # Check if the current thinking is greater than the max thinking
                if current_thinking > max_thinking:
                    # No more thinking is allowed, raise an error
                    raise RuntimeError("No orchestration blueprint was found in <orchestration> tags for 3 times thinking.")
                
                # No blueprint is found, create an error message
                message = CompletionMessage(
                    role=MessageRole.USER, 
                    content="No orchestration blueprint is found in <orchestration> tags. Please try again.", 
                    stop_reason=StopReason.NONE, 
                )
                # Append the error message to the task history
                task.history.append(message)
                # Call for completion
                message: CompletionMessage = await self.agent.think(task.history, allow_tools=False)
                # Record the completion message
                task.history.append(message)
        
        return task
    
    async def run(self, task: Task) -> Task:
        """Run the PlanFlow workflow. This workflow will create the sub-tasks for the task layer by layer. 
        
        Args:
            task (Task):
                The task to be executed.

        Returns:
            Task:
                The task after execution.
        """
        # Reason about the task and create the blueprint
        task = await self.__reason(task)
        
        # Check if the task is a leaf task
        if task.is_leaf:
            # Set the task status to running
            task.status = TaskStatus.RUNNING
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
                    if len(current_task.sub_tasks) == 0 and not current_task.is_leaf:
                        # Log the current task
                        self.custom_logger.info(f"Planning current task: \n{TaskContextView(current_task).format()}")
                        # Call the plan flow to plan the task
                        current_task = await self.__layer_create(current_task)
                
                elif current_task.status == TaskStatus.RUNNING:
                    # Check if there is any pending sub-task
                    if len(current_task.sub_tasks) > 0:
                        # Add the sub-task to the new queue
                        for sub_task in current_task.sub_tasks.values():
                            if sub_task.status == TaskStatus.CREATED and not sub_task.is_leaf:
                                new_queue.append(sub_task)
                                # Log the new sub-task
                                self.custom_logger.info(f"Add sub-task to the queue: \n{TaskContextView(sub_task).format()}")
                                
                    # Get the next task from the queue
                    try:
                        current_task = queue.pop(0)
                    except Exception as e:
                        # Queue is empty, break the loop
                        break
                
                elif current_task.status == TaskStatus.FAILED:
                    # Create a break point context for the failed task
                    self.context = self.context.create_next(task=current_task, queue=queue)
                    # TODO: Roll back the task and call for re-planning the blueprint
                    raise NotImplementedError("The task is failed, but the roll back is not implemented.")
                
                # Check if all the sub-tasks are cancelled
                elif all(sub_task.status == TaskStatus.CANCELLED for sub_task in current_task.sub_tasks.values()):
                    # Log the current task
                    self.custom_logger.info(f"Planning current task: \n{TaskContextView(current_task).format()}")
                    # Call the plan flow to plan the task
                    current_task = await self.__layer_create(current_task)
            
                
            
            # Update the queue
            queue = new_queue
            # Log the new queue
            self.custom_logger.info(f"Update queue with {len(queue)} tasks")
        
        # Done the blueprint
        self.context = self.context.done()
        
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
        if tool_call.name not in self.tools:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, plan flow allow only orchestration tools.")
        
        # Create a new context
        self.context = self.context.create_next(task=ctx, **kwargs)
        
        if tool_call.name == "create_task":
            # Call from external tools
            new_task: Task = await self.tool_functions[tool_call.name](**tool_call.args)
            
            # Check if the new task is a leaf task
            if not new_task.is_leaf:
                # Set the status of the new task
                new_task.status = TaskStatus.CREATED
            else:
                # Set the status of the new task as running
                new_task.status = TaskStatus.RUNNING
                
            # Add the new task to the task
            ctx.sub_tasks[new_task.question] = new_task
            # Add the parent task to the new task
            new_task.parent = ctx
            
            # Create ToolCallResult
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"Task {new_task.question} created.",
            )
            # Log the tool call result
            self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
            
        elif tool_call.name == "set_strategy":
            # Call from external tools
            strategy: TaskStrategy = await self.tool_functions[tool_call.name](**tool_call.args)
            # Set the strategy of the new task
            ctx.strategy = strategy
            
            # Create ToolCallResult
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"Strategy {strategy.value} set.",
            )
            # Log the tool call result
            self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
            
        elif tool_call.name == "set_as_leaf":
            # Call from external tools
            await self.tool_functions[tool_call.name]()
            
            # Create ToolCallResult
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content="The task is set as leaf.",
            )
            # Log the tool call result
            self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
            
        elif tool_call.name == "finish_planning":
            # No more orchestration is required, so we update the task status to running
            ctx.status = TaskStatus.RUNNING
            
            # Create ToolCallResult
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content="Planning finished.",
            )
            # Log the tool call result
            self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
            
        else:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, plan flow allow only orchestration tools.")
        
        # Done the current context
        self.context = self.context.done()
        
        return tool_result
