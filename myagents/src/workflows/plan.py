import re
from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.message import CompletionMessage, MessageRole, StopReason, ToolCallRequest, ToolCallResult
from myagents.src.interface import Agent, Task, TaskStatus, TaskStrategy, Logger
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.envs.task import BaseTask
from myagents.prompts.workflows.plan import REASON_PROMPT, PLAN_ATTENTION_PROMPT, EXEC_PLAN_PROMPT


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
            
        # Post initialize to initialize the tools
        self.post_init()
            
    def post_init(self) -> None:
        """Post init for the PlanFlow workflow.
        """
        @self.register_tool("create_task")
        async def create_task(question: str, description: str, is_leaf: bool) -> Task:
            """
            Create a task. You should only focus on the question and description parameters without concerning any other parameters. 
            You can modify other parameters when you planning that task. This task will be added to the current task as a sub-task. 
            
            Args:
                question (str): 
                    The question of the task. 
                description (str): 
                    The description of the task. 
                is_leaf (bool):
                    Whether the task is a leaf task. If the task is a leaf task, the task will not be orchestrated by the workflow.

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
        
        if self.debug:
            # Check the tools
            self.custom_logger.info(f"Tools: {self.tools}")
            # Check the registered tools count
            if len(self.tools) == 0:
                self.custom_logger.error(f"No tools registered for the plan flow. {format_exc()}")
                raise RuntimeError("No tools registered for the plan flow.")
            
        @self.register_tool("finish_planning")
        async def finish_planning() -> bool:
            """
            Finish the planning of the parent task. This will set the status of the parent task to running and the loop 
            of the workflow will be finished. 
            """
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
        # Set the task status to planning
        task.status = TaskStatus.PLANNING
        
        # 1. Observe the task
        history, current_observation = await self.agent.observe(task)
        # 2. Create a new message for the current observation
        message = CompletionMessage(
            role=MessageRole.USER, 
            content=PLAN_ATTENTION_PROMPT.format(task_context=current_observation), 
            stop_reason=StopReason.NONE, 
        )
        
        # 3. Append Plan Prompt and Call for Completion
        # Append for current task recording
        task.history.append(message)
        # Append for agent's completion
        history.append(message)
        # Call for completion
        message: CompletionMessage = await self.agent.think(history, allow_tools=False)
        
        # 4. Record the completion message
        task.history.append(message)
        history.append(message)
        
        tools_str = "\n".join([tool.description for tool in self.tools.values()])
        
        # Modify the orchestration
        while True:
            # 1. Observe the task planning history
            history, current_observation = await self.agent.observe(task)
            # 2. Create a new message for the current observation
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=EXEC_PLAN_PROMPT.format(
                    tools=tools_str, 
                    task_context=current_observation, 
                ), 
                stop_reason=StopReason.NONE, 
            )
            
            # 3. Append Plan Prompt and Call for Completion
            # Append for current task recording
            task.history.append(message)
            # Append for agent's completion
            history.append(message)
            # Call for completion
            message: CompletionMessage = await self.agent.think(history, allow_tools=True, external_tools=self.tools)
            
            # 4. Record the completion message
            # Append for current task recording
            task.history.append(message)
            # Append for agent's completion
            history.append(message)

            # 5. Check the stop reason
            if message.stop_reason == StopReason.TOOL_CALL:
                # 6. Check if there is more than one tool calling
                if len(message.tool_calls) > 1:
                    # BUG: Handle the case of more than one tool calling. An unexpected error should not be raised. 
                    raise ValueError("More than one tool calling is not allowed.")
                
                # 7. Get the tool call
                tool_call = message.tool_calls[0]
                
                try:
                    # 8. Call the tool
                    tool_result = await self.call_tool(task, tool_call)
                except Exception as e:
                    # 9. Handle the error
                    tool_result = ToolCallResult(
                        tool_call_id=tool_call.id, 
                        is_error=True, 
                        content=e, 
                    )
                    
                    # Handle the error and update the task status
                    self.custom_logger.error(f"Tool call {tool_call.name} failed with information: {e}")
                    
                # 10. Update the messages
                # Append for current task recording
                task.history.append(tool_result)
                # Append for agent's completion
                history.append(tool_result)
                    
            # 6. Check if the planning is finished
            if task.status == TaskStatus.RUNNING:
                # No more orchestration is required, break the loop
                break
        
        return task
        
    async def __reason(self, task: Task) -> Task:
        """Reason about the task and give a general orchestration plan.
        
        Args:
            task (Task): The task to reason about.

        Returns:
            Task: The task with the orchestration plan.
        """
        # 1. Observe the task
        history, current_observation = await self.agent.observe(task)
        # 2. Create a new message for the current observation
        message = CompletionMessage(
            role=MessageRole.USER, 
            content=REASON_PROMPT.format(task_context=current_observation), 
            stop_reason=StopReason.NONE, 
        )
        # Log the reason prompt
        self.custom_logger.info(f"Reason For General Orchestration and Action Plan: \n{message.content}")
        
        # 3. Append Reason Prompt and Call for Completion
        # Append for current task recording
        task.history.append(message)
        # Append for agent's completion
        history.append(message)
        # Call for completion
        message: CompletionMessage = await self.agent.think(history, allow_tools=False)
        
        # 4. Record the completion message
        # Append for current task recording
        task.history.append(message)
        # Append for agent's completion
        history.append(message)
        
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
        # Reason about the task 
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
            
            # Traverse the queue
            for task in queue:
                # Get the first task from the queue
                current_task = queue.pop(0)
                
                # Check if the current task is pending or failed
                if current_task.status == TaskStatus.CREATED: 
                    if len(current_task.sub_tasks) == 0:
                        # Log the current task
                        self.custom_logger.info(f"Planning current task: \n{current_task.observe()}")
                        
                        # Call the plan flow to plan the task
                        current_task = await self.__layer_create(current_task)
                    
                    # Check if all the sub-tasks are cancelled
                    if all(sub_task.status == TaskStatus.CANCELLED for sub_task in current_task.sub_tasks.values()):
                        # Log the current task
                        self.custom_logger.info(f"Planning current task: \n{current_task.observe()}")
                        
                        # Call the plan flow to plan the task
                        current_task = await self.__layer_create(current_task)
            
                # Check if there is any pending sub-task
                if current_task.sub_tasks:
                    # Add the sub-task to the new queue
                    for sub_task in current_task.sub_tasks.values():
                        if sub_task.status == TaskStatus.CREATED and not sub_task.is_leaf:
                            new_queue.append(sub_task)
                            # Log the new sub-task
                            self.custom_logger.info(f"Add sub-task to the queue: \n{sub_task.observe()}")
            
            # Update the queue
            queue = new_queue
            # Log the new queue
            self.custom_logger.info(f"Update queue with {len(queue)} tasks")
                
        return task
    
    async def call_tool(self, task: Task, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool to control the workflow.

        Args:
            task (Task):
                The task to be executed.
            tool_call (ToolCallRequest):
                The tool call request.

        Returns:
            ToolCallResult:
                The tool call result.
                
        Raises:
            ValueError:
                If the tool call name is unknown. 
        """
        if tool_call.name not in self.tools:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, plan flow allow only orchestration tools.")
        
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
            task.sub_tasks[new_task.uid] = new_task
            # Add the parent task to the new task
            new_task.parent = task
            
            # Create ToolCallResult
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=new_task.observe(),
            )
            # Log the tool call result
            self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
        elif tool_call.name == "set_strategy":
            # Call from external tools
            strategy: TaskStrategy = await self.tool_functions[tool_call.name](**tool_call.args)
            # Set the strategy of the new task
            task.strategy = strategy
            
            # Create ToolCallResult
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"Strategy {strategy.value} set. Current Task Context: \n{task.observe()}",
            )
            # Log the tool call result
            self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
            
        elif tool_call.name == "finish_planning":
            # No more orchestration is required, so we update the task status to running
            task.status = TaskStatus.RUNNING
            
            # Create ToolCallResult
            tool_result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"Planning finished. Current Task Context: \n{task.observe()}",
            )
            # Log the tool call result
            self.custom_logger.info(f"Tool call {tool_call.name} finished.\n {tool_result.content}")
        else:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, plan flow allow only orchestration tools.")
        
        return tool_result
