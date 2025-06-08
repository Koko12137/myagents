import re
import sys
import traceback
from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.message import CompletionMessage, StopReason, MessageRole, ToolCallRequest, ToolCallResult
from myagents.src.interface import Agent, Task, TaskStatus, TaskStrategy, Logger
from myagents.src.envs.task import TaskContextView
from myagents.src.utils.tools import ToolView
from myagents.src.workflows.base import BaseWorkflow
from myagents.prompts.workflows.act import ACTION_PROMPT, REFLECT_PROMPT


class TaskCancelledError(Exception):
    """This is the error that is raised when the task is cancelled."""
    def __init__(self, message: str, task: Task) -> None:
        self.message = message
        self.task: Task = task
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return f"TaskCancelledError: {self.message}, task: {TaskContextView(self.task).format()}"


class ActionFlow(BaseWorkflow):
    """
    ActionFlow is a workflow for the Action framework. This workflow could not modify the orchestration 
    of sub-workflows.
    
    Attributes:
        agent (Agent):
            The agent. 
        custom_logger (Logger, defaults to logger):
            The custom logger. If not provided, the default loguru logger will be used. 
        debug (bool, defaults to False):
            The debug flag. 
        tools (list[FastMCPTool]):
            The tools of the workflow. These tools can be used to control the workflow. 
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
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> None:
        super().__init__(agent, custom_logger, debug)
        
        # Debug setting
        if self.debug:
            self.custom_logger.enable("myagents.workflows.act")
            
        # Post initialize to initialize the tools
        self.post_init()
            
    def post_init(self) -> None:
        """Post init for the ActionFlow workflow.
        """
        @self.register_tool("cancel_task")
        async def cancel_task() -> None:
            """Cancel the task. This tool is used to cancel the task if there is any error or failure in the previous 
            context.
            
            Args:
                None
                
            Returns:
                None
            """
            return 
        
        # Check the tools
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
        self.custom_logger.info(f"Tools: \n{tools_str}")
        # Check the registered tools count
        if len(self.tools) == 0:
            self.custom_logger.error(f"No tools registered for the act flow. {traceback.format_exc()}")
            raise RuntimeError("No tools registered for the act flow.")

    async def __act(self, task: Task) -> Task:
        """Take action the task. The agent will decide to call a tool or not. 
        - If the agent decides to call a tool, the tool will be called and the result will be observed. If there is any error, 
            the task status will be set to failed and the error will be logged. 
        - If the agent decides not to call a tool, the task will be answer directly by the agent.
        
        Args:
            task (Task):
                The task to be executed.
        
        Returns:
            Task:
                The task after execution.
                
        Raises:
            ValueError:
                If more than one tool calling is required. 
        """
        # 1. Observe the task
        history, current_observation = await self.agent.observe(task)
        # 2. Create a new message for the current observation
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
        message = CompletionMessage(
            role=MessageRole.USER, 
            content=ACTION_PROMPT.format(
                tools=tools_str, 
                task_context=current_observation, 
            ), 
            stop_reason=StopReason.NONE, 
        )
        
        # 3. Append Action Prompt and Call for Completion
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
            # 4. Check if there is more than one tool calling
            if len(message.tool_calls) > 1:
                # BUG: Handle the case of more than one tool calling. An unexpected error should not be raised. 
                raise ValueError("More than one tool calling is not allowed.")
            
            # 5. Get the tool call
            tool_call = message.tool_calls[0]
            
            try:
                # 6. Call the tool
                # Check if the tool is maintained by the workflow
                if tool_call.name in self.tools:
                    # Call from external tools. 
                    tool_result = await self.call_tool(task, tool_call)
                else:
                    # Call from the agent. 
                    # If there is any error caused by the tool call, the flag `is_error` will be set to True. 
                    # However, if there is any error caused by the MCP client connection, this should raise a RuntimeError. 
                    tool_result = await self.agent.call_tool(task, tool_call)
                    
                # Check the tool result
                if tool_result.is_error:
                    # Handle the error and update the task status
                    self.custom_logger.error(f"Tool call {tool_call.name} failed with information: {tool_result.content}")
                    task.status = TaskStatus.FAILED
            except RuntimeError as runtime_error:
                # Runtime error
                self.custom_logger.error(f"{runtime_error}, traceback: {traceback.format_exc()}")
                raise runtime_error
            except Exception as e:
                # May be caused by the external tool. 
                tool_result = ToolCallResult(
                    tool_call_id=tool_call.id, 
                    is_error=True, 
                    content=f"{e}, traceback: {traceback.format_exc()}"
                )
                self.custom_logger.error(f"{e}, traceback: {traceback.format_exc()}")
            
            # 7. Update the messages
            # Append for current task recording
            task.history.append(tool_result)
            # Append for agent's completion
            history.append(tool_result)
        
        # Check if the task is cancelled
        if task.status == TaskStatus.CANCELLED:
            # The task is cancelled, end the workflow
            return task
        
        # If the stop reason is not tool call, answer the task directly
        # 8. Create a new message for the reflection
        message = CompletionMessage(
            role=MessageRole.USER, 
            content=REFLECT_PROMPT, 
            stop_reason=StopReason.NONE
        )
        
        # 9. Append Reflect Prompt and Call for Completion
        # Append for current task recording
        task.history.append(message)
        # Append for agent's completion
        history.append(message)
        # Call for completion
        message: CompletionMessage = await self.agent.think(history, allow_tools=False)
        
        # 10. Record the completion message 
        # Append for current task recording
        task.history.append(message)
        # Append for agent's completion
        history.append(message)
        
        # 11. Fill the answer
        task.answer = message.content
        # 12. Update the task status
        if task.status != TaskStatus.FAILED:
            task.status = TaskStatus.FINISHED
        return task
    
    async def __act_retry(self, task: Task, max_retry_count: int = 3) -> Task:
        """Act the task with retry.
        
        Args:
            task (Task): 
                The task to act.
            max_retry_count (int, optional): 
                The maximum number of retries. Defaults to 3.
                
        Returns:
            Task: The task after the retry.
        """
        for _ in range(max_retry_count):
            # Act the task
            task = await self.__act(task)
            
            # Check the task status
            if task.status == TaskStatus.FAILED:
                # The task is still failed, then continue the retry with the next retry count
                self.custom_logger.error(f"The task is still failed and the retry count is not reached the limit: {TaskContextView(task).format()}")
                continue
            else:
                # The task is finished, then return the task
                return task
        
        # The task is still failed and the retry count is reached the limit
        # Cancel the task
        self.custom_logger.error(f"The task is still failed and the retry count is reached the limit: {TaskContextView(task).format()}")
        return task
    
    async def run(self, task: Task) -> Task:
        """Deep traverse the task and act the task.

        Args:
            task (Task): The task to act.

        Returns:
            Task: The acted task.
        """
        # Record the sub-tasks is finished
        sub_tasks_finished = []
        # Unfinished sub-tasks
        unfinished_sub_tasks = iter(task.sub_tasks.values())
        # Get the first unfinished sub-task
        sub_task = next(unfinished_sub_tasks, None)
        
        # Traverse all the sub-tasks
        while sub_task is not None:
            
            # Check if the sub-task is created, if True, then raise an error to call for re-planning
            if sub_task.status == TaskStatus.CREATED:
                # Update the context
                self.context = self.context.create_next(task=task)
                # Raise an error to call for re-planning
                raise TaskCancelledError("The task is roll back to created status due to the cancelled sub-task.", task)
                
            # Check if the sub-task is running, if True, then act the sub-task
            elif sub_task.status == TaskStatus.RUNNING:
                try:
                    # Log the sub-task
                    self.custom_logger.info(f"Acting sub-task: \n{TaskContextView(sub_task).format()}")
                    # Act the sub-task
                    sub_task = await self.run(sub_task)
                except TaskCancelledError as e:
                    # The re-planning is needed, then raise the error
                    raise e
            
            # Check if the sub-task is failed, if True, then retry the sub-task
            elif sub_task.status == TaskStatus.FAILED:
                # Retry the sub-task
                sub_task = await self.__act_retry(sub_task)
                # Check if the sub-task is still failed
                if sub_task.status == TaskStatus.FAILED:
                    # The sub-task is still failed, cancel it
                    sub_task.status = TaskStatus.CANCELLED
            
            # Check if the sub-task is cancelled, if True, set the parent task status to created and stop the traverse
            elif sub_task.status == TaskStatus.CANCELLED:
                # Check the strategy
                if task.strategy == TaskStrategy.ANY:
                    # The strategy is ANY, then continue the traverse
                    sub_task = next(unfinished_sub_tasks, None)
                else:
                    # Cancel all the sub-tasks of the parent task
                    for sub_task in task.sub_tasks.values():
                        sub_task.status = TaskStatus.CANCELLED
                    # Log the cancelled sub-task
                    self.custom_logger.info(f"All the sub-tasks are cancelled: \n{TaskContextView(task).format()}")
                    
                    # The strategy is ALL, but one of the sub-tasks is cancelled
                    task.status = TaskStatus.CREATED
                    # Log the cancelled sub-task
                    self.custom_logger.info(f"The task roll back to created status due to the cancelled sub-task: \n{TaskContextView(sub_task).format()}")
                    return task
            
            # Check if the sub-task is finished, if True, then summarize the result of the sub-task
            elif sub_task.status == TaskStatus.FINISHED:
                # The sub-task is finished, then add the sub-task to the finished list
                sub_tasks_finished.append(sub_task)
                # Log the finished sub-task
                self.custom_logger.info(f"Finished sub-task: \n{TaskContextView(sub_task).format()}")
                # Get the next unfinished sub-task
                sub_task = next(unfinished_sub_tasks, None)
            
            # ELSE, the sub-task is not created, running, failed, cancelled, or finished, it is a critical error
            else:
                # The sub-task is not created, running, failed, cancelled, or finished, then raise an error
                raise ValueError(f"The sub-task is not created, running, failed, cancelled, or finished: {TaskContextView(sub_task).format()}")
            
            # Check if continue the traverse
            if task.strategy == TaskStrategy.ANY:
                # Check if the sub-task is finished
                if len(sub_tasks_finished) > 0: 
                    # There is at least one sub-task is finished
                    self.custom_logger.info(f"There is at least one sub-task is finished: \n{TaskContextView(task).format()}")
                    break
        
        # There are some errors in the sub-tasks, then raise an error to call for re-planning
        if len(sub_tasks_finished) == 0:
            # No sub-tasks are finished, roll back the task status to created
            task.status = TaskStatus.CREATED
            # Log the roll back
            self.custom_logger.info(f"The task roll back to created status due to the errors in the sub-tasks: \n{TaskContextView(task).format()}")
            # Raise an error to call for re-planning
            raise TaskCancelledError("There are some errors in the sub-tasks.", task)
            
        # Post traverse the task
        # All the sub-tasks are finished, then call the action flow to act the task
        task = await self.__act(task)
        
        return task 

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
        """
        # Check if the tool is maintained by the workflow
        if tool_call.name not in self.tools:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, action flow allow only action tools.")
        
        # Create a new context
        self.context = self.context.create_next(task=ctx, **kwargs)
        
        if tool_call.name == "cancel_task":
            # Cancel the task
            ctx.status = TaskStatus.CANCELLED
            # Create a new result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content="The task is cancelled.",
            )
            
        # Done the current context
        self.context = self.context.done()
        
        # Return the result
        return result
