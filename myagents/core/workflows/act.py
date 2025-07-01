import re
import traceback
from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.message import CompletionMessage, StopReason, MessageRole, ToolCallRequest, ToolCallResult
from myagents.core.interface import Agent, Task, TaskStatus, Logger, Workflow
from myagents.core.envs.task import TaskContextView, TaskAnswerView
from myagents.core.utils.tools import ToolView
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.act import ACTION_SYSTEM_PROMPT, ACTION_PROMPT, REFLECT_PROMPT, RETRY_PROMPT


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
    system_prompt: str = ACTION_SYSTEM_PROMPT
    
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
        workflows: dict[str, Workflow] = {}, 
        *args, 
        **kwargs,
    ) -> None:
        super().__init__(agent=agent, custom_logger=custom_logger, debug=debug, workflows=workflows, *args, **kwargs)
            
        # Post initialize to initialize the tools
        self.post_init()
            
    def post_init(self) -> None:
        """Post init for the ActionFlow workflow.
        """
        @self.register_tool("cancel_task")
        async def cancel_task(reason: str) -> None:
            """
            取消当前任务。如果你在当前任务中发现了致命错误或失败，你可以使用这个工具来取消任务。
            
            Args:
                reason (str):
                    取消任务的原因。
                
            Returns:
                None
            """
            # Set the task status to cancelled
            task = self.context.get("task")
            task.answer = reason
            task.status = TaskStatus.CANCELLED
        
        @self.register_tool("finish_task")
        async def finish_task() -> None:
            """
            完成当前任务。当你认为当前任务已经完成时，你可以选择以下任一方式来完成任务：
            
            - 调用这个工具来完成任务。
            - 在消息中设置完成标志为 True，并且不要调用这个工具。 
            
            Args:
                None
                
            Returns:
                None
            """
            # Set the task status to finished
            task = self.context.get("task")
            task.status = TaskStatus.FINISHED
            return True
        
        # Check the tools
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
        self.custom_logger.debug(f"工具: \n{tools_str}")
        # Check the registered tools count
        if len(self.tools) == 0:
            self.custom_logger.error(f"注册工具失败: \n{traceback.format_exc()}")
            raise RuntimeError("No tools registered for the act flow.")

    async def __act(self, task: Task) -> Task:
        """
        Take action the task. The agent will decide to call a tool or not. 
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
        # Get the blueprint from the context
        blueprint = self.context.get("blueprint")
        # Get the task from the context
        task = self.context.get("task")
        # Convert to task answer view
        task_result = TaskAnswerView(task).format()
        
        # Check if the last message is a user message
        if len(task.history[TaskStatus.RUNNING]) > 0 and task.history[TaskStatus.RUNNING][-1].role == MessageRole.USER:
            # Append the blueprint to the last message
            task.history[TaskStatus.RUNNING][-1].content += f"\n\n{ACTION_SYSTEM_PROMPT.format(blueprint=blueprint, task_result=task_result)}"
        else:
            # Create a new message for the current blueprint
            message = CompletionMessage(
                role=MessageRole.SYSTEM, 
                content=ACTION_SYSTEM_PROMPT.format(blueprint=blueprint, task_result=task_result), 
                stop_reason=StopReason.NONE, 
            )
            # Append the blueprint to the last message
            task.update(TaskStatus.RUNNING, message)
        
        # Create a new message for the current observation
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
            
        # This is used for no tool calling thinking limit.
        # If the agent is thinking more than max_thinking times, the loop will be finished.
        max_thinking = 3
        current_thinking = 0
        
        while task.status == TaskStatus.RUNNING:
            # Observe the task
            observe = await self.agent.observe(task)
            # Log the observation
            self.custom_logger.info(f"当前观察: \n{observe}")
            # Create a new message for the action prompt
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=ACTION_PROMPT.format(task_context=observe), 
                stop_reason=StopReason.NONE, 
            )
            # Append Action Prompt and Call for Completion
            task.update(TaskStatus.RUNNING, message)
            # Call for completion
            message: CompletionMessage = await self.agent.think(task.history[TaskStatus.RUNNING], allow_tools=True)
            # Log the message
            self.custom_logger.info(f"模型回复: \n{message.content}")
            # Record the completion message
            task.update(TaskStatus.RUNNING, message)
            
            # Check the stop reason
            if message.stop_reason == StopReason.TOOL_CALL:
                # Reset the current thinking
                current_thinking = 0
                
                # Traverse all the tool calls
                for tool_call in message.tool_calls:
                    try:
                        # Call from the agent. 
                        # If there is any error caused by the tool call, the flag `is_error` will be set to True. 
                        # However, if there is any error caused by the MCP client connection, this should raise a RuntimeError. 
                        tool_result = await self.agent.call_tool(task, tool_call)
                            
                        # Check the tool result
                        if tool_result.is_error:
                            # Handle the error and update the task status
                            self.custom_logger.warning(f"工具调用 {tool_call.name} 失败: \n{tool_result.content}")
                            # Set the task status to failed and call for retry
                            task.status = TaskStatus.ERROR
                            # Force the loop to break
                            break
                    
                    except RuntimeError as runtime_error:
                        # Runtime error
                        self.custom_logger.error(f"工具调用中出现了未知异常: \n{runtime_error}, traceback: {traceback.format_exc()}")
                        raise runtime_error
                    
                    except Exception as e:
                        # May be caused by the workflow tools. 
                        tool_result = ToolCallResult(
                            tool_call_id=tool_call.id, 
                            is_error=True, 
                            content=e
                        )
                        self.custom_logger.warning(f"工具调用 {tool_call.name} 失败: \n{tool_result.content}")
                    
                    # Update the messages
                    task.update(TaskStatus.RUNNING, tool_result)

            else:
                # Update the current thinking
                current_thinking += 1
            
                # Check if the current thinking is reached the limit
                if current_thinking >= max_thinking:
                    # The current thinking is reached the limit, end the workflow
                    self.custom_logger.error(f"当前任务执行已达到最大思考次数，任务执行结束: \n{TaskContextView(task).format()}")
                    task.status = TaskStatus.FINISHED
                    # Announce the current thinking
                    message = CompletionMessage(
                        role=MessageRole.USER, 
                        content=f"当前思考次数: {current_thinking}/{max_thinking}, 达到最大思考次数，任务执行结束，如果任务结束时没有提供答案，则你会被惩罚。", 
                        stop_reason=StopReason.NONE
                    )
                    task.update(TaskStatus.RUNNING, message)
                    # Force the loop to break
                    break
                
                # Announce the current thinking
                message = CompletionMessage(
                    role=MessageRole.USER, 
                    content=f"当前思考次数: {current_thinking}/{max_thinking}，如果任务结束时没有提供答案，则你会被惩罚。", 
                    stop_reason=StopReason.NONE
                )
                task.update(TaskStatus.RUNNING, message)
            
            # Observe the task
            observe = await self.agent.observe(task)
            # Log the observation
            self.custom_logger.info(f"当前观察: \n{observe}")
            # If the stop reason is not tool call, answer the task directly
            # Create a new message for the reflection
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=REFLECT_PROMPT.format(
                    tools=tools_str, 
                    task_context=observe, 
                ), 
                stop_reason=StopReason.NONE
            )
            # Append Reflect Prompt and Call for Completion
            task.update(TaskStatus.RUNNING, message)
            # Call for completion
            message: CompletionMessage = await self.agent.think(
                task.history[TaskStatus.RUNNING], 
                allow_tools=False, 
                external_tools=self.tools,
            )
            # Log the message
            self.custom_logger.info(f"模型回复: \n{message.content}")
            # Record the completion message 
            task.update(TaskStatus.RUNNING, message)
            
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
            
            # Extract the final output from the message
            final_output = extract_by_label(message.content, "final_output", "final answer", "output", "answer")
            if final_output != "":
                # Set the answer of the task
                task.answer = final_output
            
            # Check if the task is finished
            if finish_flag:
                # Set the task status to finished if the task is running else keep the status
                task.status = TaskStatus.FINISHED if task.status == TaskStatus.RUNNING else task.status
                # Force the loop to break
                break
                
            elif message.stop_reason == StopReason.TOOL_CALL:
                # Reset the current thinking
                current_thinking = 0
                # Get the first tool call
                tool_call = message.tool_calls[0]
                
                # Call the tool
                tool_result = await self.call_tool(task, tool_call)
                
                # Update the messages
                task.update(TaskStatus.RUNNING, tool_result)
            
            # Check if the task is cancelled
            if task.status == TaskStatus.CANCELLED:
                # The task is cancelled, end the workflow
                break
        
        # Set the answer of the task
        if not task.answer: 
            task.answer = "任务执行结束，但未提供答案，执行可能存在未知错误。"
            
        # Log the answer
        self.custom_logger.info(f"任务执行结束: \n{TaskContextView(task).format()}")
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
        for i in range(max_retry_count):
            # Add a penalty announcement for error action.
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=RETRY_PROMPT.format(i=i + 1, max_retry_count=max_retry_count), 
                stop_reason=StopReason.NONE,
            )
            task.update(TaskStatus.RUNNING, message)
            # Act the task
            task = await self.__act(task)
            
            # Check the task status
            if task.status == TaskStatus.ERROR:
                # The task is still failed, then continue the retry with the next retry count
                self.custom_logger.warning(f"任务执行失败，重试中: \n{TaskContextView(task).format()}")
                continue
            else:
                # The task is finished or cancelled, then return the task
                return task
        
        # The task is still failed and the retry count is reached the limit
        # Cancel the task
        self.custom_logger.error(f"任务执行失败，重试次数达到上限: \n{TaskContextView(task).format()}")
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
                # Append the error information to the planning history of the parent task
                error_message = CompletionMessage(
                    role=MessageRole.USER, 
                    content=f"子任务 {sub_task.question} 执行失败: {sub_task.answer}，所有的未执行或执行失败的子任务将被取消并删除。", 
                    stop_reason=StopReason.NONE,
                )
                task.answer = error_message.content
                # Clean up the answer of the sub-task
                sub_task.answer = ""
                # Update the planning history of the parent task
                task.update(TaskStatus.PLANNING, error_message)
                # Update the created history of the parent task
                task.update(TaskStatus.CREATED, error_message)
                # Log the error message
                self.custom_logger.error(f"已记录子任务取消信息到当前任务的`CREATED`和`PLANNING`历史: \n{error_message.content}")
                
                # Remove the sub-tasks from the parent task except the finished ones and the created ones
                for sub_task in task.sub_tasks.values():
                    if sub_task.status not in [TaskStatus.FINISHED, TaskStatus.CREATED]:
                        del task.sub_tasks[sub_task.question]
                
                # Log the cancelled sub-task
                self.custom_logger.error(f"取消所有未执行或执行失败的子任务: \n{TaskContextView(task).format()}")
                # Set the parent task status to created
                task.status = TaskStatus.CREATED
                # Log the roll back status
                self.custom_logger.error(f"任务执行失败，回滚到创建状态: \n{TaskContextView(task).format()}")
                return task
                
            # Check if the sub-task is running, if True, then act the sub-task
            elif sub_task.status == TaskStatus.RUNNING:
                # Log the sub-task
                self.custom_logger.info(f"执行子任务: \n{TaskContextView(sub_task).format()}")
                # Act the sub-task
                sub_task = await self.run(sub_task)
            
            # Check if the sub-task is failed, if True, then retry the sub-task
            elif sub_task.status == TaskStatus.ERROR:
                # Retry the sub-task
                sub_task = await self.__act_retry(sub_task)
                # Check if the sub-task is still failed
                if sub_task.status == TaskStatus.ERROR:
                    # The sub-task is still failed, cancel it
                    sub_task.status = TaskStatus.CANCELLED
            
            # Check if the sub-task is cancelled, if True, set the parent task status to created and stop the traverse
            elif sub_task.status == TaskStatus.CANCELLED:
                # Append the error information to the planning history of the parent task
                error_message = CompletionMessage(
                    role=MessageRole.USER, 
                    content=f"子任务 {sub_task.question} 执行失败: {sub_task.answer}，所有的未执行或执行失败的子任务将被取消并删除。", 
                    stop_reason=StopReason.NONE,
                )
                task.answer = error_message.content
                # Update the planning history of the parent task
                task.update(TaskStatus.PLANNING, error_message)
                # Update the created history of the parent task
                task.update(TaskStatus.CREATED, error_message)
                # Log the error message
                self.custom_logger.error(f"已记录子任务取消信息到当前任务的`CREATED`和`PLANNING`历史: \n{error_message.content}")
                
                # Cancel all the sub-tasks of the parent task except the finished ones
                for sub_task in task.sub_tasks.values():
                    if sub_task.status != TaskStatus.FINISHED:
                        del task.sub_tasks[sub_task.question]
                    
                # Log the cancelled sub-task
                self.custom_logger.error(f"取消所有未执行或执行失败的子任务: \n{TaskContextView(task).format()}")
                # Set the parent task status to created
                task.status = TaskStatus.CREATED
                # Log the roll back status
                self.custom_logger.error(f"任务执行失败，回滚到创建状态: \n{TaskContextView(task).format()}")
                return task
            
            # Check if the sub-task is finished, if True, then summarize the result of the sub-task
            elif sub_task.status == TaskStatus.FINISHED:
                # The sub-task is finished, then add the sub-task to the finished list
                sub_tasks_finished.append(sub_task)
                # Log the finished sub-task
                self.custom_logger.info(f"子任务执行完成: \n{TaskContextView(sub_task).format()}")
                # Get the next unfinished sub-task
                sub_task = next(unfinished_sub_tasks, None)
            
            # ELSE, the sub-task is not created, running, failed, cancelled, or finished, it is a critical error
            else:
                # The sub-task is not created, running, failed, cancelled, or finished, then raise an error
                raise ValueError(f"The status of the sub-task is invalid in action flow: {sub_task.status}")
        
        # There are some errors in the sub-tasks, then raise an error to call for re-planning
        if len(sub_tasks_finished) == 0 and not task.is_leaf:
            # No sub-tasks are finished, the task is cancelled
            task.status = TaskStatus.CANCELLED
            # Log the roll back
            self.custom_logger.error(f"任务执行失败: \n{TaskContextView(task).format()}")
            
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
        # Log the tool call
        self.custom_logger.info(f"Tool calling {tool_call.name} with arguments: {tool_call.args}.")
        
        # Create a new context
        self.context = self.context.create_next(task=ctx, **kwargs)
        
        try:
            if tool_call.name == "cancel_task":
                # Cancel the task
                await self.tool_functions[tool_call.name](**tool_call.args)
                # Create a new result
                result = ToolCallResult(
                    tool_call_id=tool_call.id, 
                    is_error=False, 
                    content=f"任务已取消，原因: {ctx.answer}",
                )
                # Log the result
                self.custom_logger.info(f"工具调用 {tool_call.name} 完成: \n{result.content}")
                
            elif tool_call.name == "finish_task":
                # Finish the task
                ctx.status = TaskStatus.FINISHED
                # Create a new result
                result = ToolCallResult(
                    tool_call_id=tool_call.id, 
                    is_error=False, 
                    content="任务已执行完成。",
                )
                # Log the result
                self.custom_logger.info(f"工具调用 {tool_call.name} 完成: \n{result.content}")
                
            else:
                # Error
                raise ValueError(f"Unknown tool call name: {tool_call.name} in action flow.")
                
        except Exception as e:
            # Handle the error
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=True, 
                content=f"工具调用 {tool_call.name} 失败: \n{e}", 
            )
            # Log the error
            self.custom_logger.error(f"工具调用 {tool_call.name} 失败: \n{e}")
        finally:
            # Done the current context
            self.context = self.context.done()
        
        return result
