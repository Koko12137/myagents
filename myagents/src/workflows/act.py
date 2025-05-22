import sys
import traceback
from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.message import CompletionMessage, StopReason, MessageRole, ToolCallRequest, ToolCallResult
from myagents.src.interface import Agent, Task, TaskStatus, Logger
from myagents.src.workflows.base import BaseWorkflow
from myagents.prompts.workflows.act import ACTION_PROMPT
from myagents.prompts.workflows.reflect import REFLECT_PROMPT


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
        self.custom_logger.info(f"Tools: {self.tools}")
        # Check the registered tools count
        if len(self.tools) == 0:
            self.custom_logger.error(f"No tools registered for the act flow. {traceback.format_exc()}")
            raise RuntimeError("No tools registered for the act flow.")

    async def run(self, task: Task) -> Task:
        """Run the ActionFlow workflow. In this workflow, the agent will decide to call a tool or not. 
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
        tools_str = "\n".join([tool.description for tool in self.tools.values()])
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
                    tool_result = await self.agent.call_tool(tool_call)
                    
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
        task.status = TaskStatus.FINISHED
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
        """
        # Check if the tool is maintained by the workflow
        if tool_call.name not in self.tools:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, action flow allow only action tools.")
        
        if tool_call.name == "cancel_task":
            # Cancel the task
            task.status = TaskStatus.CANCELLED
            return ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"The task is cancelled. Current Task Context: \n{task.observe()}",
            )
