import re
from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.interface import Agent, Task, TaskStatus, Logger, Workflow, Context
from myagents.src.message import ToolCallRequest, ToolCallResult, MessageRole, CompletionMessage, StopReason
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.workflows.act import TaskCancelledError
from myagents.src.utils.tools import ToolView
from myagents.prompts.workflows.react import REACT_THINK_PROMPT


class ReActFlow(BaseWorkflow):
    """This is use for orchestrating the ReAct flow. In this flow, raw question and global plans 
    are stored. This flow mainly controls the loop of the ReAct flow with following steps:
    
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
        **workflows: dict[str, Workflow],
    ) -> None:
        """Initialize the ReActFlow.
        
        Args:
            agent (Agent): 
                The agent that is used to orchestrate the task.
            custom_logger (Logger, optional): 
                The custom logger. If not provided, the default loguru logger will be used. 
            debug (bool, optional): 
                Whether to enable the debug mode.
        """
        super().__init__(agent, custom_logger, debug)

        # Initialize the workflows
        self.workflows = workflows
        
        # Post initialize to initialize the tools
        self.post_init()
        
    def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        
        This method will be called after the initialization of the workflow.
        """
        @self.register_tool("retry_task")
        async def retry_task() -> None:
            """Retry the task.
            
            Args:
                None
            
            Returns: 
                None
                    The task is retried.
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
            """Finish the planning.
            
            Args:
                None
            
            Returns:
                None
            """
            task = self.context.get("task")
            task.status = TaskStatus.RUNNING
            return
        
        @self.register_tool("error_in_plan")
        async def error_in_plan(question_path: list[str], error: str) -> None:
            """Raise an error in the plan. If you found any error in the planning, you can use this tool to raise 
            a error. The error will be recorded in the task and the task will be re-planned. 
            
            Args:
                question_path (list[str]):
                    The question path of the error task. The path will be used to find the sub-task that has 
                    the same question. The first element should be the question of the root task, and the next 
                    one should be the next layer navigating to the sub-task. 
                error (str):
                    The error information of the error task.
            
            Returns:
                None
                
            Raises:
                KeyError:
                    If the question path is not found in the task.
            """
            task = self.context.get("task")
            
            # Traverse the question path and find the sub-task that has the same question
            for question in question_path:
                task = task.sub_tasks[question]
            
            # Update the task status
            task.status = TaskStatus.CREATED
            # Cancel all the sub-tasks of the task
            for sub_task in task.sub_tasks.values():
                sub_task.status = TaskStatus.CANCELLED
                sub_task.answer = error
                
            # Append the error to the task history
            task.history.append(CompletionMessage(
                role=MessageRole.USER, 
                content=f"Error in the planning: {error}", 
                stop_reason=StopReason.NONE, 
            ))
            
            # Re plan the task
            await self.workflows["plan"].run(task)
            return
        
        # Check the registered tools count
        if len(self.tools) == 0:
            self.custom_logger.error(f"No tools registered for the react flow. {format_exc()}")
            raise RuntimeError("No tools registered for the react flow.")
        
        # Check the tools
        tool_str = ""
        for tool in self.tools.values():
            tool_str += f"{ToolView(tool).format()}\n"
        self.custom_logger.info(f"Tools: \n{tool_str}")
        
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
        while True:
            current = await self.workflows["plan"].run(env)
            
            # Check if the planning flow is finished properly
            # Observe the task
            observe = await self.agent.observe(current)
            # Think about the current task
            message = CompletionMessage(
                role=MessageRole.USER, 
                content=REACT_THINK_PROMPT.format(task_context=observe), 
                stop_reason=StopReason.NONE, 
            )
            # Log the react think prompt
            self.custom_logger.info(f"ReAct Flow thinking about the current task: \n{message.content}")
            # Append the message to the task history
            current.history.append(message)
            
            # Call for completion
            message: CompletionMessage = await self.agent.think(
                current.history, 
                allow_tools=False, 
                external_tools=self.tools, 
            )
            # Append the message to the task history
            current.history.append(message)
            
            # Reset the current thinking
            current_thinking = 0
            max_thinking = 3
            
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
                # Traverse all the tool calls
                for tool_call in message.tool_calls:
                    # Reset the current thinking due to tool calling
                    current_thinking = 0
                    
                    try:
                        # Call the tool
                        result = await self.call_tool(env, tool_call)
                    except Exception as e:
                        # Handle the unexpected error
                        self.custom_logger.error(f"Tool call {tool_call.name} failed with information: {e}")
                        raise e

                    # Append the result to the task history
                    current.history.append(result)
            elif finish_flag:
                # Set the task status to running
                current.status = TaskStatus.RUNNING
                # Break the loop
                break
            else:
                # Update the current thinking due to no tool calling
                current_thinking += 1
                # Check if the current thinking is greater than the max thinking
                if current_thinking > max_thinking:
                    # No more tool calling is allowed, break the loop
                    break
                
            # Check if the planning flow is finished properly
            if current.status == TaskStatus.RUNNING:
                break

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
        # Plan the current task
        env = await self.__plan(env)
        
        while env.status != TaskStatus.FINISHED: 
            # Log the current task
            self.custom_logger.info(f"ReAct Flow running with current task: \n{env.question}")
            
            try:
                # Act the task from root
                await self.workflows["action"].run(env)
            except TaskCancelledError as e:
                # The re-planning is needed, continue and re-plan the task
                current = self.workflows["action"].context.get("task")
                current = await self.__plan(current)
                # Resume the context of the action flow
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
