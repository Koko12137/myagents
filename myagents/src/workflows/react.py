from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.interface import Agent, MaxStepsError, Task, TaskStatus, TaskStrategy, Logger, Workflow, OrchestratedFlows
from myagents.src.message import ToolCallRequest, ToolCallResult, MessageRole, CompletionMessage, StopReason
from myagents.src.envs.task import TaskContextView
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.workflows.plan import PlanFlow
from myagents.src.workflows.act import ActionFlow
from myagents.prompts.workflows.react import REASON_PROMPT, ACTION_FAILURE_PROMPT, SUMMARY_PROMPT


class ReActFlow(BaseWorkflow, OrchestratedFlows):
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
    tools: dict[str, FastMCPTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]

    def __init__(
        self, 
        agent: Agent, 
        plan_agent: Agent, 
        action_agent: Agent, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> None:
        """Initialize the ReActFlow.
        
        Args:
            agent (Agent): 
                The agent that is used to orchestrate the task.
            plan_agent (Agent): 
                The agent that is used to plan the task.
            action_agent (Agent): 
                The agent that is used to act the task.
            custom_logger (Logger, optional): 
                The custom logger. If not provided, the default loguru logger will be used. 
            debug (bool, optional): 
                Whether to enable the debug mode.
        """
        super().__init__(agent, custom_logger, debug)
        
        # Acting Task status
        self.current_task: Task = None
        self.break_point: Task = None

        # Initialize the workflows
        self.workflows = {
            "plan": PlanFlow(agent=plan_agent, debug=debug, custom_logger=custom_logger),
            "action": ActionFlow(agent=action_agent, debug=debug, custom_logger=custom_logger),
        }
        
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
            task = self.current_task
            
            # Find the first one of the sub-tasks that is failed
            for sub_task in task.sub_tasks.values():
                if sub_task.status == TaskStatus.FAILED:
                    # Retry the sub-task
                    sub_task = await self.__act(sub_task)
                    return
            
            # No failed sub-task, then update the task status
            task.status = TaskStatus.RUNNING
            return
        
        # Check the tools
        tool_str = ""
        for tool in self.tools.values():
            tool_str += f"{tool.name}: \n{tool.description}\n"
        self.custom_logger.info(f"Tools: \n{tool_str}")
        # Check the registered tools count
        if len(self.tools) == 0:
            self.custom_logger.error(f"No tools registered for the react flow. {format_exc()}")
            raise RuntimeError("No tools registered for the react flow.")
        
    async def __reason(self, task: Task) -> Task:
        """Reason about the task and give a general orchestration and action plan.
        
        Args:
            task (Task): The task to reason about.

        Returns:
            Task: The task with the orchestration and action plan.
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
        
    async def __orchestrate(self, task: Task) -> Task:
        """Layer by layer traverse the task and orchestrate the task.

        Args:
            task (Task): The task to orchestrate.

        Returns:
            Task: The orchestrated task.
        """
        queue = [task]
        
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
                        self.custom_logger.info(f"Orchestrating current task: \n{current_task.observe()}")
                        
                        # Call the plan flow to plan the task
                        current_task = await self.workflows["plan"].run(current_task)
                    
                    # Check if all the sub-tasks are cancelled
                    if all(sub_task.status == TaskStatus.CANCELLED for sub_task in current_task.sub_tasks.values()):
                        # Log the current task
                        self.custom_logger.info(f"Orchestrating current task: \n{current_task.observe()}")
                        
                        # Call the plan flow to plan the task
                        current_task = await self.workflows["plan"].run(current_task)
            
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

    async def __act(self, task: Task) -> Task:
        """Deep traverse the task and act the task.

        Args:
            task (Task): The task to act.

        Returns:
            Task: The acted task.
        """
        # Update the current task
        self.current_task = task
        
        # Record the sub-tasks is finished
        sub_tasks_finished = []
        
        # Traverse all the sub-tasks
        for sub_task in task.sub_tasks.values():
            # Check if continue the traverse
            if task.strategy == TaskStrategy.ANY:
                # Check if the sub-task is finished
                if len(sub_tasks_finished) > 0: 
                    # There is at least one sub-task is finished
                    self.custom_logger.info(f"There is at least one sub-task is finished: \n{TaskContextView(task).format()}")
                    break
                
            # Check if the sub-task is running
            if sub_task.status == TaskStatus.RUNNING:
                # Log the sub-task
                self.custom_logger.info(f"Acting sub-task: \n{TaskContextView(sub_task).format()}")
                # Act the sub-task
                sub_task = await self.__act(sub_task)
                # Resume the current task
                self.current_task = task
                
                # Check if the sub-task is failed
                if sub_task.status == TaskStatus.FAILED:
                    # Check the task strategy
                    if task.strategy == TaskStrategy.ALL:
                        # Retry up to 3 times
                        for retry_count in range(3):
                            # Update the failure message
                            task.answer = sub_task.answer
                            
                            # Reason about the failure
                            # 1. Observe the failure
                            history, current_observation = await self.agent.observe(task)
                            # 2. Create a new message for the current observation
                            message = CompletionMessage(
                                role=MessageRole.USER, 
                                content=ACTION_FAILURE_PROMPT.format(failure=current_observation), 
                                stop_reason=StopReason.NONE, 
                            )
                            
                            # 3. Append Reason Prompt and Call for Completion
                            # Append for current task recording
                            task.history.append(message)
                            # Append for agent's completion
                            history.append(message)
                            # Call for completion
                            message: CompletionMessage = await self.agent.think(
                                history, 
                                allow_tools=True, 
                                tools=self.tools,
                            )
                            
                            # 4. Record the completion message
                            # Append for current task recording
                            task.history.append(message)
                            # Append for agent's completion
                            history.append(message)
                            
                            # 5. Check the stop reason
                            if message.stop_reason == StopReason.TOOL_CALL:
                                # The stop reason is tool call, then call the tool
                                tool_call = message.tool_calls[0]
                                tool_result = await self.call_tool(tool_call)
                                # Record the tool result
                                task.history.append(tool_result)
                                history.append(tool_result)
                            else:
                                # No retry or cancel, then update the task status
                                sub_task.status = TaskStatus.CANCELLED
                                break
                            
                            # 6. Check the retry result
                            if sub_task.status == TaskStatus.FAILED:
                                # The sub-task is still failed, then continue the retry
                                continue
                            else:
                                # The sub-task is finished, then add the sub-task to the finished list
                                sub_tasks_finished.append(sub_task)
                                break
                        
                        # Check if the retry count is equals to 3
                        if retry_count == 3 and sub_task.status == TaskStatus.FAILED:
                            # The retry count is equals to 3, then cancel the task
                            task.status = TaskStatus.CANCELLED
                            return task
                        
                elif sub_task.status == TaskStatus.CREATED:
                    # This may be a sub-task is failed and then the parent needs to be re-orchestrated
                    task.status = TaskStatus.CREATED
                    # Log the step back to the parent task
                    self.custom_logger.info(f"Step back to the parent task: \n{TaskContextView(task).format()}")
                    return task
                        
                if sub_task.status == TaskStatus.CANCELLED:
                    # The sub-task is cancelled, then update the task status to pending, and it will be re-orchestrated
                    task.status = TaskStatus.CREATED
                    # Log the cancelled sub-task
                    self.custom_logger.info(f"Cancelled sub-task: \n{TaskContextView(sub_task).format()}")
                    # Cancel the sub-task
                    for sub_task in task.sub_tasks.values():
                        sub_task.status = TaskStatus.CANCELLED
                    # Log the cancelled sub-task
                    self.custom_logger.info(f"All the sub-tasks are cancelled: \n{TaskContextView(task).format()}")
                    return task
                else:
                    # The sub-task is finished, then add the sub-task to the finished list
                    sub_tasks_finished.append(sub_task)
                    # Log the finished sub-task
                    self.custom_logger.info(f"Finished sub-task: \n{TaskContextView(sub_task).format()}")
                    # Summarize the result of the task
                    # 1. Observe the result
                    history, current_observation = await self.agent.observe(sub_task)
                    # 2. Create a new message for the current observation
                    message = CompletionMessage(
                        role=MessageRole.USER, 
                        content=SUMMARY_PROMPT.format(sub_task=current_observation), 
                        stop_reason=StopReason.NONE, 
                    )
                    # Log the summary message
                    self.custom_logger.info(f"Summary Prompt: \n{message.content}")
                    
                    # 3. Append Summary Prompt and Call for Completion
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
                    # Log the completion message
                    self.custom_logger.info(f"Summary Result: \n{message.content}")
                    
                    return task
        
        # Post traverse the task
        # All the sub-tasks are finished, then call the action flow to act the task
        task = await self.workflows["action"].run(task)
        
        return task 
        
    async def run(self, env: Task) -> Task:
        """Orchestrate the task and act the task.

        Args:
            env (Task): 
                The task to orchestrate and act.

        Returns:
            Task: 
                The task after orchestrating and acting.
        """
        self.current_task = env
        
        # Reason about the task
        env = await self.__reason(env)
        
        while env.status != TaskStatus.FINISHED: 
            # Log the current task
            self.custom_logger.info(f"ReAct Flow running with current task: \n{TaskContextView(env).format()}")
            
            # Orchestrate the current task
            current_task = await self.__orchestrate(self.current_task)
            
            try:
                # Act the task from root
                current_task = await self.__act(env)
            except MaxStepsError as e:
                # The max steps error is raised, then update the task status to cancelled
                self.custom_logger.error(f"Max steps error: {e}")
                # Request the user to reset the step counter
                reset = input(f"The limit of auto steps is reached. Do you want to reset the step counter with limit {e.limit} steps? (y/n)")
                if reset == "y":
                    # Reset the step counter and continue the loop
                    self.agent.step_counter.reset()
                    continue
                else:
                    # Stop the loop
                    raise e
            except Exception as e:
                # Unexpected error
                self.custom_logger.error(f"Unexpected error: {e}")
                raise e
            
        return env
    
    async def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """ReAct flow do not provide any external tools."""
        if tool_call.name == "retry_task":
            await self.tool_functions["retry_task"]()
            content = "The task is retried."
        else:
            raise NotImplementedError("ReAct flow do not provide any external tools.")

        return ToolCallResult(
            role=MessageRole.TOOL,
            tool_call_id=tool_call.id,
            content=content,
        )
