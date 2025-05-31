from typing import Callable
from traceback import format_exc

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.interface import Agent, Task, TaskStatus, Logger, Workflow, OrchestratedFlows
from myagents.src.message import ToolCallRequest, ToolCallResult, MessageRole
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.workflows.plan import PlanFlow
from myagents.src.workflows.act import ActionFlow, TaskCancelledError


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
        
        # Plan the current task
        env = await self.workflows["plan"].run(self.current_task)
        
        while env.status != TaskStatus.FINISHED: 
            # Log the current task
            self.custom_logger.info(f"ReAct Flow running with current task: \n{self.current_task.observe()}")
            
            try:
                # Act the task from root
                await self.workflows["action"].run(self.current_task)
            except TaskCancelledError as e:
                # The re-planning is needed, continue and re-plan the task
                await self.workflows["plan"].run(self.current_task)
                continue
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
