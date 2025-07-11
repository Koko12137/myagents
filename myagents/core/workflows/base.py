import asyncio
from abc import abstractmethod
from typing import Callable, Any

from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Workflow, Stateful
from myagents.core.messages import ToolCallResult, ToolCallRequest
from myagents.core.utils.context import BaseContext
from myagents.core.tools_mixin import ToolsMixin


class BaseWorkflow(Workflow, ToolsMixin):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to work with the workflow.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        context (BaseContext):
            The context of the tool call.
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
    """
    profile: str
    agent: Agent
    prompts: dict[str, str]
    # Tools Mixin
    context: BaseContext
    tools: dict[str, FastMcpTool]
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the BaseWorkflow. This will run the post_init method automatically. 
        
        Args:
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the workflow components
        self.profile = None
        self.agent = None
        self.prompts = {}
        
        # Initialize the tools and context
        self.tools = {}
        self.context = BaseContext()

        # Post initialize
        try:
            loop = asyncio.get_running_loop()
            # 如果已经有事件循环，创建任务并等待完成
            task = loop.create_task(self.post_init())
            loop.run_until_complete(task)  # 这行在已运行的loop下会报错
        except RuntimeError:
            # 没有事件循环，直接新建一个
            asyncio.run(self.post_init())
        
    async def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        
        This method will be called after the initialization of the workflow.
        """
        # Register the finish tool
        @self.register_tool("finish_workflow")
        async def finish_workflow() -> ToolCallResult:
            """
            完成当前任务，使用这个工具来结束工作流。
            
            Args:
                None
            
            Returns:
                ToolCallResult:
                    The tool call result.
            """
            # Get the target
            target: Stateful = self.context.get("target")
            # Get the tool call
            tool_call: ToolCallRequest = self.context.get("tool_call")
            # Set the task status to finished
            target.to_finished()
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"任务已设置为 {target.get_status().value} 状态。",
            )
            return result
            
    def register_agent(self, agent: Agent) -> None:
        """Register an agent to the workflow.
        
        Args:
            agent (Agent):
                The agent to register.
        """
        # Check if the agent is registered
        if self.agent is not None:
            return 
        
        # Register the agent to the workflow
        self.agent = agent
    
    @abstractmethod
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs, 
    ) -> Stateful:
        """Run the workflow from the environment or task.

        Args:
            target (Stateful): 
                The stateful entity to run the workflow.
            max_error_retry (int, defaults to 3):
                The maximum number of times to retry the workflow when the target is errored.
            max_idle_thinking (int, defaults to 1):
                The maximum number of times to idle thinking the workflow.
            prompts (dict[str, str], defaults to {}):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
            completion_config (dict[str, Any], defaults to {}):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool], defaults to None):
                The checker to check if the workflow should be running. 
            *args:
                The additional arguments for running the workflow.
            **kwargs:
                The additional keyword arguments for running the workflow.

        Returns:
            Stateful: 
                The stateful entity after running the workflow.
                
        Example:
        ```python
        async def run(
            self, 
            target: Stateful, 
            max_error_retry: int = 3, 
            max_idle_thinking: int = 1, 
            prompts: dict[str, str] = {}, 
            completion_config: dict[str, Any] = {}, 
            running_checker: Callable[[Stateful], bool] = None, 
            *args, 
            **kwargs,
        ) -> Stateful:
            # Update system prompt to history
            message = SystemMessage(content=self.prompts["system"])
            
            # Check if the target is running
            if running_checker(target):
                # Run the workflow
                # Observe the task
                observe = await self.observe(target)
                # Think about the task
                completion = await self.think(observe, allow_tools=True)
                # Act on the task
                target = await self.act(target, completion.tool_calls)
                # Reflect the task
                target = await self.reflect(target)
            else:
                # Log the error
                logger.error("The target is not running, the workflow is not executed.")
                # Set the target to error
                target.to_error()
            
            # Return the target
            return target
        ```
        """
        pass
