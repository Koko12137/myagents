import asyncio
from abc import abstractmethod
from enum import Enum
from typing import Union, Any, Awaitable

from fastmcp.tools import Tool as FastMCPTool

from myagents.core.interface import Agent, Workflow, Stateful
from myagents.core.utils.context import BaseContext
from myagents.core.tools_mixin import ToolsMixin


class BaseWorkflow(Workflow, ToolsMixin):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        profile (str):
            The profile of the workflow.
        system_prompt (str):
            The system prompt of the workflow.
        agent (Agent):
            The agent that is used to work with the workflow.
        context (BaseContext):
            The context of the tool call.
        tools (dict[str, FastMCPTool]):
            The tools of the workflow.
    """
    profile: str
    system_prompt: str
    agent: Agent
    # Tools Mixin
    context: BaseContext
    tools: dict[str, FastMCPTool]
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the BaseWorkflow. 
        
        Args:
            profile (str, optional):
                The profile of the workflow.
            system_prompt (str, optional):
                The system prompt of the workflow.
            agent (Agent, optional):
                The agent that is used to work with the workflow.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the workflow components
        self.profile = None
        self.system_prompt = None
        self.agent = None
        
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
        target: Union[Stateful, Any], 
        running_status: Enum, 
        finish_status: Enum, 
        error_status: Enum, 
        max_error_retry: int = 3, 
        observe_func: Awaitable[Union[str, list[dict]]] = None,
        process_error_func: Awaitable[None] = None,
        *args, 
        **kwargs,
    ) -> Union[Stateful, Any]:
        """Run the workflow from the environment or task.

        Args:
            target (Union[Stateful, Any]): 
                The stateful entity or any other entity to run the workflow.
            running_status (Enum):
                The status of the target, the workflow will stop running when the target status is the same as the 
                running status.
            finish_status (Enum):
                The status of the target, the workflow will exit when the target status is the same as the finish status.
            error_status (Enum):
                The status of the target, the workflow will enter the error process when the target status is the same as the 
                error status.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the workflow when the target is errored.
            observe_func (Awaitable[Union[str, list[dict]]], optional):
                The function to observe the target. If not provided, the default observe function will be used. 
            process_error_func (Awaitable[None], optional):
                The function to process the error of the target. If not provided, the default process error function will be used. 
            *args:
                The additional arguments for running the workflow.
            **kwargs:
                The additional keyword arguments for running the workflow.

        Returns:
            Union[Stateful, Any]: 
                The stateful entity or any other entity after running the workflow.
                
        Example:
        ```python
        async def run(
            self, 
            target: Stateful, 
            running_status: Enum, 
            finish_status: Enum, 
            error_status: Enum, 
            max_error_retry: int, 
            *args, 
            **kwargs,
        ) -> Stateful:
            # Update system prompt to history
            message = SystemMessage(content=self.system_prompt)
            
            # A while loop to run the workflow until the task is finished.
            while target.status != running_status:
            
                # Check if the target is finished
                if target.status == finish_status:
                    return target
                    
                # Check if the target is errored
                elif target.status == error_status:
                    process_error(target, max_error_retry)
                
                # Run the workflow
                else:
                    # Observe the task
                    observe = await self.observe(target)
                    # Think about the task
                    completion = await self.think(observe, allow_tools=True)
                    # Act on the task
                    target = await self.act(target, completion.tool_calls)
                    # Reflect the task
                    target = await self.reflect(target)
                
            return target
        ```
        """
        pass
