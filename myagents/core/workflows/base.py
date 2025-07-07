import asyncio
from abc import abstractmethod

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
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Run the workflow from the environment or task.

        Args:
            target (Stateful): 
                The stateful entity to run the workflow.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the workflow when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the workflow. 
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the workflow. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
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
            max_error_retry: int, 
            max_idle_thinking: int, 
            tool_choice: str, 
            exclude_tools: list[str], 
            *args, 
            **kwargs,
        ) -> Stateful:
            # Update system prompt to history
            message = SystemMessage(content=self.system_prompt)
            
            # A while loop to run the workflow until the task is finished.
            while target.is_running():
            
                # Check if the target is finished
                if target.is_finished():
                    return target
                    
                # Check if the target is errored
                elif target.is_errored():
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
