import asyncio
from abc import abstractmethod

from fastmcp.tools import Tool as FastMCPTool

from myagents.core.interface import Agent, Task, Workflow
from myagents.core.utils.context import BaseContext
from myagents.core.tools_mixin import ToolsMixin


class BaseWorkflow(Workflow, ToolsMixin):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent. 
        context (BaseContext):
            The context of the tool call.
        tools (dict[str, FastMCPTool]):
            The tools of the workflow.
    """
    profile: str
    agent: Agent
    context: BaseContext
    tools: dict[str, FastMCPTool]
    
    def __init__(
        self, 
        profile: str = "", 
        agent: Agent = None, 
        *args, 
        **kwargs, 
    ) -> None:
        """Initialize the BaseWorkflow. This will initialize the following components:
        
        - profile: The profile of the workflow.
        - agent: The agent that is used to work with the workflow.
        - context: The global context container of the workflow.
        - tools: The tools' description that can be used for the workflow.
        
        Args:
            profile (str, optional):
                The profile of the workflow.
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
        self.profile = profile
        self.agent = agent
        
        # Initialize the tools and context
        self.tools = {}
        self.context = BaseContext(
            prev=None,
            next=None,
            key_values={}
        )

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
                
        Raises:
            ValueError:
                If the workflow already has an agent.
        """
        # Check if the agent is registered
        if self.agent is not None:
            raise ValueError("The workflow already has an agent.")
        
        # Register the agent to the workflow
        self.agent = agent
        
    @abstractmethod
    async def post_init(self) -> None:
        """Post initialize the tools for the workflow.
        This method should be called after the initialization of the workflow. And you should register the tools in this method. 
        
        Example:
        ```python
        async def post_init(self) -> None:
            
            @self.register_tool("tool_name")
            def tool_function(self, *args, **kwargs) -> Any:
                pass
        ```
        """
        pass
    
    @abstractmethod
    async def run(self, task: Task) -> Task:
        """Run the workflow from the environment or task.

        Args:
            task (Task): 
                The task to run the workflow.

        Returns:
            Task: 
                The task after running the workflow.
                
        Example:
        ```python
        async def run(self, task: Task) -> Task:
            
            # A while loop to run the workflow until the task is finished.
            while task.status != TaskStatus.FINISHED:
                # 1. Observe the task
                observe = await self.observe(task)
                # 2. Think about the task
                completion = await self.think(observe, allow_tools=True)
                # 3. Act on the task
                task = await self.act(task, completion.tool_calls)
                # 4. Reflect the task
                task = await self.reflect(task)
                
            return task
        ```
        """
        pass
