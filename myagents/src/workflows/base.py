from abc import ABCMeta, abstractmethod
from typing import Callable, Any, Union

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.interface import Agent, Task, Workflow, Environment, Logger, StepCounter
from myagents.src.message import ToolCallRequest, ToolCallResult
from myagents.src.utils.context import BaseContext


class BaseWorkflow(Workflow, metaclass=ABCMeta):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        agent (Agent):
            The agent. 
        debug (bool, defaults to False):
            The debug flag. 
        custom_logger (Logger, defaults to None):
            The custom logger. 
        context (BaseContext):
            The context of the tool call.
            
        tools (dict[str, FastMCPTool]):
            The tools of the workflow.
        tool_functions (dict[str, Callable]):
            The tool functions of the workflow.
        workflows (dict[str, Workflow]):
            The workflows of the workflow.
    """
    agent: Agent
    debug: bool
    custom_logger: Logger
    context: BaseContext
    
    tools: dict[str, FastMCPTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]
    
    def __init__(
        self, 
        agent: Agent, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> None:
        """Initialize the BaseWorkflow. This will initialize the following components:
        
        - agent: The agent that is used to work with the workflow.
        - custom_logger: The custom logger. If not provided, the default loguru logger will be used. 
        - debug: The debug flag. If not provided, the default value is False. 
        - context: The global context container of the workflow.
        - tools: The tools' description that can be used for the workflow.
        - tool_functions: The tool functions that can be used for the workflow.
        - workflows: The workflows that will be orchestrated to process the task. 
        
        Args:
            agent (Agent):
                The agent that is used to work with the workflow.
            custom_logger (Logger, optional):
                The custom logger. If not provided, the default loguru logger will be used. 
            debug (bool, optional):
                The debug flag. If not provided, the default value is False. 
        """
        self.agent = agent
        self.custom_logger = custom_logger
        self.debug = debug
        self.context = BaseContext(
            prev=None,
            next=None,
            key_values={}
        )
        
        self.tools = {}
        self.tool_functions = {}
        self.workflows = {}
        
    @abstractmethod
    def post_init(self) -> None:
        """Post initialize the tools for the workflow.
        This method should be called after the initialization of the workflow. And you should register the tools in this method. 
        
        Example:
        ```python
        def post_init(self) -> None:
            
            @self.register_tool("tool_name")
            def tool_function(self, *args, **kwargs) -> Any:
                pass
        ```
        """
        pass

    def register_tool(self, name: str) -> Callable:
        """This is a FastAPI like decorator to register a tool to the workflow.
        
        Args:
            name (str):
                The name of the tool.
                
        Returns:
            Callable:
                The function register.
        """
        # Define a wrapper function to register the tool
        def wrapper(func: Callable) -> Callable:
            """Wrapper function to call the tool.
            
            Args:
                func (Callable):
                    The function to register.
            
            Returns:
                Callable:
                    The function registered.
            """
            self.add_tool(name, func)
            return func
        
        # Return the wrapper function
        return wrapper
    
    def add_tool(self, name: str, tool: Callable[[Any], Union[Task, Environment]]) -> None:
        """Add a tool to the workflow.
        
        Args:
            name (str):
                The name of the tool.
            tool (Callable[[Any], Union[Task, Environment]]):
                The tool to add. This tool should return the next task or environment. 
        """
        # Create a FastMCPTool instance
        tool_obj = FastMCPTool.from_function(tool)
        # Register the tool to the workflow
        self.tools[name] = tool_obj
        # Register the tool function to the workflow
        self.tool_functions[name] = tool
    
    @abstractmethod
    async def call_tool(
        self, 
        ctx: Union[Task, Environment], 
        tool_call: ToolCallRequest, 
        **kwargs: dict, 
    ) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            ctx (Union[Task, Environment]):
                The task or environment to call the tool.
            tool_call (ToolCallRequest):
                The tool call request.
            **kwargs (dict):
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult:
                The tool call result. 
                
        Raises:
            ValueError:
                If the tool call name is not registered. 
        """
        pass
    
    @abstractmethod
    async def run(self, env: Union[Environment, Task]) -> Union[Environment, Task]:
        """Run the workflow from the environment or task.
        
        Args:
            env (Union[Environment, Task]):
                The environment or task to run the workflow.
                
        Returns:
            Union[Environment, Task]:
                The environment or task after running the workflow.
        """
        pass
        
    def register_counter(self, counter: StepCounter) -> None:
        """Register a step counter to the workflow.
        
        Args:
            counter (StepCounter):
                The step counter to register.
        """
        # Register the step counter to the agent
        self.agent.register_counter(counter)
        # Register the step counter to the workflows
        for workflow in self.workflows.values():
            workflow.register_counter(counter)
