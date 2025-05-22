from abc import ABCMeta, abstractmethod
from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.interface import Agent, Environment, Task, Workflow, Logger
from myagents.src.message import ToolCallRequest, ToolCallResult


class BaseWorkflow(Workflow, metaclass=ABCMeta):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        agent (Agent):
            The agent. 
        debug (bool, defaults to False):
            The debug flag. 
        custom_logger (Logger, defaults to None):
            The custom logger. 
        tools (dict[str, FastMCPTool]):
            The tools of the workflow.
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
        self.agent = agent
        self.custom_logger = custom_logger
        self.debug = debug
        self.tools = {}
        self.tool_functions = {}
    
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
        
    @abstractmethod
    async def run(self, env: Environment | Task) -> Environment | Task:
        """Run the workflow from the environment or task.
        
        Args:
            env (Environment | Task):
                The environment or task to run the workflow.
                
        Returns:
            Environment | Task:
                The environment or task after running the workflow.
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
            # 1. Create a FastMCPTool instance
            tool = FastMCPTool.from_function(func)
            # 2. Register the tool to the workflow
            self.tools[name] = tool
            self.tool_functions[name] = func
            # 3. Return the function
            return func
        # Return the wrapper function
        return wrapper
    
    @abstractmethod
    async def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
                
        Returns:
            ToolCallResult:
                The tool call result. 
                
        Raises:
            ValueError:
                If the tool call name is not registered. 
        """
        pass