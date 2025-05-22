from abc import abstractmethod
from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.message import CompletionMessage, ToolCallRequest, ToolCallResult
from myagents.src.interface import Agent, RunnableEnvironment, Workflow, Logger
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.utils.logger import Logger


class BaseRunnableEnvironment(BaseWorkflow, RunnableEnvironment):
    """BaseRunnableEnvironment is the base class for all the runnable environments.
    
    Attributes:
        history (list[CompletionMessage | ToolCallRequest | ToolCallResult]):
            The history of the environment.
        agent (Agent):
            The agent of the environment.
        debug (bool):
            Whether to enable the debug mode.
        custom_logger (Logger):
            The custom logger of the environment.
        tools (dict[str, FastMCPTool]):
            The tools of the environment.
        tool_functions (dict[str, Callable]):
            The tool functions of the environment.
        workflows (dict[str, Workflow]):
            The workflows of the environment.
    """
    history: list[CompletionMessage | ToolCallRequest | ToolCallResult]
    
    agent: Agent
    debug: bool
    custom_logger: Logger
    tools: dict[str, FastMCPTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the BaseRunnableEnvironment.
        
        Args:
            *args:
                The arguments to pass to the BaseWorkflow.
            **kwargs:
                The keyword arguments to pass to the BaseWorkflow.     
        """
        # Get arguments for the BaseWorkflow
        agent = kwargs.get("agent", None)
        debug = kwargs.get("debug", False)
        custom_logger = kwargs.get("custom_logger", logger)
        
        super().__init__(agent, custom_logger, debug)
        
        # Initialize the history
        self.history = []
        
    @abstractmethod
    def post_init(self) -> None:
        """Post initialize the environment.
        """
        pass

    @abstractmethod
    async def run(self, *args, **kwargs) -> str:
        """Run the environment. This should override the run method of the BaseWorkflow.
        
        Args:
            *args:
                The arguments to pass to the run method of the BaseWorkflow.
            **kwargs:
                The keyword arguments to pass to the run method of the BaseWorkflow.
                
        Returns:
            str: 
                The result of the environment.
        """
        pass
    
    @abstractmethod
    def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool to modify the environment.
        """
        pass
