from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.src.message import CompletionMessage, ToolCallRequest, ToolCallResult
from myagents.src.interface import Agent, Workflow, Environment, Logger
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.utils.context import BaseContext
from myagents.src.utils.logger import Logger


class BaseEnvironment(BaseWorkflow, Environment):
    """BaseEnvironment is the base class for all the environments.
    
    Attributes:
        history (list[CompletionMessage | ToolCallRequest | ToolCallResult]):
            The history of the environment.
            
        agent (Agent):
            The agent of the environment.
        debug (bool):
            Whether to enable the debug mode.
        custom_logger (Logger):
            The custom logger of the environment.
        context (BaseContext):
            The context of the environment.
        
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
        """Initialize the BaseEnvironment.
        
        Args:
            agent (Agent):
                The agent of the environment.
            custom_logger (Logger):
                The custom logger of the environment.
            debug (bool):
                Whether to enable the debug mode.
        """
        super().__init__(agent, custom_logger, debug)
        
        # Initialize the history
        self.history = []
