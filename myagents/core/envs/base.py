from typing import Callable, Union, OrderedDict

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.message import CompletionMessage, ToolCallRequest, ToolCallResult, MessageRole
from myagents.core.interface import Agent, Workflow, Environment, Logger, EnvironmentStatus, Task
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.utils.context import BaseContext
from myagents.core.utils.logger import Logger


class BaseEnvironment(BaseWorkflow, Environment):
    """BaseEnvironment is the base class for all the environments.
    
    Attributes:
        system_prompt (str):
            The system prompt of the environment.
            
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
            
        tasks (OrderedDict[str, Task]):
            The tasks of the environment.
        answers (OrderedDict[str, str]):
            The answers of the tasks.
        status (EnvironmentStatus):
            The status of the environment.
        history (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]):
            The history of the environment.
    """
    system_prompt: str
    
    agent: Agent
    debug: bool
    custom_logger: Logger
    context: BaseContext
    
    tools: dict[str, FastMCPTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]
    
    tasks: OrderedDict[str, Task]
    answers: OrderedDict[str, str]
    status: EnvironmentStatus
    history: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]
    
    def __init__(
        self, 
        agent: Agent, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
        workflows: dict[str, Workflow] = {}, 
        *args: tuple, 
        **kwargs: dict, 
    ) -> None:
        """Initialize the BaseEnvironment.
        
        Args:
            agent (Agent):
                The agent of the environment.
            custom_logger (Logger):
                The custom logger of the environment.
            debug (bool):
                Whether to enable the debug mode. 
            workflows (dict[str, Workflow], optional):
                The workflows that will be orchestrated to process the task.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(agent=agent, custom_logger=custom_logger, debug=debug, workflows=workflows, *args, **kwargs)
        
        # Initialize the tasks and answers
        self.tasks = OrderedDict()
        self.answers = OrderedDict()
        # Initialize the status
        self.status = EnvironmentStatus.CREATED
        # Initialize the history
        self.history = []

    def update(
        self, 
        message: Union[CompletionMessage, ToolCallRequest, ToolCallResult], 
    ) -> None:
        """Update the environment status.
        
        Args:
            message (Union[CompletionMessage, ToolCallRequest, ToolCallResult]):
                The message to be updated.
        """
        if len(self.history) > 0 and isinstance(message, CompletionMessage) and message.role == MessageRole.USER:
            last_message = self.history[-1]
            # Check if the last message is the same role as the current message
            if last_message.role == message.role:
                # Concatenate the content of the last message and the current message
                last_message.content = f"{last_message.content}\n{message.content}"
                last_message.stop_reason = message.stop_reason
            else:
                # Append the message directly
                self.history.append(message)
        else:
            # Append the message directly
            self.history.append(message)
