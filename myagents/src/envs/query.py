from typing import Callable

from loguru import logger
from mcp import Tool as MCPTool
from fastmcp.tools import Tool as FastMcpTool

from myagents.src.message import CompletionMessage, ToolCallRequest, ToolCallResult, MessageRole
from myagents.src.interface import Agent, RunnableEnvironment, Workflow, Logger
from myagents.src.envs.task import BaseTask
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.workflows.react import ReActFlow


class Query(BaseWorkflow, RunnableEnvironment):
    """Query is the environment for the query.
    """
    history: list[CompletionMessage | ToolCallRequest | ToolCallResult]
    agent: Agent
    debug: bool
    custom_logger: Logger
    tools: dict[str, FastMcpTool | MCPTool]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]
    
    def __init__(
        self,
        agent: Agent, 
        react_flow: ReActFlow,
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> None:
        super().__init__(agent, custom_logger, debug)
        
        self.history = []
        self.workflows = {}
        
        # Initialize the react flow
        self.workflows["react"] = react_flow
        
        # Post initialize
        self.post_init()
        
    def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        """
        # Initialize the tools
        pass

    async def run(self, question: str, description: str) -> str:
        """Run the query.
        
        Args:
            question (str):
                The question to be answered.
            description (str):
                The description of the question.
        
        Returns:
            str:
                The answer to the question. 
        """
        # Record the question
        self.history.append(CompletionMessage(role=MessageRole.USER, content=question))
        # Create a new Task
        task = BaseTask(question=question, description=description)
        # Log the task
        self.custom_logger.info(f"Task created: \n{task.observe()}")
        # Run the react flow
        task = await self.workflows["react"].run(task)
        # Record the task answer
        self.history.append(CompletionMessage(role=MessageRole.ASSISTANT, content=task.answer))
        # Log the task answer
        self.custom_logger.info(f"Task answered: \n{task.observe()}")
        # Return the answer
        return task.answer
    
    async def observe(self) -> str:
        """Observe the environment.
        """
        history = []
        for message in self.history:
            # Format the message
            history.append(f"{message.role}: {message.content}")
        # Return the history
        return "\n".join(history)
    
    async def call_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """Call a tool to modify the environment.
        """
        return ToolCallResult(tool_call_id=tool_call.id, content="No tool is available.", is_error=True)
    