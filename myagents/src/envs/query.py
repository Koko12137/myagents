from typing import Callable, Union

from loguru import logger
from mcp import Tool as MCPTool
from fastmcp.tools import Tool as FastMcpTool

from myagents.src.message import CompletionMessage, ToolCallRequest, ToolCallResult, MessageRole
from myagents.src.interface import Agent, Workflow, Logger, Task, Environment, Context
from myagents.src.envs.task import BaseTask
from myagents.src.workflows.base import BaseWorkflow
from myagents.src.workflows.react import ReActFlow


class Query(BaseWorkflow, Environment):
    """Query is the environment for the query and answer the question.
    
    Attributes:
        history (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]):
            The history of the environment.
        agent (Agent):
            The agent that will be used to answer the question.
        debug (bool):
            The debug flag.
        custom_logger (Logger):
            The custom logger.
        tools (dict[str, Union[FastMcpTool, MCPTool]]):
            The tools of the environment.
        tool_functions (dict[str, Callable]):
            The functions of the tools.
        workflows (dict[str, Workflow]):
            The workflows of the environment. 
    """
    history: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]
    
    agent: Agent
    debug: bool
    custom_logger: Logger
    context: Context
    
    tools: dict[str, Union[FastMcpTool, MCPTool]]
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
        task = BaseTask(create_doc="This is the root task.", question=question, description=description)
        # Log the task
        self.custom_logger.info(f"Task created: \n{task.question}")
        # Run the react flow
        task = await self.workflows["react"].run(task)
        # Record the task answer
        self.history.append(CompletionMessage(role=MessageRole.ASSISTANT, content=task.answer))
        # Log the task answer
        self.custom_logger.info(f"Task answered: \n{task.answer}")
        # Return the answer
        return task.answer
    
    async def call_tool(self, ctx: Union[Task, Environment], tool_call: ToolCallRequest, **kwargs: dict) -> ToolCallResult:
        """Call a tool to modify the environment.
        """
        # Create a new context
        self.context = self.context.create_next(ctx=ctx, **kwargs)
        
        # Done the current context
        self.context = self.context.done()
        
        # Return the result
        return ToolCallResult(tool_call_id=tool_call.id, content="No tool is available.", is_error=True)
    