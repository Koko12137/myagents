import traceback
from typing import overload, Union

from loguru import logger
from fastmcp import Client as MCPClient
from fastmcp.exceptions import ClientError
from fastmcp.tools import Tool as FastMcpTool
from mcp import Tool as MCPTool

from myagents.src.message import CompletionMessage, ToolCallRequest, ToolCallResult
from myagents.src.interface import LLM, Agent, StepCounter, Task, Environment, Logger
from myagents.src.envs.task import TaskContextView
from myagents.src.utils.tools import tool_schema
from myagents.src.utils.context import BaseContext


class MaxStepsError(Exception):
    """MaxStepsError is the error raised when the max steps is reached.
    
    Attributes:
        message (str):
            The message of the error.
    """
    message: str
    current: int
    limit: int
    
    def __init__(self, message: str, current: int, limit: int) -> None:
        """Initialize the MaxStepsError.
        
        Args:
            message (str):
                The message of the error.
            current (int):
                The current step of the step counter.
            limit (int):
                The limit of the step counter.
        """
        self.message = message
        self.current = current
        self.limit = limit
        
    def __str__(self) -> str:
        """Return the string representation of the MaxStepsError.
        
        Returns:
            str:
                The string representation of the MaxStepsError.
        """
        return f"MaxStepsError: {self.message}, current: {self.current}, limit: {self.limit}"


class BaseStepCounter(StepCounter):
    """StepCounter is the step counter for the base agent. The limit is the max auto steps. 
    
    Attributes:
        limit (int):
            The limit of the step counter. 
        current (int):
            The current step of the step counter. 
    """
    limit: int
    current: int
    
    def __init__(self, limit: int = 10) -> None:
        """Initialize the step counter.
        
        Args:
            limit (int, optional, defaults to 10):
                The limit of the step counter. 
        """
        self.limit = limit
        self.current = 0
        
    def step(self, step: int | float = 1) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (int | float, optional, defaults to 1):
                The step to increment. 
                
        Returns:
            None 
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        self.current += step
        # Check if the current step is greater than the max auto steps
        if self.current > self.limit:
            # Request the user to reset the step counter
            reset = input(f"The limit of auto steps is reached. Do you want to reset the step counter with limit {e.limit} steps? (y/n)")
            
            if reset == "y":
                # Reset the step counter and continue the loop
                self.step_counter.reset()
            else:
                e = MaxStepsError("Max auto steps reached.", self.current, self.limit)
                # The max steps error is raised, then update the task status to cancelled
                self.custom_logger.error(f"Max steps error: {e}")
                raise e
        
    def reset(self) -> None:
        """Reset the current step of the step counter.
        
        Returns:
            None
        """
        self.current = 0
        
    def update_limit(self, limit: int) -> None:
        """Update the limit of the step counter.
        
        Args:
            limit (int):
                The limit of the step counter. 
        """
        self.limit = limit
        

class BaseAgent(Agent):
    """BaseAgent is the base class for all the agents.
    
    Attributes:
        llm (LLM):
            The LLM to use for the agent.
        debug (bool, defaults to False):
            The debug flag to use for the agent.
        custom_logger (Logger, defaults to None):
            The custom logger to use for the agent.
        context (BaseContext):
            The context of the tool call.
            
        mcp_client (MCPClient):
            The MCP client to use for the agent.
        tools (list[dict[str, str]]):
            The tool descriptions can be used for the agent. 
            
        step_counter (StepCounter):
            The step counter to use for the agent. 
    """
    llm: LLM
    debug: bool
    custom_logger: Logger
    context: BaseContext
    
    # Stateless tools
    mcp_client: MCPClient 
    tools: list[dict[str, str]]
    
    # Max auto steps
    step_counter: StepCounter
    
    def __init__(
        self, 
        llm: LLM, 
        step_counter: StepCounter, 
        mcp_client: MCPClient | None = None, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> None:
        """Initialize the BaseAgent.
        
        Args:
            llm (LLM):
                The LLM to use for the agent.
            step_counter (StepCounter):
                The step counter to use for the agent. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent. If not provided, No MCP tools can be used. 
            custom_logger (Logger, defaults to logger):
                The custom logger to use for the agent. If not provided, the default loguru logger will be used. 
            debug (bool, defaults to False):
                The debug flag to use for the agent.
        """
        self.llm = llm
        self.debug = debug
        self.custom_logger = custom_logger
        self.context = BaseContext(
            prev=None,
            next=None,
            key_values={}
        )
        
        # Initialize the MCP client
        self.mcp_client = mcp_client
        
        # Initialize the tool descriptions
        self.tools = []
        if self.mcp_client is not None:
            tools = self.mcp_client.list_tools()
            for tool in tools:
                self.tools.append(tool_schema(tool, self.llm.provider))
        
        # Initialize the max auto steps
        self.step_counter = step_counter
        
    @overload
    async def observe(self, env: Environment) -> str:
        """Observe the environment.
        
        Args:
            env (Environment): 
                The environment to observe. 
                
        Returns:
            str:
                The up to date information observed from the environment.  
        """
        pass
    
    @overload
    async def observe(self, env: Task) -> str:
        """Observe the Task.    
        
        Args:
            env (Task): 
                The task to observe. 

        Returns:
            str:
                The up to date information observed from the environment.  
        """
        pass
    
    async def observe(self, env: Union[Environment, Task]) -> str:
        """Observe the Task.    
        
        Args:
            env (Union[Environment, Task]): 
                The environment or task to observe. 

        Returns:
            str:
                The up to date information observed from the environment.  
        """
        
        if isinstance(env, Environment):
            raise NotImplementedError("Environment is not supported for observation.")
        elif isinstance(env, Task):
            observation = TaskContextView(env).format()
        else:
            raise ValueError(f"Unsupported environment type: {type(env)}")
        
        return observation
        
    async def think(
        self, 
        observe: list[CompletionMessage | ToolCallRequest | ToolCallResult], 
        allow_tools: bool, 
        external_tools: dict[str, FastMcpTool | MCPTool] = {}, 
        **kwargs: dict, 
    ) -> CompletionMessage:
        """Think about the environment.
        
        Args:
            observe (list[CompletionMessage | ToolCallRequest | ToolCallResult]):
                The messages observed from the environment. 
            allow_tools (bool):
                Whether to allow the tools provided by the agent to be used. This do not affect the 
                external tools provided by the workflow. 
            external_tools (dict[str, FastMcpTool | MCPTool]):
                The external tools to use for the agent.  
            **kwargs (dict): 
                The additional keyword arguments for thinking about the observed messages. 

        Returns:
            CompletionMessage:
                The completion message thought about by the LLM. 
        """
        # Increment the current step
        self.step_counter.step()
        
        # Get the available tools
        external_tools_list = list(external_tools.values())
        external_tools_list = [tool_schema(tool, self.llm.provider) for tool in external_tools_list]
        if allow_tools:
            available_tools = self.tools + external_tools_list
        else:
            available_tools = external_tools_list
        
        # Call for completion from the LLM
        message = await self.llm.completion(observe, available_tools=available_tools, **kwargs)
        
        # Return the response
        return message
    
    async def call_tool(
        self, 
        ctx: Union[Task, Environment], 
        tool_call: ToolCallRequest, 
        **kwargs: dict, 
    ) -> ToolCallResult:
        """Call a tool. 
        If there is any error caused by the tool call, the flag `is_error` will be set to True. 
        However, if there is any error caused by the MCP client connection, this should raise a RuntimeError.  
        
        Args:
            ctx (Union[Task, Environment]):
                The task or environment to call the tool.
            tool_call (ToolCallRequest): 
                The tool call request including the tool call id and the tool call arguments.
            **kwargs (dict):
                The additional keyword arguments for calling the tool.
        Returns:
            ToolCallResult: 
                The result of the tool call. 
                
        Raises:
            RuntimeError:
                The runtime error raised by the MCP client connection. 
        """
        # Create a new context
        self.context = self.context.create_next(ctx=ctx, **kwargs)
        
        # Get the tool call id and the tool call arguments
        tool_call_id = tool_call.id
        tool_call_name = tool_call.name
        tool_call_args = tool_call.args
        
        try:
            # Call the tool 
            res = await self.mcp_client.call_tool(tool_call_name, tool_call_args)
        except ClientError as e:
            # Record and return the error
            self.custom_logger.error(f"Error calling tool {tool_call_name}: {e}")
            return ToolCallResult(
                tool_call_id=tool_call_id, 
                content=str(e), 
                is_error=True, 
            )
        except RuntimeError as e:
            # Raise the error 
            self.custom_logger.error(f"{e}, traceback: {traceback.format_exc()}")
            raise e
        
        # Done the current context
        self.context = self.context.done()
        
        # Return the result
        return ToolCallResult(
            tool_call_id=tool_call_id, 
            content=res, 
            is_error=False, 
        )
