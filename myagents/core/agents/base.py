import traceback
from uuid import uuid4
from typing import overload, Union, Optional
from asyncio import Lock

from loguru import logger
from fastmcp import Client as MCPClient
from fastmcp.exceptions import ClientError
from fastmcp.tools import Tool as FastMcpTool
from mcp import Tool as MCPTool

from myagents.core.message import CompletionUsage, CompletionMessage, ToolCallRequest, ToolCallResult
from myagents.core.interface import LLM, Agent, StepCounter, Task, Environment, Logger
from myagents.core.envs.task import TaskContextView
from myagents.core.utils.tools import tool_schema
from myagents.core.utils.context import BaseContext


class MaxStepsError(Exception):
    """MaxStepsError is the error raised when the max steps is reached.
    
    Attributes:
        current (int):
            The current step of the step counter.
        limit (int):
            The limit of the step counter.
    """
    current: int
    limit: int
    
    def __init__(self, current: int, limit: int) -> None:
        """Initialize the MaxStepsError.
        
        Args:
            current (int):
                The current step of the step counter.
            limit (int):
                The limit of the step counter.
        """
        self.current = current
        self.limit = limit
        
    def __str__(self) -> str:
        """Return the string representation of the MaxStepsError.
        
        Returns:
            str:
                The string representation of the MaxStepsError.
        """
        return f"Max auto steps reached. Current: {self.current}, Limit: {self.limit}"


class MaxStepCounter(StepCounter):
    """MaxStepCounter allows the user to set the limit of the step counter, and the limit will **never** be reset. 
    
    Attributes:
        limit (int):
            The limit of the step counter. 
        current (int):
            The current step of the step counter. 
        custom_logger (Logger, defaults to logger):
            The custom logger to use for the step counter. 
    """
    uid: str
    limit: int
    current: int
    lock: Lock
    custom_logger: Logger
    
    def __init__(self, limit: int = 10, custom_logger: Logger = logger) -> None:
        """Initialize the step counter.
        
        Args:
            limit (int, optional, defaults to 10):
                The limit of the step counter. 
        """
        self.uid = uuid4().hex
        self.limit = limit
        self.current = 0
        self.custom_logger = custom_logger
        self.lock = Lock()
        
    async def check_limit(self) -> bool:
        """Check if the limit of the step counter is reached.
        
        Returns:
            bool:
                Whether the limit of the step counter is reached.
                
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        if self.current >= self.limit:
            e = MaxStepsError(self.current, self.limit)
            self.custom_logger.error(e)
            raise e
        return False
        
    async def step(self, step: Union[int, float] = 1) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (Union[int, float], optional, defaults to 1):
                The step to increment. 
                
        Returns:
            None 
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        async with self.lock:
            self.current += 1
            
        # Check if the current step is greater than the max auto steps
        if await self.check_limit():
            e = MaxStepsError(self.current, self.limit)
            # The max steps error is raised, then update the task status to cancelled
            self.custom_logger.error(e)
            raise e
        
    async def reset(self) -> None:
        """Reset the current step of the step counter.
        
        Returns:
            None
        """
        raise NotImplementedError("Reset is not supported for the max step counter.")
    
    async def update_limit(self, limit: Union[int, float]) -> None:
        """Update the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        raise NotImplementedError("Update limit is not supported for the max step counter.")
    
    async def recharge(self, limit: Union[int, float]) -> None:
        """Recharge the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        raise NotImplementedError("Recharge is not supported for the max step counter.")


class BaseStepCounter(MaxStepCounter):
    """BaseStepCounter count only the action steps, without any concern of the token usage. The step counter could be 
    reset by the user. 
    
    Attributes:
        uid (str):
            The unique identifier of the step counter. 
        limit (int):
            The limit of the step counter. 
        current (int):
            The current step of the step counter. 
        custom_logger (Logger):
            The custom logger to use for the step counter. 
    """
    uid: str
    limit: int
    current: int
    custom_logger: Logger
    
    def __init__(self, limit: int = 10, custom_logger: Logger = logger) -> None:
        """Initialize the step counter.
        
        Args:
            limit (int, optional, defaults to 10):
                The limit of the step counter. 
        """
        super().__init__(limit, custom_logger)
        
    async def step(self, step: CompletionUsage) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (CompletionUsage):
                The step to increment. 
                
        Returns:
            None 
        
        Raises:
            MaxStepsError:
                The max steps error raised by the step counter. 
        """
        async with self.lock:
            self.current += 1
            
        # Check if the current step is greater than the max auto steps
        if await self.check_limit():
            # Request the user to reset the step counter
            reset = input(f"The limit of auto steps is reached. Do you want to reset the step counter with limit {e.limit} steps? (y/n)")
            
            if reset == "y":
                # Reset the step counter and continue the loop
                await self.reset()
            else:
                e = MaxStepsError(self.current, self.limit)
                # The max steps error is raised, then update the task status to cancelled
                self.custom_logger.error(e)
                raise e
        
    async def reset(self) -> None:
        """Reset the current step of the step counter.
        
        Returns:
            None
        """
        async with self.lock:
            self.current = 0
        
    async def update_limit(self, limit: int) -> None:
        """Update the limit of the step counter.
        
        Args:
            limit (int):
                The limit of the step counter. 
        """
        async with self.lock:
            self.limit = limit
        
    async def recharge(self, limit: Union[int, float]) -> None:
        """Recharge the limit of the step counter.
        
        Args:
            limit (Union[int, float]):
                The limit of the step counter. 
        """
        async with self.lock:
            self.limit += limit


class TokenStepCounter(BaseStepCounter):
    """TokenStepCounter counts the token usage of the LLM. The step counter will ask for reset when the token usage is greater than the limit. 
    
    Attributes:
        uid (str):
            The unique identifier of the step counter. 
        limit (int):
            The limit of the step counter. 
        current (int):
            The current step of the step counter. 
        custom_logger (Logger):
            The custom logger to use for the step counter. 
    """
    uid: str
    limit: int
    current: int
    custom_logger: Logger
    
    def __init__(self, limit: int = 10000, custom_logger: Logger = logger) -> None:
        """Initialize the step counter.
        
        Args:
            limit (int, optional, defaults to 10000):
                The limit of the step counter. Default to 10 thousand. 
            custom_logger (Logger, optional, defaults to logger):
                The custom logger to use for the step counter. 
        """
        super().__init__(limit, custom_logger)
    
    async def step(self, step: CompletionUsage) -> None:
        """Increment the current step of the step counter.
        
        Args:
            step (CompletionUsage):
                The step to increment. 
        """
        async with self.lock:
            self.current += step.total_tokens
            self.custom_logger.warning(f"The current Token Usage is {self.current}, the Limit is {self.limit}.")
            
        # Check if the current step is greater than the max auto steps
        if await self.check_limit():
            # Request the user to reset the step counter
            reset = input(f"The limit of auto steps is reached. Do you want to reset the step counter with limit {self.limit} steps? (y/n)")
            
            if reset == "y":
                # Reset the step counter and continue the loop  
                await self.reset()
            else:
                e = MaxStepsError(self.current, self.limit)
                # The max steps error is raised, then update the task status to cancelled
                self.custom_logger.error(e)
                raise e
            
    async def reset(self) -> None:
        """Reset is not supported for the token step counter.
        """
        raise NotImplementedError("Reset is not supported for the token step counter. Please use `recharge` to update the limit.")

        
class DummyAgent(Agent):
    """DummyAgent do nothing but return a dummy response.
    """
    llm: LLM
    debug: bool
    custom_logger: Logger
    context: BaseContext
    
    # Stateless tools
    mcp_client: MCPClient 
    tools: list[dict[str, str]]
    
    # Max auto steps
    step_counters: dict[str, StepCounter]
    
    def __init__(
        self, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        custom_logger: Optional[Logger] = logger, 
        debug: bool = False, 
    ) -> None: 
        """Initialize the DummyAgent.
        
        Args:
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent. If not provided, No MCP tools can be used. 
            custom_logger (Logger, defaults to logger):
                The custom logger to use for the agent. If not provided, the default loguru logger will be used. 
            debug (bool, defaults to False):
                The debug flag to use for the agent.
        """
        # Initialize the LLM
        self.llm = llm
        # Initialize the debug flag
        self.debug = debug
        # Initialize the custom logger
        self.custom_logger = custom_logger
        # Initialize the context
        self.context = BaseContext(
            prev=None,
            next=None,
            key_values={}
        )
        
        # Initialize the step counters
        self.step_counters = {counter.uid: counter for counter in step_counters}
    
    async def think(
        self, 
        observe: list[CompletionMessage, ToolCallRequest, ToolCallResult], 
        allow_tools: bool, 
        external_tools: dict[str, Union[FastMcpTool, MCPTool]] = {}, 
        **kwargs: dict, 
    ) -> CompletionMessage:
        """Think about the environment.
        
        Args:
            observe (list[CompletionMessage, ToolCallRequest, ToolCallResult]):
                The messages observed from the environment. 
            allow_tools (bool):
                Whether to allow the tools provided by the agent to be used. This do not affect the 
                external tools provided by the workflow. 
            external_tools (dict[str, Union[FastMcpTool, MCPTool]]):
                The external tools to use for the agent.  
            **kwargs (dict): 
                The additional keyword arguments for thinking about the observed messages. 

        Returns:
            CompletionMessage:
                The completion message thought about by the LLM. 
        """
        # Check if the limit of the step counters is reached
        for step_counter in self.step_counters.values():
            await step_counter.check_limit()
        
        return CompletionMessage(
            role="assistant",
            content="Dummy agent is thinking...",
        )
    
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
        """
        return ToolCallResult(
            tool_call_id=tool_call.id,
            content="Dummy agent is calling tool.",
            is_error=False,
        )
        
    async def observe(self, env: Union[Task, Environment]) -> str:
        """Observe the environment.
        
        Args:
            env (Union[Task, Environment]):
                The task or environment to observe.

        Returns:
            str:
                The up to date information observed from the environment.  
        """
        return "Dummy agent is observing the environment."
    
    def register_counter(self, counter: StepCounter) -> None:
        """Register a step counter to the dummy agent.
        
        Args:
            counter (StepCounter):
                The step counter to register.
        """
        self.step_counters[counter.uid] = counter
        

class BaseAgent(Agent):
    """BaseAgent is the base class for all the agents.
    
    Attributes:
        llm (LLM):
            The LLM to use for the agent.
        debug (bool, defaults to False):
            The debug flag to use for the agent.
        custom_logger (Logger, defaults to logger):
            The custom logger to use for the agent.
        context (BaseContext):
            The context of the tool call.
            
        mcp_client (MCPClient):
            The MCP client to use for the agent.
        tools (list[dict[str, str]]):
            The tool descriptions can be used for the agent. 
            
        step_counters (dict[str, StepCounter]):
            The step counters to use for the agent. 
    """
    llm: LLM
    debug: bool
    custom_logger: Logger
    context: BaseContext
    
    # Stateless tools
    mcp_client: MCPClient 
    tools: list[dict[str, str]]
    
    # Max auto steps
    step_counters: dict[str, StepCounter]
    
    def __init__(
        self, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        custom_logger: Optional[Logger] = logger, 
        debug: bool = False, 
    ) -> None:
        """Initialize the BaseAgent.
        
        Args:
            llm (LLM):
                The LLM to use for the agent.
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
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
        self.step_counters = {counter.uid: counter for counter in step_counters}
        
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
    
    async def observe(self, env: Union[Environment, Task], **kwargs) -> str:
        """Observe the Task.    
        
        Args:
            env (Union[Environment, Task]): 
                The environment or task to observe. 
            **kwargs:
                The additional keyword arguments for observing the environment or task. 
            
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
        observe: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]], 
        allow_tools: bool, 
        external_tools: dict[str, Union[FastMcpTool, MCPTool]] = {}, 
        **kwargs, 
    ) -> CompletionMessage:
        """Think about the environment.
        
        Args:
            observe (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]):
                The messages observed from the environment. 
            allow_tools (bool):
                Whether to allow the tools provided by the agent to be used. This do not affect the 
                external tools provided by the workflow. 
            external_tools (dict[str, Union[FastMcpTool, MCPTool]]):
                The external tools to use for the agent.  
            **kwargs: 
                The additional keyword arguments for thinking about the observed messages. 

        Returns:
            CompletionMessage:
                The completion message thought about by the LLM. 
        """
        # Check if the limit of the step counters is reached
        for step_counter in self.step_counters.values():
            await step_counter.check_limit()
        
        # Get the available tools
        external_tools_list = list(external_tools.values())
        external_tools_list = [tool_schema(tool, self.llm.provider) for tool in external_tools_list]
        if allow_tools:
            available_tools = self.tools + external_tools_list
        else:
            available_tools = external_tools_list
        
        # Call for completion from the LLM
        message = await self.llm.completion(observe, available_tools=available_tools, **kwargs)
        
        errors = []
        # Increase the current step
        for step_counter in self.step_counters.values():
            try:
                await step_counter.step(message.usage)
            except MaxStepsError as e:
                errors.append(e)
            except Exception as e:
                raise e
        
        if len(errors) > 0:
            raise errors[0] from errors[0]
        
        # Return the response
        return message
    
    async def call_tool(
        self, 
        ctx: Union[Task, Environment], 
        tool_call: ToolCallRequest, 
        **kwargs, 
    ) -> ToolCallResult:
        """Call a tool. 
        If there is any error caused by the tool call, the flag `is_error` will be set to True. 
        However, if there is any error caused by the MCP client connection, this should raise a RuntimeError.  
        
        Args:
            ctx (Union[Task, Environment]):
                The task or environment to call the tool.
            tool_call (ToolCallRequest): 
                The tool call request including the tool call id and the tool call arguments.
            **kwargs:
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

    def register_counter(self, counter: StepCounter) -> None:
        """Register a step counter to the base agent.
        
        Args:
            counter (StepCounter):
                The step counter to register.
        """
        self.step_counters[counter.uid] = counter
