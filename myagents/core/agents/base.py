import traceback
from uuid import uuid4
from asyncio import Queue, Lock
from typing import Union, Optional, Callable, Awaitable, Any

from loguru import logger
from fastmcp import Client as MCPClient
from fastmcp.exceptions import ClientError
from fastmcp.tools import Tool as FastMcpTool
from mcp import Tool as MCPTool

from myagents.core.messages import AssistantMessage, ToolCallRequest, ToolCallResult
from myagents.core.interface import LLM, Agent, StepCounter, Environment, Stateful, Workflow
from myagents.core.agents.types import AgentType
from myagents.core.utils.tools import tool_schema
from myagents.core.utils.step_counters import MaxStepsError


class BaseAgent(Agent):
    """BaseAgent is the base class for all the agents that can:
    - Observe the environment or task.
    - Think about the environment or task.
    - Call the tools.
    - Run the agent.
    
    Attributes:
        uid (str):
            The unique identifier of the agent.
        name (str):
            The name of the agent.
        type (AgentType):
            The type of the agent.
        profile (str):
            The profile of the agent.
        llm (LLM):
            The LLM to use for the agent. 
        mcp_client (MCPClient):
            The MCP client to use for the agent.
        workflow (Workflow):
            The workflow to that the agent is running on.
        env (Environment):
            The environment to that the agent is running on.
        step_counters (dict[str, StepCounter]):
            The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
        lock (Lock):
            The synchronization lock of the agent. The agent can only work on one task at a time. 
            If the agent is running concurrently, the global context may not be working properly.
    """
    # Basic information
    uid: str
    name: str
    type: AgentType
    profile: str
    # LLM and MCP client
    llm: LLM
    mcp_client: MCPClient
    # Workflow and environment
    workflow: Workflow
    env: Environment
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    # Concurrency limit
    lock: Lock
    
    def __init__(
        self, 
        name: str, 
        type: AgentType, 
        profile: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        *args, 
        **kwargs, 
    ) -> None:
        """Initialize the BaseAgent.
        
        Args:
            name (str):
                The name of the agent.
            type (AgentType):
                The type of the agent.
            profile (str):
                The profile of the agent.
            llm (LLM):
                The LLM to use for the agent. 
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent. If not provided, No MCP tools can be used.  
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(*args, **kwargs)
        
        # Initialize the basic information
        self.uid = str(uuid4())
        self.name = name
        self.type = type
        self.profile = profile
        # Initialize the LLM and MCP client
        self.llm = llm
        self.mcp_client = mcp_client
        # Initialize the workflow and environment
        self.workflow = None
        self.env = None
        # Initialize the step counters
        self.step_counters = {counter.uname: counter for counter in step_counters}
        # Initialize the synchronization lock
        self.lock = Lock()
    
    async def observe(
        self, 
        target: Union[Stateful, Any], 
        observe_func: Optional[Callable[..., Awaitable[Union[str, list[dict]]]]] = None, 
        **kwargs, 
    ) -> Union[str, list[dict]]:
        """Observe the target. If the target is not a task or environment, you should provide the observe 
        function to get the string or list of dicts observation. 
        
        Args:
            target (Union[Stateful, Any]): 
                The stateful entity or any other entity to observe. 
            observe_func (Callable[..., Awaitable[Union[str, list[dict]]]], optional):
                The function to observe the target. If not provided, the default observe function will be used. 
            **kwargs:
                The additional keyword arguments for observing the target. 
            
        Returns:
            Union[str, list[dict]]:
                The up to date information observed from the stateful entity or any other entity.  
        """
        # Check if the target is observable
        if not isinstance(target, Stateful):
            if observe_func is None:
                raise ValueError("The target is not observable and the observe function is not provided.")
            else:
                observation = await observe_func(target, **kwargs)
                return observation

        # Call the observe function of the target
        observation = await target.observe(**kwargs)
        return observation
        
    async def think(
        self, 
        observe: list[Union[AssistantMessage, ToolCallRequest, ToolCallResult]], 
        allow_tools: bool, 
        external_tools: dict[str, Union[FastMcpTool, MCPTool]] = {}, 
        tool_choice: str = None, 
        stream: bool = False, 
        queue: Optional[Queue] = None, 
        **kwargs, 
    ) -> AssistantMessage:
        """Think about the environment.
        
        Args:
            observe (list[Union[AssistantMessage, ToolCallRequest, ToolCallResult]]):
                The messages observed from the environment. 
            allow_tools (bool):
                Whether to allow the tools provided by the agent to be used. This do not affect the 
                external tools provided by the workflow. 
            external_tools (dict[str, Union[FastMcpTool, MCPTool]]):
                The external tools to use for the agent.  
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            **kwargs: 
                The additional keyword arguments for thinking about the observed messages. 

        Returns:
            AssistantMessage:
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
        message = await self.llm.completion(
            observe, 
            available_tools=available_tools, 
            tool_choice=tool_choice, 
            stream=stream, 
            queue=queue, 
            **kwargs
        )
        
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
    
    async def act(self, tool_call: ToolCallRequest, **kwargs) -> ToolCallResult:
        """Call a tool. 
        If there is any error caused by the tool call, the flag `is_error` will be set to True. 
        However, if there is any error caused by the MCP client connection, this should raise a RuntimeError.  
        
        Args:
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
        # Check if the tool call belongs to the agent
        if tool_call.name not in self.tools:
            if tool_call.name in self.env.tools:
                # Call the tool from the environment
                result = await self.env.call_tool(tool_call, **kwargs)
                return result
            elif tool_call.name in self.workflow.tools:
                # Call the tool from the workflow
                result = await self.workflow.call_tool(tool_call, **kwargs)
                return result
            else:
                raise ValueError(f"Tool {tool_call.name} is not registered to the agent or environment.")
        
        # Get the tool call id and the tool call arguments
        tool_call_id = tool_call.id
        tool_call_name = tool_call.name
        tool_call_args = tool_call.args
        
        try:
            # Call the tool 
            res = await self.mcp_client.call_tool(tool_call_name, tool_call_args)
        except ClientError as e:
            # Record and return the error
            logger.error(f"Error calling tool {tool_call_name}: {e}")
            return ToolCallResult(
                tool_call_id=tool_call_id, 
                content=str(e), 
                is_error=True, 
            )
        except RuntimeError as e:
            # Raise the error 
            logger.error(f"{e}, traceback: {traceback.format_exc()}")
            raise e
        
        # Return the result
        return ToolCallResult(
            tool_call_id=tool_call_id, 
            content=res.content, 
            is_error=res.is_error, 
        )

    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        *args, 
        **kwargs
    ) -> AssistantMessage:
        """Run the agent on the task or environment. Before running the agent, you should get the lock of the agent. 
        
        Args:
            target (Stateful): 
                The stateful entity to run the agent on. 
            max_error_retry (int, optional, defaults to 3): 
                The maximum number of times to retry the agent when the target is errored. 
            max_idle_thinking (int, optional, defaults to 1): 
                The maximum number of times to idle thinking the agent. 
            prompts (dict[str, str], optional, defaults to {}):
                The prompts of the agent. The key is the prompt name and the value is the prompt content. 
            completion_config (dict[str, Any], optional, defaults to {}):
                The completion config of the agent. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            AssistantMessage:
                The assistant message returned by the agent after running on the stateful entity or any other entity.
        """
        # Check if the workflow is registered
        if self.workflow is None:
            # Log the error
            logger.error("The workflow is not registered to the agent.")
            raise ValueError("The workflow is not registered to the agent.")
        
        # Check if the environment is registered
        if self.env is None:
            # Log the error
            logger.error("The environment is not registered to the agent.")
            raise ValueError("The environment is not registered to the agent.")
        
        # Get the lock of the agent
        await self.lock.acquire()
        
        # Call for running the workflow
        target = await self.workflow.run(
            target, 
            max_error_retry, 
            max_idle_thinking, 
            prompts, 
            completion_config, 
            *args, 
            **kwargs,
        )
        
        # Release the lock of the agent
        self.lock.release()
        
        return target

    def register_counter(self, counter: StepCounter) -> None:
        """Register a step counter to the base agent.
        
        Args:
            counter (StepCounter):
                The step counter to register.
        """
        self.step_counters[counter.uname] = counter


    def register_workflow(self, workflow: Workflow) -> None:
        """Register a workflow to the base agent.
        
        Args:
            workflow (Workflow):
                The workflow to register.
                
        Raises:
            ValueError:
                If the type of the workflow is not valid.
            ValueError:
                If the workflow is already registered to the agent.
        """
        # Check if the workflow is available
        if not isinstance(workflow, Workflow):
            raise ValueError(f"The type of the workflow is not valid. Expected Workflow, but got {type(workflow)}.")
        
        # Check if the workflow is already registered
        if self.workflow is not None:
            raise ValueError("The workflow is already registered to the agent.")
        
        # Register the workflow
        self.workflow = workflow
        
    def register_env(self, env: Environment) -> None:
        """Register an environment to the base agent.
        
        Args:
            env (Environment):
                The environment to register.
                
        Raises:
            ValueError:
                If the type of the environment is not valid.
            ValueError:
                If the environment is already registered to the agent.
        """
        # Check if the environment is available
        if not isinstance(env, Environment):
            raise ValueError(f"The type of the environment is not valid. Expected Environment, but got {type(env)}.")
        
        # Check if the environment is already registered
        if self.env is not None:
            raise ValueError("The environment is already registered to the agent.")
        
        # Register the environment
        self.env = env
