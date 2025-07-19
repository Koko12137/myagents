import traceback
from uuid import uuid4
from asyncio import Lock
from typing import Union, Optional, Callable

from loguru import logger
from fastmcp import Client as MCPClient
from fastmcp.exceptions import ClientError
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.messages import AssistantMessage, ToolCallRequest, ToolCallResult, SystemMessage, UserMessage
from myagents.core.interface import LLM, Agent, StepCounter, Environment, Stateful, Workflow
from myagents.core.agents.types import AgentType
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
        tools (dict[str, FastMcpTool]):
            The tools to use for the agent.
        workflow (Workflow):
            The workflow to that the agent is running on.
        env (Environment):
            The environment to that the agent is running on.
        step_counters (dict[str, StepCounter]):
            The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
        lock (Lock):
            The synchronization lock of the agent. The agent can only work on one task at a time. 
            If the agent is running concurrently, the global context may not be working properly.
        prompts (dict[str, str]):
            The prompts for running the workflow. 
        observe_format (dict[str, str]):
            The format of the observation the target. 
    """
    # Basic information
    uid: str
    name: str
    agent_type: AgentType
    profile: str
    # LLM and MCP client
    llm: LLM
    mcp_client: MCPClient
    # Tools
    tools: dict[str, FastMcpTool]
    # Workflow and environment
    workflow: Workflow
    env: Environment
    # Step counters for the agent
    step_counters: dict[str, StepCounter]
    # Concurrency limit
    lock: Lock
    # Prompts and observe format
    prompts: dict[str, str]
    observe_format: dict[str, str]
    
    def __init__(
        self, 
        name: str, 
        agent_type: AgentType, 
        profile: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: Optional[MCPClient] = None, 
        prompts: dict[str, str] = None, 
        observe_format: dict[str, str] = None, 
        **kwargs, 
    ) -> None:
        """Initialize the BaseAgent.
        
        Args:
            name (str):
                The name of the agent.
            agent_type (AgentType):
                The type of the agent.
            profile (str):
                The profile of the agent.
            llm (LLM):
                The LLM to use for the agent. 
            step_counters (list[StepCounter]):
                The step counters to use for the agent. Any of one reach the limit, the agent will be stopped. 
            mcp_client (MCPClient, optional):
                The MCP client to use for the agent. If not provided, No MCP tools can be used.  
            prompts (dict[str, str], optional):
                The prompts for running specific workflow of the workflow. 
            observe_format (dict[str, str], optional):
                The format of the observation. 
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Initialize the basic information
        self.uid = str(uuid4())
        self.name = name
        self.agent_type = agent_type
        self.profile = profile
        # Initialize the LLM and MCP client
        self.llm = llm
        self.mcp_client = mcp_client
        self.tools = {}
        # Initialize the prompts and observe format
        self.prompts = prompts
        self.observe_format = observe_format
        # Initialize the workflow and environment
        self.workflow = None
        self.env = None
        # Initialize the step counters
        self.step_counters = {counter.uid: counter for counter in step_counters}
        # Initialize the synchronization lock
        self.lock = Lock()
        
    async def observe(
        self, 
        target: Stateful, 
        prompt: str, 
        observe_format: str, 
        **kwargs, 
    ) -> list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
        """Observe the target. A series of messages will be returned, including the system message, user message, 
        assistant message and tool call result. 
        
        Args:
            target (Stateful):
                The stateful entity to observe. 
            prompt (str): 
                The prompt instruction after the observation. 
            observe_format (str):
                The format of the observation. This must be a valid observe format of the target
            **kwargs:
                The additional keyword arguments for observing the target. 
            
        Returns:
            list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
                The up to date information observed from the stateful entity.  
        """
        # Check if the target is observable
        if not isinstance(target, Stateful):
            raise ValueError("The target is not observable.")

        # Call the observe function of the target
        observation = await target.observe(observe_format, **kwargs)
        # Create a new user message
        user_message = UserMessage(content=f"## 观察\n以下是观察到的信息:\n{observation}\n\n# 任务指令\n\n{prompt}")
        # Update the user_message to the target
        target.update(user_message)
        # Return the history of the target
        return target.get_history()
    
    async def think(
        self, 
        observe: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]], 
        completion_config: BaseCompletionConfig, 
        **kwargs, 
    ) -> AssistantMessage:
        """Think about the environment.
        
        Args:
            observe (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
                The messages observed from the environment. 
            completion_config (CompletionConfig):
                The completion config of the agent. 
            **kwargs: 
                The additional keyword arguments for thinking about the observed messages. 

        Returns:
            AssistantMessage:
                The completion message thought about by the LLM. 
        """
        # Check if the limit of the step counters is reached
        for step_counter in self.step_counters.values():
            await step_counter.check_limit()
        
        # Call for completion from the LLM
        message = await self.llm.completion(
            observe, 
            completion_config=completion_config, 
            **kwargs,
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
        completion_config: BaseCompletionConfig = None,
        running_checker: Callable[[Stateful], bool] = None,
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
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the agent. 
            running_checker (Callable[[Stateful], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
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

        # Initialize the tools
        if self.mcp_client is not None:
            # Get a new loop for running the coroutine
            tools = await self.mcp_client.list_tools()
            for tool in tools:
                self.tools[tool.name] = tool
        
        # Get the lock of the agent
        await self.lock.acquire()
        
        # Call for running the workflow
        target = await self.workflow.run(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            running_checker=running_checker, 
            **kwargs,
        )
        
        # Release the lock of the agent
        self.lock.release()
        
        # Observe the target    
        # BUG: 这里没有判断任务执行后的状态，如果任务执行后是error，应该返回error message
        observe = target.observe(self.observe_format["agent_format"])
        # Create a new assistant message
        message = AssistantMessage(content=f"{self.prompts['agent_format']}\n{observe}")
        # Log the message
        if logger.level == "DEBUG":
            logger.debug(f"Full Assistant Message: \n{message}")
        else:
            logger.info(f"Assistant Message: \n{message.content}")
        return message

    def register_counter(self, counter: StepCounter) -> None:
        """Register a step counter to the base agent.
        
        Args:
            counter (StepCounter):
                The step counter to register.
        """
        self.step_counters[counter.uid] = counter


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
