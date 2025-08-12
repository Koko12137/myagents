import traceback
from uuid import uuid4
from asyncio import Lock
from typing import Union

from loguru import logger
from fastmcp import Client as MCPClient
from fastmcp.exceptions import ClientError
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import LLM, Agent, StepCounter, Environment, Stateful, Workflow
from myagents.core.messages import AssistantMessage, ToolCallRequest, ToolCallResult, SystemMessage, UserMessage
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.agents.types import AgentType
from myagents.core.utils.step_counters import MaxStepsError


class BaseAgent(Agent):
    """BaseAgent 是所有智能体的基类，可以：
    - 观察环境或任务。
    - 思考环境或任务。
    - 调用工具。
    - 运行智能体。
    
    属性：
        uid (int):
            智能体的唯一标识符。
        name (str):
            智能体名称。
        type (AgentType):
            智能体类型。
        profile (str):
            智能体简介。
        llm (LLM):
            智能体使用的大语言模型。
        mcp_client (MCPClient):
            智能体使用的 MCP 客户端。
        tools (dict[str, FastMcpTool]):
            智能体可用的工具。
        workflow (Workflow):
            智能体运行的工作流。
        env (Environment):
            智能体运行的环境。
        step_counters (dict[str, StepCounter]):
            智能体的步数计数器，任一计数器达到上限时，智能体将停止。
        lock (Lock):
            智能体的同步锁，保证同一时刻只能处理一个任务。
            若并发运行，可能导致全局上下文异常。
        prompts (dict[str, str]):
            工作流运行时用到的提示词。
        observe_format (dict[str, str]):
            观察目标时的信息格式。
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
    observe_formats: dict[str, str]
    
    def __init__(
        self, 
        name: str, 
        agent_type: AgentType, 
        profile: str, 
        llm: LLM, 
        step_counters: list[StepCounter], 
        mcp_client: MCPClient = None, 
        prompts: dict[str, str] = None, 
        observe_formats: dict[str, str] = None, 
        **kwargs, 
    ) -> None:
        """初始化 BaseAgent。
        
        参数：
            name (str):
                智能体名称。
            agent_type (AgentType):
                智能体类型。
            profile (str):
                智能体简介。
            llm (LLM):
                智能体使用的大语言模型。
            step_counters (list[StepCounter]):
                智能体的步数计数器，任一计数器达到上限时，智能体将停止。
            mcp_client (MCPClient):
                智能体使用的 MCP 客户端。若未提供，则无法使用 MCP 工具。
            prompts (dict[str, str]):
                运行特定工作流时用到的提示词。
            observe_formats (dict[str, str]):
                观察信息的格式。
            **kwargs:
                传递给父类的其他参数。
        """
        # Initialize the parent class
        super().__init__(**kwargs)
        
        # Initialize the basic information
        self.uid = uuid4().hex
        self.name = name
        self.agent_type = agent_type
        self.profile = profile
        # Initialize the LLM and MCP client
        self.llm = llm
        self.mcp_client = mcp_client
        self.tools = {}
        # Initialize the workflow and environment
        self.workflow = None
        self.env = None
        # Initialize the step counters
        self.step_counters = {counter.uid: counter for counter in step_counters}
        # Initialize the synchronization lock
        self.lock = Lock()
        # Initialize the prompts and observe format
        self.prompts = prompts
        self.observe_formats = observe_formats
        
    async def prompt(
        self, 
        prompt: Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult], 
        target: Stateful, 
        **kwargs,
    ) -> None:
        """环境向智能体发送提示
        
        参数:
            prompt (Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]):
                提示信息
            target (Stateful):
                提示的目标
            **kwargs:
                提示的额外关键字参数
        """
        if isinstance(prompt, SystemMessage):
            if len(target.get_history()) == 0:
                # Update the system message to the target
                target.update(prompt)
        elif isinstance(prompt, UserMessage):
            # Update the user message to the target
            target.update(prompt)
        elif isinstance(prompt, AssistantMessage):
            # Update the assistant message to the target
            target.update(prompt)
        elif isinstance(prompt, ToolCallResult):
            # Update the tool call result to the target
            target.update(prompt)
        else:
            raise ValueError(f"The type of the prompt is not valid. Expected SystemMessage, UserMessage, AssistantMessage, or ToolCallResult, but got {type(prompt)}.")
    
    async def observe(
        self, 
        target: Stateful, 
        observe_format: str, 
        **kwargs, 
    ) -> list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
        """观察目标对象。返回一系列消息，包括系统消息、用户消息、助手消息和工具调用结果。
        
        参数：
            target (Stateful):
                需要观察的有状态实体。
            observe_format (str):
                观察信息的格式，必须为目标支持的格式。
            **kwargs:
                观察目标时的其他参数。
        返回：
            list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
                从有状态实体中获取的最新信息。
        """
        # Check if the target is observable
        if not isinstance(target, Stateful):
            raise ValueError("The target is not observable.")

        # Call the observe function of the target
        observation = await target.observe(observe_format, **kwargs)
        # Create a new user message
        user_message = UserMessage(content=f"## 观察\n以下是观察到的信息:\n{observation}")
        # Update the user_message to the target
        await self.prompt(user_message, target)
        # Return the history of the target
        return target.get_history()
    
    async def think(
        self, 
        observe: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]], 
        completion_config: BaseCompletionConfig, 
        **kwargs, 
    ) -> AssistantMessage:
        """对环境进行思考。
        
        参数：
            observe (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
                从环境中观察到的消息。
            completion_config (CompletionConfig):
                智能体的补全配置。
            **kwargs:
                其他思考参数。
        返回：
            AssistantMessage:
                LLM 生成的思考回复。
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
                logger.error(f"Unexpected error in step counter: {e}")
                raise e
        
        if len(errors) > 0:
            raise errors[0] from errors[0]
        
        # Return the response
        return message
    
    async def act(self, tool_call: ToolCallRequest, **kwargs) -> ToolCallResult:
        """调用工具。
        
        参数：
            tool_call (ToolCallRequest):
                工具调用请求，包括工具调用 id 和参数。
            **kwargs:
                其他调用参数。
        返回：
            ToolCallResult:
                工具调用结果。
        异常：
            RuntimeError:
                MCP 客户端连接异常。
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
                # 记录错误
                logger.error(f"Tool {tool_call.name} is not registered to the agent or environment.")
                # 返回错误
                return ToolCallResult(
                    tool_call_id=tool_call.id,
                    content=f"Tool {tool_call.name} is not registered to the agent or environment.",
                    is_error=True,
                )
        
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
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: BaseCompletionConfig = None, 
        **kwargs
    ) -> AssistantMessage:
        """在任务或环境上运行智能体。运行前需获取智能体锁。
        
        参数：
            target (Stateful):
                运行目标。
            max_error_retry (int):
                目标出错时最大重试次数。
            max_idle_thinking (int):
                最大空闲思考次数。
            completion_config (CompletionConfig, 默认为 None):
                智能体补全配置。
            **kwargs:
                其他运行参数。
        返回：
            AssistantMessage:
                智能体在目标上运行后的回复。
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
        
        try:
            # Call for running the workflow
            target = await self.workflow.run(
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                **kwargs,
            )
        finally:
            # Release the lock of the agent
            self.lock.release()
        
        # Observe the target    
        # Check if the target is in error state
        if target.is_error():
            error_message = AssistantMessage(content="## 任务执行失败\n\n任务执行过程中发生错误，请检查任务配置和输入。")
            logger.error(f"{str(self)}: Task execution failed")
            return error_message
            
        observe = await target.observe(self.observe_formats["agent_format"])
        # Create a new assistant message
        message = AssistantMessage(content=f"## 任务执行结果\n\n{observe}")
        # Log the message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self)}: \n{message}")
        else:
            logger.info(f"{str(self)}: \n{message.content}")
        return message

    def register_counter(self, counter: StepCounter) -> None:
        """为智能体注册步数计数器。
        
        参数：
            counter (StepCounter):
                要注册的步数计数器。
        """
        self.step_counters[counter.uid] = counter


    def register_workflow(self, workflow: Workflow) -> None:
        """为智能体注册工作流。
        
        参数：
            workflow (Workflow):
                要注册的工作流。
        异常：
            ValueError:
                workflow 类型不正确。
        """
        # Check if the workflow is available
        if not isinstance(workflow, Workflow):
            raise ValueError(f"The type of the workflow is not valid. Expected Workflow, but got {type(workflow)}.")
        
        # Register the workflow
        self.workflow = workflow
        
    def register_env(self, env: Environment) -> None:
        """为智能体注册环境。
        
        参数：
            env (Environment):
                要注册的环境。
        异常：
            ValueError:
                env 类型不正确。
        """
        # Check if the environment is available
        if not isinstance(env, Environment):
            raise ValueError(f"The type of the environment is not valid. Expected Environment, but got {type(env)}.")
        
        # Register the environment
        self.env = env
