from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Stateful
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.messages import SystemMessage, UserMessage, ToolCallResult, ToolCallRequest, StopReason
from myagents.core.utils.context import BaseContext
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.react import PROFILE, SYSTEM_PROMPT, THINK_PROMPT, REFLECT_PROMPT


class ReActFlow(BaseWorkflow):
    """Reason and Act Flow is the workflow for the react agent.
    
    Attributes:
        profile (str):
            The profile of the workflow.
        system_prompt (str):
            The system prompt of the workflow.
        agent (Agent):
            The agent that is used to reason and act. 
        context (BaseContext):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
    """
    # Basic information
    profile: str
    system_prompt: str
    agent: Agent
    # Context and tools
    context: BaseContext
    tools: dict[str, FastMcpTool]
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Initialize the workflow components
        self.profile = PROFILE
        self.system_prompt = SYSTEM_PROMPT
        self.agent = None
        
    async def post_init(self) -> None:
        
        @self.register_tool("finish")
        def finish() -> ToolCallResult:
            """
            完成当前任务，使用这个工具来结束工作流。
            
            Args:
                None
            
            Returns:
                ToolCallResult:
                    The tool call result.
            """
            # Get the target
            target: Stateful = self.context.get("target")
            # Get the tool call
            tool_call: ToolCallRequest = self.context.get("tool_call")
            # Set the task status to finished
            target.to_finished()
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"任务已设置为 {target.get_status().value} 状态。",
            )
            return result
        
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> Stateful:
        
        """Run the agent on the task or environment. Before running the agent, you should get the lock of the agent. 
        
        Args:
            target (Stateful):
                The task or environment to run the agent on.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent.
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            Stateful:
                The task or environment after working with the workflow.
        """

        # A while loop to run the workflow until the task is finished.
        while target.is_running():
            
            # Check if the target is finished
            if target.is_finished():
                return target
            
            # Check if the target is errored
            elif target.is_error():
                # Set the target to cancelled
                target.to_cancelled()
                # Return the target
                return target
                
            # Run the workflow
            else:
                await self.__reason_and_act(
                    target, 
                    max_error_retry, 
                    max_idle_thinking, 
                    tool_choice, 
                    exclude_tools, 
                    *args, 
                    **kwargs,
                )
                
        return target

    async def __reason_and_act(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> None:
        """Reason and act on the task or environment.
        
        Args:
            target (Stateful):
                The task or environment to reason and act on.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
        """
        # Check whether the target is observable or the `observe` function is provided
        if not hasattr(target, "observe"):
            # Log the critical error
            logger.critical("The target is not observable.")
            # Raise the error
            raise ValueError("The target is not observable.")
        
        # Update system prompt to history
        message = SystemMessage(content=self.system_prompt.format(profile=self.profile))
        target.update(message)
        
        # Error and idle thinking control
        current_thinking = 0
        current_error = 0
        
        while target.is_running():
        
            # === Reason Stage ===
            # Observe the target
            observe = await self.agent.observe(target)
            # Log the observe
            logger.info(f"Observe: \n{observe}")
            # Create new user message
            message = UserMessage(content=THINK_PROMPT.format(observe=observe))
            # Update the target with the user message
            target.update(message)
            # Prepare the thinking kwargs
            kwargs = self.__prepare_thinking_kwargs(tool_choice, exclude_tools, *args, **kwargs)
            # Think about the target
            message = await self.agent.think(observe, **kwargs)
            # Log the assistant message
            logger.info(f"Assistant Message: \n{message}")
            # Update the target with the assistant message
            target.update(message)
            
            # === Act Stage ===
            # Get all the tool calls from the assistant message
            if message.stop_reason == StopReason.TOOL_CALL:
                # Act on the task or environment
                for tool_call in message.tool_calls:
                    # Reset the idle thinking counter
                    current_thinking = 0
                    # Act on the target
                    result = await self.agent.act(
                        tool_call=tool_call, 
                        target=target, 
                        **kwargs, 
                    )
                    # Log the tool call result
                    logger.info(f"Tool Call Result: \n{result}")
                    # Update the target with the tool call results
                    target.update(result)
                    # Check if the tool call is errored
                    if result.is_error:
                        # Increment the error counter
                        current_error += 1
                        # Notify the error limit to Agent
                        message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                        target.update(message)
                        # Log the error message
                        logger.info(f"Error Message: \n{message}")
                        # Check if the error counter is greater than the max error retry
                        if current_error >= max_error_retry:
                            # Set the task status to error
                            target.to_error()
                            # Force the react loop to finish
                            break
            else:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
                target.update(message)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Force the react loop to finish
                    break
            
            # === Reflect Stage ===
            # Observe the target after acting
            observe = await self.agent.observe(target)
            # Log the observe
            logger.info(f"Observe: \n{observe}")
            # Create new user message
            message = UserMessage(content=REFLECT_PROMPT.format(observe=observe))
            # Update the target with the user message
            target.update(message)
            # Reflect the action taken on the target
            message = await self.agent.think(observe, tools=self.tools)
            # Log the assistant message
            logger.info(f"Assistant Message: \n{message}")
            # Update the target with the assistant message
            target.update(message)
            
            # === Finish Stage ===
            # Check if the finish flag is set
            finish_flag = extract_by_label(message.content, "finish", "finish_flag")
            if finish_flag == "True":
                # Set the task status to finished
                target.to_finished()

    def __prepare_thinking_kwargs(
        self, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> dict:
        """Prepare the thinking kwargs.
        
        Args:
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            dict:
                The thinking kwargs.
        """
        # Prepare the thinking kwargs
        arguments = {}
        
        # External tools including the tools from the agent and the environment
        external_tools = {**self.agent.tools, **self.env.tools}
        # Exclude the tools
        external_tools = {tool_name: tool for tool_name, tool in external_tools.items() if tool_name not in exclude_tools}
        
        # Set the tool choice
        if tool_choice is not None:
            # Convert the tool choice to a FastMcpTool
            tool = external_tools.get(tool_choice, tool_choice)
            arguments["tool_choice"] = tool
            arguments["tools"] = [tool]
        else:
            arguments["tools"] = external_tools
        
        return arguments
