from typing import Callable, Any

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Workflow, Stateful, Context, TreeTaskNode, CompletionConfig
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.messages import SystemMessage, UserMessage, StopReason, ToolCallResult, ToolCallRequest
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.react import PROFILE


class BaseReActFlow(BaseWorkflow):
    """BaseReActFlow implements the ReAct workflow interface.
    
    Attributes:
        context (Context):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
        
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to run the workflow. 
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        observe_formats (dict[str, str]):
            The format of the observation. The key is the observation name and the value is the format content. 
        sub_workflows (dict[str, Workflow]):
            The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
            sub-workflow instance. 
    """
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_formats: dict[str, str]
    # Sub-worflows
    sub_workflows: dict[str, Workflow]
    
    def __init__(
        self, 
        profile: str = PROFILE, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, Workflow] = {}, 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the BaseReActFlow.

        Args:
            profile (str, optional):
                The profile of the workflow.
            prompts (dict[str, str], optional):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
            observe_formats (dict[str, str], optional):
                The formats of the observation. The key is the observation name and the value is the format method name. 
            sub_workflows (dict[str, Workflow], optional):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
                sub-workflow instance. 
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Check the prompts
        if "system_prompt" not in prompts:
            raise ValueError("The system prompt is required.")
        if "reason_act_prompt" not in prompts:
            raise ValueError("The reason act prompt is required.")
        if "reflect_prompt" not in prompts:
            raise ValueError("The reflect prompt is required.")
        
        # Check the observe formats
        if "reason_act" not in observe_formats:
            raise ValueError("The reason act format is required.")
        if "reflect" not in observe_formats:
            raise ValueError("The reflect format is required.")
        
        # Initialize the workflow
        super().__init__(
            profile=profile, 
            prompts=prompts, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
            *args, 
            **kwargs,
        )
    
    def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        
        This method will be called after the initialization of the workflow.
        """
        # Register the finish tool
        @self.register_tool("finish_workflow")
        async def finish_workflow() -> ToolCallResult:
            """
            完成当前任务，使用这个工具来结束工作流。
            
            Args:
                None
            
            Returns:
                ToolCallResult:
                    The tool call result.
            """
            # Get the target
            target: TreeTaskNode = self.context.get("target")
            # Get the tool call
            tool_call: ToolCallRequest = self.context.get("tool_call")
            # Get status update function
            status_update_func: Callable[[TreeTaskNode], None] = self.context.get("status_update_func", lambda target: target.to_finished())
            # Set the task status to finished
            status_update_func(target)
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
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Run the agent on the target. Before running the agent, you should get the lock of the agent. 
        
        Args:
            target (Stateful):
                The target to run the agent on.
            max_error_retry (int, optional):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            running_checker (Callable[[Stateful], bool], optional):
                The checker to check if the workflow should be running.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            Stateful:
                The target after working with the workflow.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default checker
            running_checker = lambda target: target.is_running()
        
        # Run the workflow
        if running_checker(target):
            # Reason and act on the target
            target = await self.reason_act_reflect(
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                running_checker=running_checker, 
                *args, 
                **kwargs,
            )
        else:
            # Log the error
            logger.error("The target is not running, the workflow is not executed.")
            # Set the target to error
            target.to_error()
            
        # Return the target
        return target

    async def reason_act_reflect(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on.
            max_error_retry (int, optional):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional):
                The maximum number of times to idle thinking the agent. 
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            running_checker (Callable[[Stateful], bool], optional):
                The checker to check if the workflow should be running.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent. 
        """
        if len(target.get_history()) == 0:
            # Get the system prompt from the agent
            system_prompt = self.prompts["system_prompt"]
            # Update system prompt to history
            message = SystemMessage(content=system_prompt)
            target.update(message)
        
        # Error and idle thinking control
        current_thinking = 0
        current_error = 0
        
        while running_checker(target):
        
            # === Reason Stage ===
            # Reason and act on the target
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                *args, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
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
                    # Record the error as answer
                    target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    # Force the react loop to finish
                    break
                
            # Check allow finish
            if tool_call_flag or error_flag:
                allow_finish = False
            else:
                allow_finish = True
            
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await self.reflect(
                target=target, 
                allow_finish=allow_finish, 
                completion_config=completion_config, 
                *args, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag:
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
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
                    # Record the error as answer
                    target.results += f"\n连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。"
                    # Log the error message
                    logger.critical(f"连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。")
                    # Force the loop to break
                    break
            
        return target
    
    async def reason_act(
        self, 
        target: Stateful, 
        completion_config: CompletionConfig = None, 
        *args, 
        **kwargs,
    ) -> tuple[Stateful, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on. 
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[Stateful, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = CompletionConfig()
        
        # Initialize the error and tool call flag
        error_flag = False
        tool_call_flag = False
        
        # Prepare external tools
        external_tools = {**self.agent.tools, **self.agent.env.tools}
        
        # === Instruction ===
        # Get the reason act prompt
        reason_act_prompt = self.prompts["reason_act_prompt"]
        # Create new user message with the reason act prompt
        message = UserMessage(content=reason_act_prompt)
        # Update the target with the user message
        target.update(message)
        
        # === Thinking ===
        # Observe the target
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reason_act"])
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Update the completion config
        completion_config.update(tools=external_tools)
        # Think about the target
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
        
        # === Act ===
        # Get all the tool calls from the assistant message
        if message.stop_reason == StopReason.TOOL_CALL:
            # Set the tool call flag to True
            tool_call_flag = True
            
            # Act on the task or environment
            for tool_call in message.tool_calls:
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
                    # Set the error flag to True
                    error_flag = True
        
        return target, error_flag, tool_call_flag

    async def reflect(
        self, 
        target: Stateful, 
        allow_finish: bool = False, 
        completion_config: CompletionConfig = None, 
        *args, 
        **kwargs,
    ) -> tuple[Stateful, bool]:
        """Reflect on the target.
        
        Args:
            target (Stateful):
                The target to reflect on.
            allow_finish (bool, optional):
                Whether to allow the workflow to finish.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
        
        Returns:
            tuple[Stateful, bool]:
                The target and the finish flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = CompletionConfig()
        
        # === Instruction ===
        # Get the reflect prompt
        reflect_prompt = self.prompts["reflect_prompt"]
        # Create new user message with the reflect prompt
        message = UserMessage(content=reflect_prompt)
        # Update the target with the user message
        target.update(message)
        # Check if the allow finish is set
        if not allow_finish:
            # Create a new user message announcing the finish not allowed
            message = UserMessage(content="\n\n**【注意】：前一阶段发生工具调用或存在错误，不允许直接结束。**")
            # Update the target with the user message
            target.update(message)
        
        # === Thinking ===
        # Observe the target after acting
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reflect"])
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Reflect the action taken on the target
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
            
        # === Act ===
        # Get all the tool calls from the assistant message
        if message.stop_reason == StopReason.TOOL_CALL:
            # Act on the task or environment
            for tool_call in message.tool_calls:
                # Act on the target
                result = await self.agent.act(
                    tool_call=tool_call, 
                    target=target, 
                    allow_finish=allow_finish, 
                    **kwargs, 
                )
                # Log the tool call result
                logger.info(f"Tool Call Result: \n{result}")
                # Update the target with the tool call results
                target.update(result)
        
        # === Check Finish Flag ===
        # Check if the finish flag is set
        finish_flag = extract_by_label(message.content, "finish", "finish_flag", "finish_workflow")
        if finish_flag == "True" or finish_flag == "true":
            if not allow_finish:
                # Create a new user message announcing the finish not allowed
                message = UserMessage(content="\n\n**【警告】：前一阶段发生工具调用或存在错误，不允许直接结束。**")
                # Update the target with the user message
                target.update(message)
                # Set the finish flag to False
                finish_flag = False
            else:
                # Set the finish flag to True
                finish_flag = True
        elif target.is_finished():
            # Set the finish flag to True
            finish_flag = True
        else:
            # Set the finish flag to False
            finish_flag = False

        return target, finish_flag
    
    def prepare_thinking_kwargs(
        self, 
        tools: dict[str, FastMcpTool] = {}, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> dict:
        """Prepare the thinking kwargs.
        
        Args:
            tools (dict[str, FastMcpTool], optional):
                The tools to use for the agent. 
            tool_choice (str, optional):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional):
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
        
        # Exclude the tools
        tools = {tool_name: tool for tool_name, tool in tools.items() if tool_name not in exclude_tools}
        
        # Set the tool choice
        if tool_choice is not None:
            # Check if the tool choice is in the tools
            assert tool_choice in tools, f"The tool choice {tool_choice} is not in the tools."
            # Set the tool choice
            arguments["tool_choice"] = tool_choice
            # Set the tools
            arguments["tools"] = {tool_choice: tools[tool_choice]}
        else:
            arguments["tools"] = tools
        
        return arguments
