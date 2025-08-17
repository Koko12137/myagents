import re

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Workflow, Stateful, Workspace, TreeTaskNode, CompletionConfig, MemoryAgent, CallStack
from myagents.core.messages import SystemMessage, UserMessage, StopReason, ToolCallResult, ToolCallRequest
from myagents.core.messages.message import AssistantMessage
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.tasks import DocumentTaskView
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.react import PROFILE


class BaseReActFlow(BaseWorkflow):
    """BaseReActFlow implements the ReAct workflow interface.
    
    Attributes:
        workspace (Workspace):
            The workspace of the workflow.
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
    # Tools
    tools: dict[str, FastMcpTool]
    # Call stack
    call_stack: CallStack
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_formats: dict[str, str]
    # Sub-worflows
    sub_workflows: dict[str, Workflow]
    # Workspace
    workspace: Workspace
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        profile: str = PROFILE, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, Workflow] = {}, 
        **kwargs,
    ) -> None:
        """Initialize the BaseReActFlow.

        Args:
            call_stack (CallStack):
                The call stack of the workflow.
            workspace (Workspace):
                The workspace of the workflow.
            profile (str):
                The profile of the workflow.
            prompts (dict[str, str]):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
            observe_formats (dict[str, str]):
                The formats of the observation. The key is the observation name and the value is the format method name. 
            sub_workflows (dict[str, Workflow]):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
                sub-workflow instance. 
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
        if "reason_act_format" not in observe_formats:
            raise ValueError("The reason act format is required.")
        if "reflect_format" not in observe_formats:
            raise ValueError("The reflect format is required.")
        
        # Initialize the workflow
        super().__init__(
            call_stack=call_stack,
            workspace=workspace,
            profile=profile, 
            prompts=prompts, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
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
            target: TreeTaskNode = self.call_stack.get_value("target")
            # Get the tool call
            tool_call: ToolCallRequest = self.call_stack.get_value("tool_call")
            # Get the message
            message: AssistantMessage = target.get_history()[-1]
            
            if message.content is not None:
                # 检测并处理 finish_flag
                finish_flag_content = extract_by_label(message.content, "finish_flag", "finish_workflow", "finish")
                
                if finish_flag_content:
                    # 如果存在 finish_flag，检查值是否为 True
                    if finish_flag_content.strip() == "True":
                        # 已经是 True，无需修改
                        pass
                    else:
                        # 不是 True，强制修改为 True
                        message.content = re.sub(r"<finish_flag>.*?</finish_flag>", "<finish_flag>True</finish_flag>", message.content)
                else:
                    # 如果不存在 finish_flag，添加它
                    message.content += "\n<finish_flag>True</finish_flag>"
            else:
                # 如果 message.content 为 None，则设置 finish_flag 为 True
                message.content = "<finish_flag>True</finish_flag>"
            
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"已设置<finish_flag>True</finish_flag>，工作流即将结束。",
            )
            return result

    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> Stateful:
        """Run the agent on the target. Before running the agent, you should get the lock of the agent. 
        
        Args:
            target (Stateful):
                The target to run the agent on.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            Stateful:
                The target after working with the workflow.
        """
        return await self.schedule(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config,
            **kwargs,
        )
        
    async def schedule(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> Stateful:
        """Schedule the workflow. This method is used to schedule the workflow.
        
        Args:
            target (Stateful):
                The target to schedule.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for scheduling the workflow. 
                
        Returns:
            Stateful:
                The target after scheduling.
                
        Raises:
            RuntimeError:
                If the target is not in the valid statuses.
        """
        # Check if the target is running
        if not target.is_running():
            # Log the error
            logger.error(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
            # Raise an error
            raise RuntimeError(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
        
        # Get the system prompt from the workflow
        system_prompt = self.prompts["system_prompt"]
        # Update the system prompt to the history
        message = SystemMessage(content=system_prompt)
        await self.agent.prompt(message, target)
        
        # Initialize the error and idle thinking counter
        current_thinking = 0
        current_error = 0
        
        # Run the workflow
        while target.is_running():
        
            # === Reason Stage ===
            # Reason and act on the target
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
                # Increment the error counter
                current_error += 1
                # Notify the error limit to Agent
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the error message
                logger.info(f"Error Message: \n{message}")
                # Check if the error counter is greater than the max error retry
                if current_error >= max_error_retry:
                    # Set the task status to error
                    target.to_error()
                    # Force the react loop to finish
                    break
            
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await self.reflect(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag:
                # Set the task status to finished
                target.to_finished()
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Log the error message
                    logger.critical(f"连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。")
                    # Force the loop to break
                    break
            
        return target
    
    async def reason_act(
        self, 
        target: Stateful, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[Stateful, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on. 
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[Stateful, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Prepare external tools
        external_tools = []
        # Add the tools from the agent
        for tool in self.agent.tools.values():
            external_tools.append(tool)
        # Add the tools from the environment
        for tool in self.agent.env.tools.values():
            external_tools.append(tool)

        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = BaseCompletionConfig(tools=external_tools)
        else:
            # Update the completion config
            completion_config.update(tools=external_tools)
        
        # Initialize the error and tool call flag
        error_flag = False
        tool_call_flag = False
        
        # === Thinking ===
        # Prompt the agent
        await self.agent.prompt(UserMessage(content=self.prompts["reason_act_prompt"]), target)
        # Observe the target
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reason_act_format"])
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Think about the target
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # Update the message to the target
        await self.agent.prompt(message, target)
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
                await self.agent.prompt(result, target)
                # Check if the tool call is errored
                if result.is_error:
                    # Set the error flag to True
                    error_flag = True
        
        return target, error_flag, tool_call_flag

    async def reflect(
        self, 
        target: Stateful, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[Stateful, bool]:
        """Reflect on the target.
        
        Args:
            target (Stateful):
                The target to reflect on.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
        
        Returns:
            tuple[Stateful, bool]:
                The target and the finish flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = BaseCompletionConfig(tools=list(self.tools.values()))
        else:
            # Update the completion config
            completion_config.update(
                tools=list(self.tools.values()),
            )
        
        # === Thinking ===
        # Prompt the agent
        await self.agent.prompt(UserMessage(content=self.prompts["reflect_prompt"]), target)
        # Observe the target after acting
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reflect_format"])
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Reflect the action taken on the target
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # Update the message to the target
        await self.agent.prompt(message, target)
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
                    message=message,
                    **kwargs, 
                )
                # Log the tool call result
                logger.info(f"Tool Call Result: \n{result}")
                # Update the target with the tool call results
                await self.agent.prompt(result, target)
        
        # === Check Finish Flag ===
        # Check if the finish flag is set
        finish_flag = extract_by_label(message.content, "finish", "finish_flag", "finish_workflow")
        if finish_flag == "True" or finish_flag == "true":
            # Set the finish flag to True
            finish_flag = True
        else:
            # Set the finish flag to False
            finish_flag = False

        return target, finish_flag
    
    
class MemoryReActFlow(BaseReActFlow):
    """MemoryReActFlow is a workflow for the memory ReAct workflow.
    """
    agent: MemoryAgent
    
    def get_memory_agent(self) -> MemoryAgent:
        """Get the memory agent.
        
        Returns:
            MemoryAgent:
                The memory agent.
        """
        return self.agent
    
    async def extract_memory(
        self, 
        target: Stateful, 
        **kwargs,
    ) -> str:
        """从目标中提取记忆，将临时记忆清空，返回压缩后的记忆
        
        参数:
            target (Stateful):
                目标
            **kwargs:
                额外参数
                
        返回:
            str:
                压缩后的记忆
        """
        # Get the memory agent
        memory_agent = self.get_memory_agent()
        # Extract the memory from the target
        return await memory_agent.extract_memory(target, **kwargs)
        
    async def schedule(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> Stateful:
        """Schedule the workflow. This method is used to schedule the workflow.
        
        Args:
            target (Stateful):
                The target to schedule.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for scheduling the workflow. 
                
        Returns:
            Stateful:
                The target after scheduling.
                
        Raises:
            RuntimeError:
                If the target is not in the valid statuses.
        """
        # Check if the target is running
        if not target.is_running():
            # Log the error
            logger.error(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
            # Raise an error
            raise RuntimeError(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        # Run the workflow
        while target.is_running():
            
            # === Prepare System Instruction ===
            # Get the system prompt from the workflow
            if "system_prompt" not in self.prompts:
                raise KeyError("system_prompt not found in workflow prompts")
            system_prompt = self.prompts["system_prompt"]
            # Append the system prompt to the history
            message = SystemMessage(content=system_prompt)
            # Update the system message to the history
            await self.agent.prompt(message, target)
        
            # === Reason Stage ===
            # Reason and act on the target
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
                # Increment the error counter
                current_error += 1
                # Notify the error limit to Agent
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the error message
                logger.info(f"Error Message: \n{message}")
                # Check if the error counter is greater than the max error retry
                if current_error >= max_error_retry:
                    # Set the task status to error
                    target.to_error()
                    # Force the react loop to finish
                    break
                
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await self.reflect(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag:
                # Set the task status to finished
                target.to_finished()
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Log the error message
                    logger.critical(f"连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。")
                    # Force the loop to break
                    break
            
            # === Extract Memory ===
            # Extract the memory from the target
            compressed_memory = await self.extract_memory(target, **kwargs)
            # Update the compressed memory to the history
            await self.agent.update_temp_memory(temp_memory=compressed_memory, target=target)
            
        return target
    

class TreeTaskReActFlow(BaseReActFlow):
    """TreeTaskReActFlow is a workflow for the tree task ReAct workflow.
    """
    
    async def schedule(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> Stateful:
        """Schedule the workflow. This method is used to schedule the workflow.
        
        Args:
            target (Stateful):
                The target to schedule.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for scheduling the workflow. 
                
        Returns:
            Stateful:
                The target after scheduling.
                
        Raises:
            RuntimeError:
                If the target is not in the valid statuses.
        """
        # Check if the target is running
        if not target.is_running():
            # Log the error
            logger.error(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
            # Raise an error
            raise RuntimeError(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
        
        # Check if the target has history
        if len(target.get_history()) == 0:
            # === Prepare System Instruction ===
            # Get the prompts from the agent
            if "system_prompt" not in self.prompts:
                raise KeyError("system_prompt not found in workflow prompts")
            exec_system = self.prompts["system_prompt"]
            # Append the system prompt to the history
            message = SystemMessage(content=exec_system)
            # Update the system message to the history
            await self.agent.prompt(message, target)
            
            # Get the task from the context
            root_task = self.agent.env.workspace.get("root_task")
            # Create a UserMessage for the task results
            task_message = UserMessage(content=f"## 任务目前结果进度\n\n{DocumentTaskView(task=root_task).format()}")
            # Update the task message to the history
            await self.agent.prompt(task_message, target)
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        while target.is_running():
            # === Reason and Act ===
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
                # Increment the error counter
                current_error += 1
                # Notify the error limit to Agent
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the error message
                logger.info(f"Error Message: \n{message}")
                # Check if the error counter is greater than the max error retry
                if current_error >= max_error_retry:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    if len(target.get_history()) > 0:
                        target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    else:
                        target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}"
                    # Force the react loop to finish
                    break
            
            # Check if the tool call flag is not set
            if not tool_call_flag:
                # Get the last message
                message = target.get_history()[-1]
                # Extract the final output from the message
                final_output = extract_by_label(message.content, "final_output", "final answer", "output", "answer")
                if final_output != "":
                    # Set the answer of the task
                    target.results = final_output
                else:
                    # Announce the empty final output
                    logger.warning(f"Empty final output: \n{message.content}")
                    # Create a new user message to record the empty final output
                    message = UserMessage(content=f"【警告】：没有在<final_output>标签中找到任何内容，你必须将最终输出放在<final_output>标签中。")
                    await self.agent.prompt(message, target)
            
            # === Reflect ===
            target, finish_flag = await self.reflect(
                target=target, 
                completion_config=completion_config, 
            )
            # Check if the target is finished
            if finish_flag:
                # Set the task status to finished
                target.to_finished()
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag and not target.results:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请遵守反思结果，尽快输出最终输出。")
                await self.agent.prompt(message, target)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    target.results += f"\n连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。"
        
        # Set the answer of the task
        if not target.results and target.is_finished(): 
            target.results = "任务执行结束，但未提供答案，执行可能存在未知错误。"
            
        # Log the answer
        logger.info(f"任务执行结束: \n{DocumentTaskView(target).format()}")
        return target


class MemoryTreeTaskReActFlow(TreeTaskReActFlow):
    """MemoryTreeTaskReActFlow is a workflow for the memory tree task ReAct workflow.
    """
    agent: MemoryAgent
        
    def get_memory_agent(self) -> MemoryAgent:
        """Get the memory agent.
        
        Returns:
            MemoryAgent:
                The memory agent.
        """
        return self.agent
    
    async def extract_memory(
        self, 
        target: Stateful, 
        **kwargs,
    ) -> str:
        """从目标中提取记忆，将临时记忆清空，返回压缩后的记忆
        
        参数:
            target (Stateful):
                目标
            **kwargs:
                额外参数
                
        返回:
            str:
                压缩后的记忆
        """
        # Get the memory agent
        memory_agent = self.get_memory_agent()
        # Extract the memory from the target 
        return await memory_agent.extract_memory(target, **kwargs)

    
    async def schedule(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> Stateful:
        """Schedule the workflow. This method is used to schedule the workflow.
        
        Args:
            target (Stateful):
                The target to schedule.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for scheduling the workflow. 
                
        Returns:
            Stateful:
                The target after scheduling.
                
        Raises:
            RuntimeError:
                If the target is not in the valid statuses.
        """
        # Check if the target is running
        if not target.is_running():
            # Log the error
            logger.error(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
            # Raise an error
            raise RuntimeError(f"ReAct workflow requires the target status to be running, but the target status is {target.get_status().value}.")
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        while target.is_running():
            
            # === Prepare System Instruction ===
            # Get the prompts from the agent
            if "system_prompt" not in self.prompts:
                raise KeyError("system_prompt not found in workflow prompts")
            system_prompt = self.prompts["system_prompt"]
            # Append the system prompt to the history
            message = SystemMessage(content=system_prompt)
            # Update the system message to the history
            await self.agent.prompt(message, target)
            
            # Get the task from the context
            root_task = self.agent.env.workspace.get("root_task")
            # Create a UserMessage for the task results
            task_message = UserMessage(content=f"## 任务目前结果进度\n\n{DocumentTaskView(task=root_task).format()}")
            # Update the task message to the history
            await self.agent.prompt(task_message, target)
            
            # === Reason and Act ===
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
                # Increment the error counter
                current_error += 1
                # Notify the error limit to Agent
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # Log the error message
                logger.info(f"Error Message: \n{message}")
                # Check if the error counter is greater than the max error retry
                if current_error >= max_error_retry:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    if len(target.get_history()) > 0:
                        target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    else:
                        target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}"
                    # Force the react loop to finish
                    break
            
            # Check if the tool call flag is not set
            if not tool_call_flag:
                # Get the last message
                message = target.get_history()[-1]
                # Extract the final output from the message
                final_output = extract_by_label(message.content, "final_output", "final answer", "output", "answer")
                if final_output != "":
                    # Set the answer of the task
                    target.results = final_output
                else:
                    # Announce the empty final output
                    logger.warning(f"Empty final output: \n{message.content}")
                    # Create a new user message to record the empty final output
                    message = UserMessage(content=f"【警告】：没有在<final_output>标签中找到任何内容，你必须将最终输出放在<final_output>标签中。")
                    await self.agent.prompt(message, target)
            
            # === Reflect ===
            target, finish_flag = await self.reflect(
                target=target, 
                completion_config=completion_config, 
            )
            # Check if the target is finished
            if finish_flag:
                # Set the task status to finished
                target.to_finished()
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag and not target.results:
                # Increment the idle thinking counter
                current_thinking += 1
                # Notify the idle thinking limit to Agent
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请遵守反思结果，尽快输出最终输出。")
                await self.agent.prompt(message, target)
                # Log the idle thinking message
                logger.info(f"Idle Thinking Message: \n{message}")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    target.results += f"\n连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。"
                    # Force the loop to break
                    break
            
            # === Extract Memory ===
            # Extract the memory from the target
            compressed_memory = await self.extract_memory(target, **kwargs)
            # Update the compressed memory to the history
            await self.agent.update_temp_memory(temp_memory=compressed_memory, target=target)
        
        # Set the answer of the task
        if not target.results and target.is_finished(): 
            target.results = "任务执行结束，但未提供答案，执行可能存在未知错误。"
            
        # Log the answer
        logger.info(f"任务执行结束: \n{DocumentTaskView(target).format()}")
        return target
