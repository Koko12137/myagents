import json

from json_repair import repair_json
from loguru import logger

from myagents.core.messages import UserMessage, SystemMessage
from myagents.core.interface import TreeTaskNode, CompletionConfig, Workflow, MemoryAgent
from myagents.core.workflows.react import BaseReActFlow
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.agents.memory import BaseMemoryOperation
from myagents.prompts.workflows.memory import PROFILE, SYSTEM_PROMPT, REASON_ACT_PROMPT, REFLECT_PROMPT


class MemoryWorkflow(BaseReActFlow):
    """MemoryWorkflow is a workflow for managing the memory of the agent.
    
    Attributes:
        context (Context):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
    """
    agent: MemoryAgent
    
    def __init__(
        self, 
        profile: str = PROFILE, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, Workflow] = {}, 
        **kwargs,
    ) -> None:
        """Initialize the MemoryWorkflow.
        
        Args:
            profile (str):
                The profile of the workflow.
            prompts (dict[str, str]):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
            observe_formats (dict[str, str]):
                The format of the observation. The key is the observation name and the value is the format content. 
            sub_workflows (dict[str, Workflow]):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
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
        
        super().__init__(
            profile=profile, 
            prompts=prompts, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
            **kwargs, 
        )
    
    async def reason_act(
        self, 
        target: TreeTaskNode, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on. 
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[TreeTaskNode, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = BaseCompletionConfig(format_json=True)
        else:
            # Update the format_json to True
            completion_config.update(format_json=True)
        
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
            
        # === Extract Memories ===
        # Extract episode and semantic memories
        memories = json.loads(repair_json(message.content))
        # 更新记忆
        for memory_op in memories:
            # 获取 memory 的 operation
            operation = memory_op.pop("operation")
            # 获取 memory 的 memory_type
            memory_type = memory_op.pop("memory_type")
            # 根据 memory_type 获取记忆类
            memory_class = self.agent.get_memory_classes()[memory_type.value]
            # 解析 Memory 内容
            memory = json.loads(memory_op.pop("memory"))
            # 获取当前的 env_id、agent_id、task_id、task_status
            env_id = self.env.uid
            agent_id = self.uid
            task_id = target.uid
            task_status = target.status.value
            # 更新 memory 的 env_id、agent_id、task_id、task_status
            memory["env_id"] = env_id
            memory["agent_id"] = agent_id
            memory["task_id"] = task_id
            memory["task_status"] = task_status
            
            # 获取嵌入向量
            embedding = await self.agent.embed(
                memory["content"], 
                dimensions=self.agent.get_vector_memory().get_dimension(),
            )
            # 更新 memory 的 embedding
            memory["embedding"] = embedding
            
            # 根据 memory_type 创建对应的记忆对象
            memory_op = BaseMemoryOperation(
                operation=operation,
                memory=memory_class(**memory),
            )
            # 更新记忆
            await self.agent.update_memory([memory_op])
        
    async def schedule(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Override the schedule method of the react workflow.
        
        Args:
            target (TreeTaskNode):
                The target to schedule.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for scheduling the workflow.
                
        Returns:
            TreeTaskNode:
                The target after scheduling.
                
        Raises:
            RuntimeError:
                If the target is not in the valid statuses.
        """
        if not target.is_created():
            # Log the error
            logger.error(f"Plan workflow requires the target status to be created, but the target status is {target.get_status().value}.")
            # Raise an error
            raise RuntimeError(f"Plan workflow requires the target status to be created, but the target status is {target.get_status().value}.")
            
        # Check if the target has history
        if len(target.get_history()) == 0:
            # Get the system prompt from the workflow
            system_prompt = self.prompts["system_prompt"]
            # Update the system prompt to the history
            await self.agent.prompt(SystemMessage(content=system_prompt), target)
        
        # This is used for no tool calling thinking limit.
        current_thinking = 0
        current_error = 0
        
        # Run the workflow
        while target.is_created():
            
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await super().reflect(
                target=target, 
                completion_config=completion_config, 
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
            
        return target

    async def run(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Run the workflow.
        
        Args:
            target (TreeTaskNode):
                The target to run the workflow on.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            TreeTaskNode:
                The target after running the workflow.
        """
        # Run the workflow
        return await self.schedule(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            **kwargs,
        )
