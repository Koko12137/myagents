import json

from json_repair import repair_json
from loguru import logger

from myagents.core.messages import UserMessage, SystemMessage
from myagents.core.interface import TreeTaskNode, CompletionConfig, Workflow, MemoryAgent
from myagents.core.workflows.react import BaseReActFlow
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.agents.memory import BaseMemoryOperation
from myagents.core.tasks import BaseTreeTaskNode
from myagents.prompts.workflows.memory import PROFILE, CREATE_MEMORY_EXTRACTION_TASK_PROMPT, MEMORY_KEY_RESULTS


class BaseMemoryWorkflow(BaseReActFlow):
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
        for memory_op in memories['todo']:
            # 获取 memory 的 operation
            operation = memory_op.pop("operation")
            # 获取 memory
            memory = memory_op.pop("memory")
            # 获取 memory 的 memory_type
            memory_type = memory["memory_type"]
            # 根据 memory_type 获取记忆类
            memory_class = self.agent.get_memory_classes()[memory_type.value]
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
            
        return target, False, True
        
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
        # Check if the target has history
        if len(target.get_history()) == 0:
            # Get the system prompt from the workflow
            system_prompt = self.prompts["system_prompt"]
            # Update the system prompt to the history
            await self.agent.prompt(SystemMessage(content=system_prompt), target)

        # Run the workflow
        while True:
            
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
        
            # === Reason Stage ===
            # Reason and act on the target
            target, _, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            # Check if the tool call flag is True
            if tool_call_flag:
                # Force the loop to break
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

    async def extract_memory(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 2, 
        **kwargs,
    ) -> TreeTaskNode:
        """从有状态实体中抽取记忆。"""
        
        # 将 target 的 history 转为 string
        history = ""
        for message in target.get_history():
            history += f"{message.role}: {message.content}\n"
        # 提取 Prompt
        prompt = CREATE_MEMORY_EXTRACTION_TASK_PROMPT.format(history=history)
        
        # 创建一个新的任务节点
        new_target = BaseTreeTaskNode(
            name="Memory Extraction", 
            objective=prompt,
            key_results=MEMORY_KEY_RESULTS, 
            sub_task_depth=0, 
        )
        
        # 运行记忆提取 workflow
        new_target = await self.run(
            target=new_target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            **kwargs,
        )
        # 返回更新后的目标
        return target
