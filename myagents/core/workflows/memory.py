import json
import traceback

from json_repair import repair_json
from loguru import logger

from myagents.core.messages import UserMessage, SystemMessage
from myagents.core.interface import TreeTaskNode, CompletionConfig, Workflow, MemoryAgent, MemoryWorkflow, Workspace, CallStack
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.memories.schemas import BaseMemoryOperation, MemoryOperationType
from myagents.core.tasks import BaseTreeTaskNode
from myagents.prompts.memories.compress import PROFILE
from myagents.prompts.memories.episode import PROFILE as EPISODE_PROFILE


class EpisodeMemoryFlow(BaseWorkflow):
    """提取事件记忆(Episode Memory)工作流
    """
    
    agent: MemoryAgent
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        profile: str = EPISODE_PROFILE, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, Workflow] = {}, 
        **kwargs,
    ) -> None:
        """初始化记忆提取工作流
        
        Args:
            profile (str):
                工作流的配置文件
            prompts (dict[str, str]):
                工作流的提示词。键是提示词名称，值是提示词内容
            observe_formats (dict[str, str]):
                观察的格式。键是观察名称，值是格式内容
            sub_workflows (dict[str, Workflow]):
                子工作流。键是子工作流名称，值是子工作流对象
            **kwargs:
                传递给父类的关键字参数
        """
        # 检查提示词
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
            call_stack=call_stack,
            workspace=workspace,
            profile=profile, 
            prompts=prompts, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
            **kwargs, 
        )
        
    def post_init(self) -> None:
        """初始化工作流
        """
        pass
        
    def get_memory_agent(self) -> MemoryAgent:
        """获取当前工作流的记忆智能体
        
        Returns:
            MemoryAgent:
                记忆智能体
        """
        return self.agent
    
    async def reason_act(
        self, 
        target: TreeTaskNode, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool, bool]:
        """对目标进行记忆提取
        
        Args:
            target (TreeTaskNode):
                要提取记忆的目标
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                运行代理的额外关键字参数
                
        Returns:
            TreeTaskNode:
                提取记忆后的目标
            bool:
                是否提取成功
            bool:
                是否工具调用
        """
        # 初始化错误和工具调用标志
        error_flag = False
        tool_call_flag = False
        
        # 检查是否提供了完成配置
        if completion_config is None:
            # 将完成配置设置为默认完成配置
            completion_config = BaseCompletionConfig(format_json=True)
        else:
            # 将format_json更新为True
            completion_config.update(format_json=True)
        
        # === 提取阶段 ===
        # 提示代理
        await self.agent.prompt(UserMessage(content=self.prompts["reason_act_prompt"]), target)
        # 观察目标
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reason_act_format"])
        # 记录观察结果
        logger.info(f"观察结果: \n{observe[-1].content}")
        # 思考目标
        message = await self.agent.think_extract(observe=observe, completion_config=completion_config)
        # 将消息更新到目标
        await self.agent.prompt(message, target)
        # 记录助手消息
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
        
        # === 提取记忆 ===
        try:
            # 提取情节和语义记忆
            memory = json.loads(repair_json(message.content))['memory']
            # 检测 is_error 是否为 str 类型
            if isinstance(memory["is_error"], str):
                memory["is_error"] = True if memory["is_error"].lower() == "true" else False
            
            # 设定记忆的操作类型为添加
            operation = MemoryOperationType.ADD
            # 获取当前的env_id、agent_id、task_id、task_status
            memory["env_id"] = self.agent.env.uid
            memory["agent_id"] = self.agent.uid
            memory["task_id"] = target.uid
            memory["task_status"] = target.status.value
            # 获取嵌入向量
            embedding = await self.agent.embed(
                f"Abstract: {memory['abstract']}\nKeywords: {memory['keywords']}", 
                dimensions=self.agent.get_memory_collection(memory_type="episode").get_dimension(),
            )
            # 更新记忆的嵌入向量
            memory["embedding"] = embedding
            # 更新记忆
            memory_op = BaseMemoryOperation(
                operation=operation,
                memory=self.agent.create_memory(memory_type="episode", **memory),
            )
            # 更新记忆
            await self.agent.update_memory([memory_op])
            
            # === 更新历史 ===
            op_info = memory_op.model_dump()
            op_info["memory"].pop("embedding")
            # 新建 UserMessage 声明记忆更新成功
            message = UserMessage(content=f"记忆更新成功: {op_info}")
            # 更新记忆
            await self.agent.prompt(message, target)
            # 记录用户消息
            logger.info(f"用户消息: \n{message.content}")
            
            # 设置工具调用标志
            tool_call_flag = True
                
        except Exception as e:
            # 设置错误标志
            error_flag = True
            # 记录错误
            logger.error(f"提取记忆失败: {e}", traceback.format_exc())
            # 新建 UserMessage 声明记忆更新失败
            message = UserMessage(content=f"记忆更新失败: {e}")
            # 更新记忆
            await self.agent.prompt(message, target)
            # 记录用户消息
            logger.error(f"用户消息: \n{message}")
            
        return target, error_flag, tool_call_flag
        
    async def schedule(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """重写react工作流的schedule方法
        
        Args:
            target (TreeTaskNode):
                要调度的目标
            max_error_retry (int):
                目标出错时重试代理的最大次数
            max_idle_thinking (int):
                代理空闲思考的最大次数
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                调度工作流的额外关键字参数
                
        Returns:
            TreeTaskNode:
                调度后的目标
                
        Raises:
            RuntimeError:
                如果目标不在有效状态中
        """
        # 检查目标是否有历史记录
        if len(target.get_history()) == 0:
            # 从工作流获取系统提示词
            if "system_prompt" not in self.prompts:
                raise KeyError("system_prompt not found in workflow prompts")
            
            system_prompt = self.prompts["system_prompt"]
            # 将系统提示词更新到历史记录
            await self.agent.prompt(SystemMessage(content=system_prompt), target)
        
        # 初始化空闲思考计数器
        current_thinking = 0
        # 初始化错误计数器
        current_error = 0

        # 运行工作流
        while True:
            
            # === 推理阶段 ===
            # 对目标进行推理和行动
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            
            # === 检查是否设置了工具调用标志 ===
            if tool_call_flag and not error_flag:
                # 强制react循环结束
                break
            
            # 检查是否设置了错误标志
            if error_flag:
                # 增加错误计数器
                current_error += 1
                # 通知代理错误限制
                message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # 记录错误消息
                logger.info(f"错误消息: \n{message}")
                # 检查错误计数器是否大于最大错误重试次数
                if current_error >= max_error_retry:
                    # 将任务状态设置为错误
                    target.to_error()
                    # 将错误记录为答案
                    if len(target.get_history()) > 0:
                        target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    else:
                        target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}"
                    # 强制react循环结束
                    break
            
            # 检查工具调用标志是否为 True
            elif not tool_call_flag:
                # 增加空闲思考计数器
                current_thinking += 1
                # 通知代理空闲思考限制
                message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
                await self.agent.prompt(message, target)
                # 记录空闲思考消息
                logger.info(f"空闲思考消息: \n{message}")
                # 检查空闲思考计数器是否大于最大空闲思考次数
                if current_thinking >= max_idle_thinking:
                    # 将任务状态设置为错误
                    target.to_error()
                    # 将错误记录为答案
                    if len(target.get_history()) > 0:
                        target.results += f"\n\n空闲思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，空闲思考原因: {target.get_history()[-1].content}"
                    else:
                        target.results += f"\n\n空闲思考次数限制已达上限: {current_thinking}/{max_idle_thinking}"
                    # 强制跳出循环
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
        """运行工作流
        
        Args:
            target (TreeTaskNode):
                要运行工作流的目标
            max_error_retry (int):
                目标出错时重试代理的最大次数
            max_idle_thinking (int):
                代理空闲思考的最大次数
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                运行工作流的额外关键字参数
                
        Returns:
            TreeTaskNode:
                运行工作流后的目标
        """
        # 运行工作流
        return await self.schedule(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            **kwargs,
        )

    async def extract_memory(
        self, 
        text: str, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 2, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> str:
        """从有状态实体中抽取记忆
        
        Args:
            text (str):
                要从中提取记忆的文本
            max_error_retry (int):
                目标出错时重试代理的最大次数
            max_idle_thinking (int):
                代理空闲思考的最大次数
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                运行工作流的额外关键字参数
                
        Returns:
            str:
                提取的记忆
        """
        # 创建一个新的任务节点
        new_target = BaseTreeTaskNode(
            name="记忆提取", 
            objective=text,
            key_results="", 
            sub_task_depth=0, 
        )
        
        # 运行记忆提取工作流
        new_target = await self.run(
            target=new_target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            **kwargs,
        )
        return new_target.get_history()[-1].content

class MemoryCompressWorkflow(BaseWorkflow):
    """压缩记忆(Memory Compress)工作流
    """
    agent: MemoryAgent
    sub_workflows: dict[str, MemoryWorkflow]
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        profile: str = PROFILE, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, MemoryWorkflow] = {}, 
        **kwargs,
    ) -> None:
        """初始化压缩记忆工作流
        
        Args:
            call_stack (CallStack):
                工作流的调用栈
            workspace (Workspace):
                工作流的 workspace
            profile (str):
                工作流的配置文件
            prompts (dict[str, str]):
                工作流的提示词。键是提示词名称，值是提示词内容
            observe_formats (dict[str, str]):
                观察的格式。键是观察名称，值是格式内容
            sub_workflows (dict[str, MemoryWorkflow]):
                子工作流。键是子工作流名称，值是子工作流对象
            **kwargs:
                传递给父类的关键字参数
        """
        # 检查提示词
        if "system_prompt" not in prompts:
            raise ValueError("The system prompt is required.")
        if "reason_act_prompt" not in prompts:
            raise ValueError("The reason act prompt is required.")
        # Check the observe formats
        if "reason_act_format" not in observe_formats:
            raise ValueError("The reason act format is required.")
        
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
        """初始化工作流
        """
        pass
        
    def get_memory_agent(self) -> MemoryAgent:
        """获取当前工作流的记忆智能体
        
        Returns:
            MemoryAgent:
                记忆智能体
        """
        return self.agent
    
    async def compress(
        self, 
        target: TreeTaskNode, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """对目标进行压缩
        
        Args:
            target (TreeTaskNode):
                要压缩的目标
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                运行代理的额外关键字参数
                
        Returns:
            TreeTaskNode:
                压缩后的目标
        """
        # 检查是否提供了完成配置
        if completion_config is None:
            # 将完成配置设置为默认完成配置
            completion_config = BaseCompletionConfig()
        else:
            # 将format_json更新为True
            completion_config.update()
        
        # === 思考阶段 ===
        # 提示代理
        await self.agent.prompt(UserMessage(content=self.prompts["reason_act_prompt"]), target)
        # 观察目标
        observe = await self.agent.observe(target, observe_format=self.observe_formats["reason_act_format"])
        # 记录观察结果
        logger.info(f"观察结果: \n{observe[-1].content}")
        # 思考目标
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # 将消息更新到目标
        await self.agent.prompt(message, target)
        # 记录助手消息
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
            
        return target
        
    async def schedule(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """重写react工作流的schedule方法
        
        Args:
            target (TreeTaskNode):
                要调度的目标
            max_error_retry (int):
                目标出错时重试代理的最大次数
            max_idle_thinking (int):
                代理空闲思考的最大次数
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                调度工作流的额外关键字参数
                
        Returns:
            TreeTaskNode:
                调度后的目标
                
        Raises:
            RuntimeError:
                如果目标不在有效状态中
        """
        # 检查目标是否有历史记录
        if len(target.get_history()) == 0:
            # 从工作流获取系统提示词
            if "system_prompt" not in self.prompts:
                raise KeyError("system_prompt not found in workflow prompts")
            
        system_prompt = self.prompts["system_prompt"]
        # 将系统提示词更新到历史记录
        await self.agent.prompt(SystemMessage(content=system_prompt), target)
    
        # === 推理阶段 ===
        # 对目标进行推理和行动
        target = await self.compress(
            target=target, 
            completion_config=completion_config, 
            **kwargs,
        )
        
        # === 提取阶段 === 
        # 提取记忆
        # for workflow in self.sub_workflows.values():
        #     await workflow.extract_memory(
        #         text=target.objective, 
        #         max_error_retry=max_error_retry, 
        #         max_idle_thinking=max_idle_thinking, 
        #         completion_config=completion_config, 
        #         **kwargs,
        #     )
        
        return target

    async def run(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """运行工作流
        
        Args:
            target (TreeTaskNode):
                要运行工作流的目标
            max_error_retry (int):
                目标出错时重试代理的最大次数
            max_idle_thinking (int):
                代理空闲思考的最大次数
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                运行工作流的额外关键字参数
                
        Returns:
            TreeTaskNode:
                运行工作流后的目标
        """
        # 运行工作流
        return await self.schedule(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            **kwargs,
        )

    async def extract_memory(
        self, 
        text: str, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 2, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> str:
        """从文本中提取记忆
        
        Args:
            text (str):
                要从中提取记忆的文本
            max_error_retry (int):
                目标出错时重试代理的最大次数
            max_idle_thinking (int):
                代理空闲思考的最大次数
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                运行工作流的额外关键字参数
                
        Returns:
            str:
                提取的记忆
        """
        # 创建一个新的任务节点
        new_target = BaseTreeTaskNode(
            name="记忆提取",
            objective=text,
            key_results="", 
            sub_task_depth=0, 
        )
        
        # 运行记忆提取工作流
        new_target = await self.run(
            target=new_target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            **kwargs,
        )
        return new_target.get_history()[-1].content
