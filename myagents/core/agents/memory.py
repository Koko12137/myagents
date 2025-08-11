import uuid
import time
from typing import Union

from myagents.core.interface import Stateful, VectorMemoryCollection, EmbeddingLLM, MemoryWorkflow, VectorMemoryItem
from myagents.core.messages import AssistantMessage, ToolCallResult, SystemMessage, UserMessage
from myagents.core.agents.base import BaseAgent
from myagents.core.memories.schemas import (
    EpisodeMemoryItem, 
    MemoryType, 
    BaseMemoryOperation, 
    MemoryOperationType, 
)
from myagents.core.workflows import EpisodeMemoryFlow, MemoryCompressWorkflow


class BaseMemoryAgent(BaseAgent):
    """BaseMemoryAgent 是所有具备记忆能力的智能体基类。
    
    属性:
        workflow (MemoryWorkflow):
            记忆工作流
        memory_workflow (MemoryWorkflow):
            记忆提取工作流
        embedding_llm (EmbeddingLLM):
            嵌入语言模型
        vector_memory (VectorMemory):
            向量记忆，包含事实、知识、信息、数据等
    """
    # 替换 workflow 为 MemoryWorkflow
    workflow: MemoryWorkflow
    # 记忆提取 workflow
    memory_workflow: MemoryWorkflow
    # 嵌入语言模型
    embedding_llm: EmbeddingLLM
    # 向量记忆
    episode_memory: VectorMemoryCollection
    # 记忆提示词模板
    prompt_template: str
    
    def __init__(
        self, 
        episode_memory: VectorMemoryCollection, 
        embedding_llm: EmbeddingLLM, 
        # Memory Compress
        memory_compress_system_prompt: str, 
        memory_compress_reason_act_prompt: str, 
        # Episode Memory
        episode_memory_system_prompt: str, 
        episode_memory_reason_act_prompt: str, 
        episode_memory_reflect_prompt: str, 
        # Memory Format Template
        memory_prompt_template: str, 
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.episode_memory = episode_memory
        self.embedding_llm = embedding_llm
        
        # 初始化 EpisodeMemoryWorkflow
        episode_memory_workflow = EpisodeMemoryFlow(
            prompts={
                "system_prompt": episode_memory_system_prompt, 
                "reason_act_prompt": episode_memory_reason_act_prompt, 
                "reflect_prompt": episode_memory_reflect_prompt, 
            }, 
            observe_formats={
                "reason_act_format": "document", 
                "reflect_format": "document", 
            }, 
        )
        
        # 初始化 MemoryCompressWorkflow
        memory_workflow = MemoryCompressWorkflow(
            prompts={
                "system_prompt": memory_compress_system_prompt, 
                "reason_act_prompt": memory_compress_reason_act_prompt, 
            }, 
            observe_formats={
                "reason_act_format": "document", 
            }, 
            sub_workflows={
                "episode_memory_workflow": episode_memory_workflow, 
            }, 
        )
        
        self.prompt_template = memory_prompt_template
        # 记忆提取 workflow
        self.memory_workflow = memory_workflow
        # 注册 self 为 memory_workflow 的 agent
        self.memory_workflow.register_agent(self)
        
    def get_memory_workflow(self) -> MemoryWorkflow:
        """获取记忆工作流
        
        返回:
            MemoryWorkflow:
                记忆工作流
        """
        return self.memory_workflow
    
    def get_episode_memory(self) -> VectorMemoryCollection:
        """获取向量记忆
        
        返回:
            VectorMemoryCollection:
                向量记忆
        """
        return self.episode_memory
        
    async def embed(self, text: str, dimensions: int, **kwargs) -> list[float]:
        """嵌入文本
        
        参数:
            text (str):
                文本
            dimensions (int):
                嵌入维度
            **kwargs:
                额外参数
                
        返回:
            list[float]:
                嵌入向量
        """
        return await self.embedding_llm.embed(text, dimensions=dimensions, **kwargs)
    
    async def extract_memory(
        self, 
        target: Stateful, 
        **kwargs,
    ) -> Stateful:
        """
        从文本中抽取记忆。
        
        参数:
            target (Stateful):
                有状态实体
            **kwargs:
                其他参数
        返回:
            Stateful:
                更新后的有状态实体
        """
        # 获取历史上下文
        history = target.get_history()
        # 转为字符串
        history_str = "\n".join([f"{message.role}: {message.content}" for message in history])
        
        # 调用记忆提取 workflow
        await self.memory_workflow.extract_memory(history_str, **kwargs)
        # 清空旧的历史上下文
        target.reset()
        # 返回更新后的目标
        return target
        
    def create_memory(self, memory_type: str, **kwargs) -> VectorMemoryItem:
        """创建记忆
        
        参数:
            memory_type (str):
                记忆类型
        """
        # 检查是否存在 memory_id
        if "memory_id" not in kwargs:
            kwargs["memory_id"] = uuid.uuid4().hex
        # 检查是否存在 created_at
        if "created_at" not in kwargs:
            kwargs["created_at"] = int(time.time())
        
        if memory_type == MemoryType.EPISODE.value:
            return EpisodeMemoryItem(**kwargs)
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")
    
    async def update_memory(
        self, 
        memories: list[BaseMemoryOperation], 
        **kwargs,
    ) -> None:
        """更新记忆。"""
        for memory_op in memories:
            if memory_op.operation == MemoryOperationType.ADD:
                await self.episode_memory.add([memory_op.memory])
            elif memory_op.operation == MemoryOperationType.UPDATE:
                await self.episode_memory.update([memory_op.memory])
            elif memory_op.operation == MemoryOperationType.DELETE:
                await self.episode_memory.delete([memory_op.memory.memory_id])
        
    async def search_memory(
        self, 
        text: str, 
        limit: int, 
        score_threshold: float, 
        target: Stateful,  
        **kwargs,
    ) -> str:
        """从记忆中搜索信息。
        
        参数:
            text (str):
                文本
            limit (int):
                限制
            score_threshold (float):
                分数阈值
            **kwargs:
                其他参数
        """
        # 把 text 转换为向量
        embedding = await self.embedding_llm.embed(text, dimensions=self.episode_memory.get_dimension())
        # 从向量记忆中搜索 episode_memory
        episode_memories = await self.episode_memory.search(
            query_embedding=embedding, 
            top_k=limit, 
            score_threshold=score_threshold, 
            env_id=self.env.uid, 
            agent_id=self.uid, 
            task_id=target.uid, 
            task_status=target.status.value, 
        )
        # 将 dict 转为 MemoryItem 列表
        episode_memories = [EpisodeMemoryItem(**memory[0]) for memory in episode_memories]
        # 格式化记忆
        format_episode_memories: str = "\n".join([memory.format() for memory in episode_memories])
        # 拼接记忆
        format_memories = self.prompt_template.format(
            episode_memories=format_episode_memories,
        )
        # 返回格式化后的记忆
        return format_memories
    
    async def observe(
        self, 
        target: Stateful, 
        observe_format: str, 
        **kwargs,
    ) -> list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
        """观察目标对象，同时提取观察中的语义、情节和程序性记忆。
        
        参数:
            target (Stateful):
                需要观察的有状态实体。
            observe_format (str):
                观察信息的格式，必须为目标支持的格式。
            **kwargs:
                观察目标时的其他参数。
        返回:
            list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
                从有状态实体中获取的最新信息。
        """
        # 观察
        observation = await super().observe(target=target, observe_format=observe_format, **kwargs)
        
        # 转换为字符串
        observation_str = "\n".join([message.content for message in observation])
        # 提取记忆
        memories = await self.search_memory(
            text=observation_str, 
            limit=20, 
            score_threshold=0.1, 
            target=target, 
            **kwargs,
        )
        
        # 拼接记忆到 history 中
        await self.prompt(UserMessage(content=memories), target)
        # 返回历史信息
        return target.get_history()
