from typing import Union

from myagents.core.interface import Stateful, VectorMemoryCollection, TableMemoryDB, EmbeddingLLM, MemoryWorkflow
from myagents.core.messages import AssistantMessage, ToolCallResult, SystemMessage, UserMessage
from myagents.core.agents.base import BaseAgent
from myagents.core.memories.schemas import (
    SemanticMemoryItem, EpisodeMemoryItem, MemoryType, BaseMemoryOperation, MemoryOperationType, 
)


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
        trajectory_memory (TableMemory):
            轨迹记忆，包含历史信息
    """
    # 替换 workflow 为 MemoryWorkflow
    workflow: MemoryWorkflow
    # 记忆提取 workflow
    memory_workflow: MemoryWorkflow
    # 嵌入语言模型
    embedding_llm: EmbeddingLLM
    # 向量记忆
    vector_memory: VectorMemoryCollection
    # 轨迹记忆
    trajectory_memory: TableMemoryDB
    # 记忆提示词模板
    prompt_template: str
    
    def __init__(
        self, 
        vector_memory: VectorMemoryCollection, 
        embedding_llm: EmbeddingLLM, 
        # trajectory_memory: TableMemoryDB, # TODO: 暂时不使用轨迹记忆
        memory_workflow: MemoryWorkflow, 
        memory_prompt_template: str, 
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vector_memory = vector_memory
        self.embedding_llm = embedding_llm
        # self.trajectory_memory = trajectory_memory # TODO: 暂时不使用轨迹记忆
        
        self.prompt_template = memory_prompt_template
        # 记忆提取 workflow
        self.memory_workflow = memory_workflow
        
    def get_memory_workflow(self) -> MemoryWorkflow:
        """获取记忆工作流
        
        返回:
            MemoryWorkflow:
                记忆工作流
        """
        return self.memory_workflow
    
    def get_vector_memory(self) -> VectorMemoryCollection:
        """获取向量记忆
        
        返回:
            VectorMemoryCollection:
                向量记忆
        """
        return self.vector_memory
        
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
        # 调用记忆提取 workflow
        target = await self.memory_workflow.extract_memory(target, **kwargs)
        # 返回更新后的目标
        return target
    
    async def update_memory(
        self, 
        memories: list[BaseMemoryOperation], 
        **kwargs,
    ) -> None:
        """更新记忆。"""
        for memory_op in memories:
            if memory_op.operation == MemoryOperationType.ADD:
                await self.vector_memory.add([memory_op.memory])
            elif memory_op.operation == MemoryOperationType.UPDATE:
                await self.vector_memory.update([memory_op.memory])
            elif memory_op.operation == MemoryOperationType.DELETE:
                await self.vector_memory.delete([memory_op.memory.memory_id])
        
    async def search_memory(
        self, 
        text: str, 
        limit: int, 
        score_threshold: float, 
        target: Stateful, 
        **kwargs,
    ) -> str:
        """从记忆中搜索信息。"""
        # 把 text 转换为向量
        embedding = await self.embedding_llm.embed(text, dimensions=self.vector_memory.get_dimension())
        # 从向量记忆中搜索 semantic_memory
        semantic_memories = await self.vector_memory.search(
            query_embedding=embedding, 
            top_k=limit, 
            score_threshold=score_threshold, 
            env_id=self.env.uid, 
            agent_id=self.uid, 
            task_id=target.uid, 
            task_status=target.status.value, 
            memory_type=MemoryType.SEMANTIC_MEMORY, 
        )
        # 将 dict 转为 MemoryItem 列表
        semantic_memories = [SemanticMemoryItem(**memory[0]) for memory in semantic_memories]
        # 格式化记忆
        format_semantic_memories: str = "\n".join([memory.format() for memory in semantic_memories])
        # 从向量记忆中搜索 episode_memory
        episode_memories = await self.vector_memory.search(
            query_embedding=embedding, 
            top_k=limit, 
            score_threshold=score_threshold, 
            env_id=self.env.uid, 
            agent_id=self.uid, 
            task_id=target.uid, 
            task_status=target.status.value, 
            memory_type=MemoryType.EPISODE_MEMORY, 
        )
        # 将 dict 转为 MemoryItem 列表
        episode_memories = [EpisodeMemoryItem(**memory[0]) for memory in episode_memories]
        # 格式化记忆
        format_episode_memories: str = "\n".join([memory.format() for memory in episode_memories])
        # 拼接记忆
        format_memories = self.prompt_template.format(
            semantic_memories=format_semantic_memories, 
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
        # 丢掉不必要的历史信息
        history = []
        if isinstance(observation[0], SystemMessage):
            history.append(observation[0])
            
        # 提取记忆
        memories = await self.search_memory(
            observation[-1].content, 
            target, 
            memory_type=MemoryType.SEMANTIC_MEMORY, 
            **kwargs,
        )
        
        # 拼接记忆到 history 中
        if isinstance(observation[-1], UserMessage):
            history.append(observation[-1])
            history[-1].content = history[-1].content + f"\n\n{memories}"
        else:
            memory_message = UserMessage(content=f"{memories}")
            history.append(memory_message)
        # 返回历史信息
        return history
