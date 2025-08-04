import json
from typing import Union

from myagents.core.interface import Stateful,  VectorMemoryCollection, TableMemoryDB, EmbeddingLLM
from myagents.core.messages import AssistantMessage, ToolCallResult, SystemMessage, UserMessage
from myagents.core.llms import BaseCompletionConfig
from myagents.core.agents.base import BaseAgent
from myagents.core.memories.schemas import (
    SemanticMemoryItem, EpisodeMemoryItem, ProceduralMemoryItem, MemoryType, BaseMemoryOperation, MemoryOperationType, 
)


PROMPT_TEMPLATE = """
## 程序性记忆

- 程序性记忆是关于操作、步骤、方法等的记忆。
=============
以下是程序性记忆：
{procedural_memories}

## 语义记忆

- 语义记忆是关于事实、知识、信息、数据等的记忆。
=============
以下是语义记忆：
{semantic_memories}

## 情节记忆

- 情节记忆是关于事件、情节、故事等的记忆。
=============
以下是情节记忆：
{episode_memories}

## 当前观察到的信息

- 当前观察到的信息是关于当前环境、任务、目标、状态等的记忆。
=============
以下是当前观察到的信息：
{observation}

=============
请你开始执行任务。
"""


class BaseMemoryAgent(BaseAgent):
    """BaseMemoryAgent 是所有具备记忆能力的智能体基类。
    
    属性:
        embedding_llm (EmbeddingLLM):
            嵌入语言模型
        vector_memory (VectorMemory):
            向量记忆，包含事实、知识、信息、数据等
        trajectory_memory (TableMemory):
            轨迹记忆，包含历史信息
        extract_prompts (dict[str, str]):
            记忆提取提示词
        memory_classes (dict[str, type[Union[SemanticMemory, EpisodeMemory, ProceduralMemory]]]):
            记忆类，用于创建记忆对象
    """
    # 嵌入语言模型
    embedding_llm: EmbeddingLLM
    # 向量记忆
    vector_memory: VectorMemoryCollection
    # 轨迹记忆
    trajectory_memory: TableMemoryDB
    # Prompt 样板，用于格式化召回的信息
    prompt_template: dict[str, str]
    # 记忆提取提示词
    extract_prompts: dict[str, str]
    # 记忆类
    memory_classes: dict[str, type[Union[SemanticMemoryItem, EpisodeMemoryItem, ProceduralMemoryItem]]] = {
        "semantic_memory": SemanticMemoryItem,
        "episode_memory": EpisodeMemoryItem,
        "procedural_memory": ProceduralMemoryItem,
    }
    
    def __init__(
        self, 
        vector_memory: VectorMemoryCollection, 
        embedding_llm: EmbeddingLLM, 
        # trajectory_memory: TableMemoryDB, # TODO: 暂时不使用轨迹记忆
        memory_prompts: dict[str, str], 
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vector_memory = vector_memory
        self.embedding_llm = embedding_llm
        # self.trajectory_memory = trajectory_memory # TODO: 暂时不使用轨迹记忆
        
        self.prompt_template = {
            "semantic_memory": memory_prompts["semantic_prompt_template"],
            "episode_memory": memory_prompts["episode_prompt_template"],
            "procedural_memory": memory_prompts["procedural_prompt_template"],
        }
        self.extract_prompts = {
            "semantic_memory": memory_prompts["semantic_extract_prompt"],
            "episode_memory": memory_prompts["episode_extract_prompt"],
            "procedural_memory": memory_prompts["procedural_extract_prompt"],
        }
        
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
        text: str, 
        target: Stateful, 
        memory_type: MemoryType, 
        **kwargs,
    ) -> list[BaseMemoryOperation]:
        """
        从文本中抽取记忆。
        
        参数:
            text (str):
                需要提取记忆的文本
            target (Stateful):
                任务目标，应该是 Stateful 的子类
            memory_type (MemoryType):
                需要提取的记忆类型
            **kwargs:
                其他参数
        返回:
            list[MemoryOperation]:
                从文本中抽取的记忆列表
        """
        # 根据 memory_type 获取记忆类
        MEMORY = self.memory_classes[memory_type.value]
        
        # 检测相似的记忆，避免冲突
        sim_memories = await self.search_memory(
            text=text,
            limit=20,
            score_threshold=0.5,
            memory_type=memory_type,
            target=target,
        )
        sim_memories = "\n".join([memory.content for memory in sim_memories])
        
        messages = []
        # 构建 SystemMessage
        system_message = SystemMessage(
            content=self.extract_prompts[memory_type.value].format(sim_memories=sim_memories)
        )
        # 将 SystemMessage 添加到 messages 中
        messages.append(system_message)
        # 构建 UserMessage
        user_message = UserMessage(content=text)
        # 将 UserMessage 添加到 messages 中
        messages.append(user_message)
        
        # 创建 CompletionConfig
        completion_config = BaseCompletionConfig(format_json=True)
        # 调用 LLM 提取语义记忆
        response = await self.llm.completion(messages, completion_config=completion_config)
        # 格式化 response
        response: list[dict] = json.loads(response.content)
        # 将 response 转换为 SemanticMemory、EpisodeMemory、ProceduralMemory 列表
        memory_ops = []
        for memory_op in response:
            # 获取 memory 的 operation
            operation = memory_op.pop("operation")
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
            embedding = await self.embedding_llm.embed(memory["content"], dimensions=self.vector_memory.get_dimension())
            # 更新 memory 的 embedding
            memory["embedding"] = embedding
            
            # 根据 memory_type 创建对应的记忆对象
            memory_ops.append(BaseMemoryOperation(
                operation=operation,
                memory=MEMORY(**memory),
            ))
            
        return memory_ops
    
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
        memory_type: MemoryType, 
        target: Stateful, 
        **kwargs,
    ) -> list[Union[SemanticMemoryItem, EpisodeMemoryItem, ProceduralMemoryItem]]:
        """从记忆中搜索信息。"""
        # 根据 memory_type 获取记忆类
        MEMORY = self.memory_classes[memory_type.value]
        
        # 把 text 转换为向量
        embedding = await self.embedding_llm.embed(text, dimensions=self.vector_memory.get_dimension())
        # 从向量记忆中搜索
        memories = await self.vector_memory.search(
            query_embedding=embedding, 
            top_k=limit, 
            score_threshold=score_threshold, 
            env_id=self.env.uid, 
            agent_id=self.uid, 
            task_id=target.uid, 
            task_status=target.status.value, 
            memory_type=memory_type.value, 
        )
        
        # 将 dict 转为 MemoryItem 列表
        memories = [MEMORY(**memory[0]) for memory in memories]
        return memories
    
    async def prompt(
        self, 
        prompt: Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult], 
        target: Stateful, 
        **kwargs,
    ) -> None:
        """为智能体发送提示，同时提取提示中的语义、情节和程序性记忆。
        
        参数:
            prompt (Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]):
                提示信息
            target (Stateful):
                提示的目标
            **kwargs:
                提示的额外关键字参数
                
        返回:
            None
        """
        await super().prompt(prompt, target, **kwargs)
        
        # 从 prompt 中抽取 semantic 记忆
        memories = await self.extract_memory(prompt.content, target, MemoryType.SEMANTIC)
        # 更新向量记忆
        await self.update_memory(memories)
        # 从 prompt 中抽取 episode 记忆
        memories = await self.extract_memory(prompt.content, target, MemoryType.EPISODE)
        # 更新向量记忆
        await self.update_memory(memories)
        # 从 prompt 中抽取 procedural 记忆
        memories = await self.extract_memory(prompt.content, target, MemoryType.PROCEDURAL)
        # 更新向量记忆
        await self.update_memory(memories)
    
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
        observation = await target.observe(observe_format, **kwargs)
        
        # 提取程序性记忆
        text = f"根据我所观察到的信息，我现在该做什么？该注意什么？#观察信息: {observation}"
        procedural_memories = await self.search_memory(
            text=text, 
            limit=20, 
            score_threshold=0.5, 
            memory_type=MemoryType.PROCEDURAL, 
            target=target, 
        )
        procedural_memories = self.prompt_template["procedural_memory"].format(
            "\n".join([memory.format() for memory in procedural_memories])
        )
        
        # 提取语义记忆
        text = f"根据我所观察到的信息，我需要什么信息？#观察信息: {observation}"
        semantic_memories = await self.search_memory(
            text=text, 
            limit=20, 
            score_threshold=0.5, 
            memory_type=MemoryType.SEMANTIC, 
            target=target, 
        )
        semantic_memories = self.prompt_template["semantic_memory"].format(
            "\n".join([memory.format() for memory in semantic_memories])
        )
        
        # 提取情节记忆
        text = f"根据我所观察到的信息，我历史的行动中有什么值得注意的？#观察信息: {observation}"
        episode_memories = await self.search_memory(
            text=text, 
            limit=20, 
            score_threshold=0.5, 
            memory_type=MemoryType.EPISODE, 
            target=target, 
        )
        episode_memories = self.prompt_template["episode_memory"].format(
            "\n".join([memory.format() for memory in episode_memories])
        )
        
        history: list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]] = []
        # 格式化 memories 并构成 SystemMessage
        system_message = SystemMessage(content=PROMPT_TEMPLATE.format(
            procedural_memories=procedural_memories,
            semantic_memories=semantic_memories,
            episode_memories=episode_memories,
            observation=observation,
        ))
        # 将 system_message 添加到 history 中
        history.append(system_message)
        # 将 observation 构建 UserMessage
        user_message = UserMessage(content=f"#观察信息: {observation}")
        # 将 user_message 添加到 history 中
        history.append(user_message)
        
        return history
