import json
from typing import Union

from myagents.core.interface import Stateful,  VectorMemory, TableMemory, EmbeddingLLM
from myagents.core.messages import AssistantMessage, ToolCallResult, SystemMessage, UserMessage
from myagents.core.llms import BaseCompletionConfig
from myagents.core.agents.base import BaseAgent
from myagents.core.memories.schemas import SemanticMemory, EpisodeMemory, ProceduralMemory, MemoryType


class BaseMemoryAgent(BaseAgent):
    """BaseMemoryAgent 是所有具备记忆能力的智能体基类。
    
    属性:
        embedding_llm (EmbeddingLLM):
            嵌入语言模型
        vector_memory (VectorMemory):
            向量记忆，包含事实、知识、信息、数据等
        trajectory_memory (TableMemory):
            轨迹记忆，包含历史信息
        prompt_template (dict[str, str]):
            用户提示词模板，用于格式化向量记忆
        extract_prompts (dict[str, str]):
            记忆提取提示词
        memory_classes (dict[str, type[Union[SemanticMemory, EpisodeMemory, ProceduralMemory]]]):
            记忆类，用于创建记忆对象
    """
    # 嵌入语言模型
    embedding_llm: EmbeddingLLM
    # 向量记忆
    vector_memory: VectorMemory
    # 轨迹记忆
    trajectory_memory: TableMemory
    # Prompt 样板，用于格式化召回的信息
    prompt_template: dict[str, str]
    # 记忆提取提示词
    extract_prompts: dict[str, str]
    # 记忆类
    memory_classes: dict[str, type[Union[SemanticMemory, EpisodeMemory, ProceduralMemory]]] = {
        "semantic_memory": SemanticMemory,
        "episode_memory": EpisodeMemory,
        "procedural_memory": ProceduralMemory,
    }
    
    def __init__(
        self, 
        vector_memory: VectorMemory, 
        trajectory_memory: TableMemory, 
        prompts: dict[str, str], 
        **kwargs,
    ) -> None:
        super().__init__(prompts=prompts, **kwargs)
        self.vector_memory = vector_memory
        self.trajectory_memory = trajectory_memory
        
        self.prompt_template = {
            "semantic_memory": prompts["semantic_prompt_template"],
            "episode_memory": prompts["episode_prompt_template"],
            "procedural_memory": prompts["procedural_prompt_template"],
        }
        self.extract_prompts = {
            "semantic_memory": prompts["semantic_extract_prompts"],
            "episode_memory": prompts["episode_extract_prompts"],
            "procedural_memory": prompts["procedural_extract_prompts"],
        }
    
    async def extract_memory(
        self, 
        text: str, 
        target: Stateful, 
        memory_type: MemoryType, 
        **kwargs,
    ) -> list[Union[SemanticMemory, EpisodeMemory, ProceduralMemory]]:
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
            list[Union[SemanticMemory, EpisodeMemory, ProceduralMemory]]:
                从文本中抽取的记忆列表
        """
        # 根据 memory_type 获取记忆类
        MEMORY = self.memory_classes[memory_type.value]
        
        # 检测相似的记忆，避免冲突
        memories = await self.search_memory(
            text=text, 
            limit=20, 
            score_threshold=0.5, 
            memory_type=memory_type,
            target=target,
        )
        
        messages = []
        # 构建 SystemMessage
        system_message = SystemMessage(content=self.extract_prompts[memory_type.value])
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
        memories = []
        for memory in response:
            # 获取当前的 env_id、agent_id、task_id
            env_id = self.env.uid
            agent_id = self.uid
            task_id = target.uid
            # 更新 memory 的 env_id、agent_id、task_id
            memory["env_id"] = env_id
            memory["agent_id"] = agent_id
            memory["task_id"] = task_id
            
            # 获取嵌入向量
            embedding = await self.embedding_llm.embed(memory["content"])
            # 更新 memory 的 embedding
            memory["embedding"] = embedding
            
            # 根据 memory_type 创建对应的记忆对象
            memories.append(MEMORY(**memory))
            
        return memories
    
    async def update_memory(
        self, 
        memories: list[Union[SemanticMemory, EpisodeMemory, ProceduralMemory]], 
        **kwargs,
    ) -> None:
        """更新记忆。"""
        await self.vector_memory.update(memories)
        
    async def search_memory(
        self, 
        text: str, 
        limit: int, 
        score_threshold: float, 
        memory_type: MemoryType, 
        target: Stateful, 
        **kwargs,
    ) -> list[Union[SemanticMemory, EpisodeMemory, ProceduralMemory]]:
        """从记忆中搜索信息。"""
        # 把 text 转换为向量
        embedding = await self.embedding_llm.embed(text)
        expr = f"memory_type == {memory_type}"
        # 从向量记忆中搜索
        memories = await self.vector_memory.search(
            query_embedding=embedding, 
            top_k=limit, 
            score_threshold=score_threshold, 
            condition=expr, 
            env_id=self.env.uid, 
            agent_id=self.uid, 
            task_id=target.uid, 
        )
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
        super().prompt(prompt, target, **kwargs)
        
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
        history: list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]] = []
        
        # 观察
        observation = await target.observe(observe_format, **kwargs)
        
        # 提取程序性记忆
        text = f"根据我所观察到的信息，我现在该做什么？该注意什么？#观察信息: {observation}"
        memories = await self.search_memory(
            text=text, 
            limit=20, 
            score_threshold=0.5, 
            memory_type=MemoryType.PROCEDURAL, 
            target=target, 
        )
        # 根据程序性记忆，构建 SystemMessage
        system_message = SystemMessage(content=self.prompt_template.format(memories))
        # 将 SystemMessage 添加到 history 中
        history.append(system_message)
        
        # 提取语义记忆
        text = f"根据我所观察到的信息，我需要什么信息？#观察信息: {observation}"
        memories = await self.search_memory(
            text=text, 
            limit=20, 
            score_threshold=0.5, 
            memory_type=MemoryType.SEMANTIC, 
            target=target, 
        )
        # 根据语义记忆，构建 UserMessage
        user_message = UserMessage(content=self.prompt_template.format(memories))
        # 将 UserMessage 添加到 history 中
        history.append(user_message)
        
        # 提取情节记忆
        text = f"根据我所观察到的信息，我历史的行动中有什么值得注意的？#观察信息: {observation}"
        memories = await self.search_memory(
            text=text, 
            limit=20, 
            score_threshold=0.5, 
            memory_type=MemoryType.EPISODE, 
            target=target, 
        )
        # 根据情节记忆，构建 UserMessage
        user_message = UserMessage(content=self.prompt_template.format(memories))
        # 将 UserMessage 添加到 history 中
        history.append(user_message)
        
        # 将 observation 添加到 history 中
        observe_message = UserMessage(content=observation)
        history.append(observe_message)
        
        return history
