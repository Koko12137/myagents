import json
from typing import Union

from myagents.core.agents.base import BaseAgent
from myagents.core.interface import Stateful, TableMemory, VectorMemory
from myagents.core.messages import AssistantMessage, ToolCallResult, SystemMessage, UserMessage
from myagents.core.llms import BaseCompletionConfig


class BaseMemoryAgent(BaseAgent):
    """BaseMemoryAgent 是所有具备记忆能力的智能体基类。
    
    属性:
        semantic_memory (VectorMemory):
            语义记忆(事实记忆)，包含事实、知识、信息、数据等
        episodic_memory (VectorMemory):
            情节记忆(过程记忆)，包含情节、事件、场景、状态等
        procedural_memory (VectorMemory):
            程序性记忆(指令记忆)，包含指令、步骤、操作、方法等
        trajectory_memory (VectorMemory):
            轨迹记忆(轨迹记忆)，包含完整的行动轨迹，包括行动、结果、反馈等
        system_prompt_template (str):
            系统提示词模板，用于格式化程序性记忆
        user_prompt_template (str):
            用户提示词模板，用于格式化语义记忆、情节记忆
        semantic_search_prompt (str):
            语义记忆检索提示词
        episodic_search_prompt (str):
            情节记忆检索提示词
        procedural_search_prompt (str):
            程序性记忆检索提示词
    """
    semantic_memory: VectorMemory
    episodic_memory: VectorMemory
    procedural_memory: VectorMemory
    trajectory_memory: TableMemory
    # Prompt 样板，用于格式化召回的信息
    system_prompt_template: str
    user_prompt_template: str
    # 记忆提取提示词
    semantic_extract_prompt: str
    episodic_extract_prompt: str
    procedural_extract_prompt: str
    # 记忆检索提示词
    semantic_search_prompt: str
    episodic_search_prompt: str
    procedural_search_prompt: str
    
    def __init__(
        self, 
        semantic_memory: VectorMemory, 
        episodic_memory: VectorMemory, 
        procedural_memory: VectorMemory, 
        trajectory_memory: TableMemory, 
        system_prompt_template: str, 
        user_prompt_template: str, 
        semantic_extract_prompt: str, 
        episodic_extract_prompt: str, 
        procedural_extract_prompt: str, 
        semantic_search_prompt: str, 
        episodic_search_prompt: str, 
        procedural_search_prompt: str, 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.procedural_memory = procedural_memory
        self.trajectory_memory = trajectory_memory
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.semantic_extract_prompt = semantic_extract_prompt
        self.episodic_extract_prompt = episodic_extract_prompt
        self.procedural_extract_prompt = procedural_extract_prompt
        self.semantic_search_prompt = semantic_search_prompt
        self.episodic_search_prompt = episodic_search_prompt
        self.procedural_search_prompt = procedural_search_prompt
    
    async def extract_semantic_memory(
        self, 
        text: str, 
        **kwargs,
    ) -> list[str]:
        """从文本中抽取语义（事实）记忆。"""
        messages = []
        # 构建 SystemMessage
        system_message = SystemMessage(content=self.semantic_extract_prompt)
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
        response = json.loads(response.content)
        # 嵌入 response 到语义记忆
        await self.semantic_memory.update(response)
    
    async def extract_episodic_memory(
        self, 
        text: str, 
        **kwargs,
    ) -> list[str]:
        """从文本中抽取情节（过程）记忆。"""
        messages = []
        # 构建 SystemMessage
        system_message = SystemMessage(content=self.episodic_extract_prompt)
        # 将 SystemMessage 添加到 messages 中
        messages.append(system_message)
        # 构建 UserMessage
        user_message = UserMessage(content=text)
        # 将 UserMessage 添加到 messages 中
        messages.append(user_message)
        
        # 创建 CompletionConfig
        completion_config = BaseCompletionConfig(format_json=True)
        # 调用 LLM 提取情节记忆
        response = await self.llm.completion(messages, completion_config=completion_config)
        # 格式化 response 
        response = json.loads(response.content)
        # 嵌入 response 到情节记忆
        await self.episodic_memory.update(response)
    
    async def extract_procedural_memory(
        self, 
        text: str, 
        **kwargs,
    ) -> list[str]:
        """从文本中抽取程序性（指令）记忆。"""
        messages = []
        # 构建 SystemMessage
        system_message = SystemMessage(content=self.procedural_extract_prompt)
        # 将 SystemMessage 添加到 messages 中
        messages.append(system_message)
        # 构建 UserMessage
        user_message = UserMessage(content=text)
        # 将 UserMessage 添加到 messages 中
        messages.append(user_message)
        
        # 创建 CompletionConfig
        completion_config = BaseCompletionConfig(format_json=True)
        # 调用 LLM 提取程序性记忆
        response = await self.llm.completion(messages, completion_config=completion_config)
        # 格式化 response 
        response = json.loads(response.content)
        # 嵌入 response 到程序性记忆
        await self.procedural_memory.update(response)
    
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
        
        if isinstance(prompt, SystemMessage) or isinstance(prompt, UserMessage):
            # 从 prompt 中抽取程序性记忆并更新
            procedural_memory = await self.extract_procedural_memory(prompt.content)
            # 更新程序性记忆
            self.procedural_memory.update(procedural_memory)
        elif isinstance(prompt, AssistantMessage):
            # 从 prompt 中抽取情节记忆并更新
            episodic_memory = await self.extract_episodic_memory(prompt.content)
            # 更新情节记忆
            self.episodic_memory.update(episodic_memory)
        elif isinstance(prompt, ToolCallResult):
            # 从 prompt 中抽取语义记忆并更新
            semantic_memory = await self.extract_semantic_memory(prompt.content)
            # 更新语义记忆
            self.semantic_memory.update(semantic_memory)
    
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
        
        # 从 procedural memory 中获取当前的程序性记忆
        procedural_memory = await self.procedural_memory.search(self.procedural_search_prompt)
        # 根据程序性记忆，构建 SystemMessage
        system_message = SystemMessage(content=self.system_prompt_template.format(procedural_memory))
        # 将 SystemMessage 添加到 history 中
        history.append(system_message)
        
        # 从 semantic memory 中获取当前的语义记忆
        semantic_memory = await self.semantic_memory.search(self.semantic_search_prompt)
        # 从 episodic memory 中获取当前的情节记忆
        episodic_memory = await self.episodic_memory.search(self.episodic_search_prompt)
        # 根据语义记忆和情节记忆，构建 UserMessage
        user_message = UserMessage(content=self.user_prompt_template.format(semantic_memory, episodic_memory))
        # 将 UserMessage 添加到 history 中
        history.append(user_message)
        
        # 观察
        observation = await target.observe(observe_format, **kwargs)
        # 根据观察结果，构建 UserMessage
        user_message = UserMessage(content=observation)
        # 将 UserMessage 添加到 history 中
        history.append(user_message)
        # 将 observation 嵌入到 trajectory memory 中
        await self.trajectory_memory.add(observation)
        
        return history
