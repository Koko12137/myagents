from abc import ABC, abstractmethod
from asyncio import Semaphore, Lock
from enum import Enum
from typing import Any, Union

from fastmcp.tools import Tool as FastMcpTool
from fastmcp import Client as MCPClient

from myagents.core.interface.base import Stateful, ToolsCaller, Scheduler, CallStack
from myagents.core.interface.llm import LLM, CompletionConfig, EmbeddingLLM
from myagents.core.interface.memory import MemoryOperation, VectorMemoryCollection, VectorMemoryItem
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, ToolCallRequest


class StepCounter(ABC):
    """步骤计数器的协议。限制可以是最大自动步骤或最大余额成本。最好为所有Agent使用相同的步骤计数器。
    
    属性:
        uid (str):
            步骤计数器的唯一名称
        limit (Union[int, float]):
            步骤计数器的限制
        current (Union[int, float]):
            步骤计数器的当前步骤
        lock (Lock):
            步骤计数器的锁
    """
    uid: str
    limit: Union[int, float]
    current: Union[int, float]
    lock: Lock
    
    @abstractmethod
    async def reset(self) -> None:
        """重置步骤计数器的当前步骤
        
        返回:
            None
        """
        pass
    
    @abstractmethod
    async def update_limit(self, limit: Union[int, float]) -> None:
        """更新步骤计数器的限制
        
        参数:
            limit (Union[int, float]):
                步骤计数器的限制
        """
        pass
    
    @abstractmethod
    async def check_limit(self) -> bool:
        """检查步骤计数器的限制是否已达到
        
        返回:
            bool:
                步骤计数器的限制是否已达到
        
        异常:
            MaxStepsError:
                步骤计数器引发的最大步骤错误
        """
        pass
    
    @abstractmethod
    async def step(self, step: Union[int, float]) -> None:
        """增加步骤计数器的当前步骤
        
        参数:
            step (Union[int, float]):
                要增加的步骤
        
        返回:
            None 
        
        异常:
            MaxStepsError:
                步骤计数器引发的最大步骤错误
        """
        pass
    
    @abstractmethod
    async def recharge(self, limit: Union[int, float]) -> None:
        """为步骤计数器充值限制
        
        参数:
            limit (Union[int, float]):
                步骤计数器的限制
        """
        pass


class Agent(ABC):
    """在环境中运行的Agent，根据工作流处理任务
    
    属性:
        uid (str):
            Agent的唯一标识符
        name (str):
            Agent的名称
        agent_type (Enum):
            Agent的类型
        profile (str):
            Agent的描述文件。描述Agent的行为和目标
        llms (dict[str, LLM]):
            Agent使用的语言模型
        mcp_client (MCPClient):
            Agent使用的MCP客户端
        tools (dict[str, FastMcpTool]):
            Agent使用的工具
        workflow (Workflow):
            Agent运行的工作流
        env (Environment):
            Agent运行的环境
        step_counters (dict[str, StepCounter]):
            Agent使用的步骤计数器。任何一个达到限制，Agent就会停止
        lock (Lock):
            Agent的同步锁。Agent一次只能处理一个任务
        prompts (dict[str, str]):
            运行工作流的提示
        observe_format (dict[str, str]):
            目标观察的格式
    """
    # 基本信息
    uid: str
    name: str
    agent_type: Enum
    profile: str
    # 语言模型和MCP客户端
    llms: dict[str, LLM]
    mcp_client: MCPClient
    tools: dict[str, FastMcpTool]
    # 工作流和环境以及运行上下文
    workflow: 'Workflow'
    env: 'Environment'
    # Agent的步骤计数器
    step_counters: dict[str, StepCounter]
    # 同步锁
    lock: Lock
    # 提示和观察格式
    prompts: dict[str, str]
    observe_format: dict[str, str]
    
    @abstractmethod
    async def prompt(
        self, 
        prompt: Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult], 
        target: Stateful, 
        **kwargs
    ) -> None:
        """环境向智能体发送提示
        
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
        pass
    
    @abstractmethod
    async def observe(
        self, 
        target: Stateful, 
        observe_format: str, 
        **kwargs, 
    ) -> list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
        """观察目标
        
        参数:
            target (Stateful):
                要观察的有状态实体
            observe_format (str):
                观察的格式。这必须是目标的有效观察格式
            **kwargs:
                观察目标的额外关键字参数

        返回:
            list[Union[SystemMessage, UserMessage, AssistantMessage, ToolCallResult]]:
                从目标观察到的最新信息
        """
        pass 
    
    @abstractmethod
    async def think(
        self, 
        llm_name: str, 
        observe: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]], 
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> AssistantMessage:
        """思考任务或环境的观察结果
        
        参数:
            llm_name (str):
                语言模型的名称，用于按照阶段名称选择对应的LLM
            observe (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
                从任务或环境观察到的消息
            completion_config (CompletionConfig):
                Agent的完成配置
            **kwargs:
                思考任务或环境的额外关键字参数
                
        返回:
            AssistantMessage:
                语言模型思考的完成消息
        """
    
    @abstractmethod
    async def act(self, tool_call: ToolCallRequest, **kwargs) -> ToolCallResult:
        """根据工具调用采取行动。其他参数可以通过关键字参数提供给工具调用
        
        参数:
            tool_call (ToolCallRequest):
                工具调用请求，包括工具调用ID和工具调用参数
            **kwargs:
                调用工具的额外关键字参数
                
        返回:
            ToolCallResult:
                Agent在环境或任务上行动后返回的工具调用结果
            
        异常:
            ValueError:
                如果工具调用名称未注册到工作流或环境
        """
    
    @abstractmethod
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig, 
        **kwargs
    ) -> AssistantMessage:
        """在任务或环境上运行Agent。在运行Agent之前，应该获取Agent的锁
        
        参数:
            target (Stateful):
                运行Agent的有状态实体
            max_error_retry (int):
                目标出错时重试Agent的最大次数
            max_idle_thinking (int):
                Agent空闲思考的最大次数
            completion_config (CompletionConfig):
                Agent的完成配置
            **kwargs:
                运行Agent的额外关键字参数
                
        返回:
            AssistantMessage:
                Agent在有状态实体上运行后返回的助手消息
        """
    
    @abstractmethod
    def register_counter(self, counter: StepCounter) -> None:
        """向Agent注册步骤计数器
        
        参数:
            counter (StepCounter):
                要注册的步骤计数器
        """
    
    @abstractmethod
    def register_workflow(self, workflow: 'Workflow') -> None:
        """向Agent注册工作流
        
        参数:
            workflow (Workflow):
                要注册的工作流
        """
    
    @abstractmethod
    def register_env(self, env: 'Environment') -> None:
        """向Agent注册环境
        
        参数:
            env (Environment):
                要注册的环境
        """


class MemoryAgent(Agent):
    """MemoryAgent 是一个可以使用记忆来思考和行动的Agent
    """
    
    @abstractmethod
    def get_embedding_llm(self) -> EmbeddingLLM:
        """获取智能体的嵌入模型
        
        返回:
            LLM:
                智能体的嵌入模型
        """
        pass
    
    @abstractmethod
    def get_extraction_llm(self) -> LLM:
        """获取智能体的记忆提取模型
        
        返回:
            LLM:
                智能体的记忆提取模型
        """
        pass
    
    @abstractmethod
    def get_memory_workflow(self) -> 'MemoryWorkflow':
        """获取记忆工作流
        
        返回:
            MemoryWorkflow:
                记忆工作流
        """
        pass
    
    @abstractmethod
    def get_memory_collection(self, memory_type: str) -> VectorMemoryCollection:
        """获取向量记忆集合
        
        参数:
            memory_type (str):
                记忆类型
        
        返回:
            VectorMemoryCollection:
                向量记忆集合
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def extract_memory(
        self, 
        target: Stateful, 
        **kwargs,
    ) -> str:
        """从有状态实体中提取记忆，并更新状态和记忆
        
        参数:
            target (Stateful):
                有状态实体
            **kwargs:
                额外参数
                
        返回:
            str:
                提取的记忆
        """
        pass
    
    @abstractmethod
    async def search_memory(
        self, 
        text: str, 
        limit: int, 
        **kwargs, 
    ) -> str:
        """从记忆中搜索信息
        
        参数:
            text (str):
                文本
            limit (int):
                限制
            **kwargs:
                额外参数
        
        返回:
            str:
                从记忆中搜索的信息
        """
        pass
    
    @abstractmethod
    def create_memory(self, memory_type: str, **kwargs) -> VectorMemoryItem:
        """创建记忆
        
        参数:
            memory_type (str):
                记忆类型
            **kwargs:
                额外参数
                
        返回:
            VectorMemoryItem:
                向量记忆
        """
        pass
    
    @abstractmethod
    async def update_memory(
        self, 
        memories: list[MemoryOperation], 
        **kwargs, 
    ) -> None:
        """更新记忆
        
        参数:
            memories (list[MemoryOperation]):
                要更新的记忆
            **kwargs:
                额外参数
        """
        pass
    
    @abstractmethod
    async def update_temp_memory(
        self, 
        temp_memory: str, 
        target: Stateful, 
        **kwargs, 
    ) -> None:
        """更新临时记忆
        
        参数:
            temp_memory (str):
                临时记忆
            target (Stateful):
                目标
            **kwargs:
                额外参数
        """


class Workflow(ToolsCaller, Scheduler):
    """工作流是无状态的，它不存储任何关于状态的信息，仅用于编排任务或环境。
    工作流不负责任务或环境的状态。
    
    属性:
        profile (str):
            工作流的描述文件。描述工作流的行为和目标
        agent (Agent):
            与工作流一起工作的Agent
        prompts (dict[str, str]):
            工作流的提示。键是提示名称，值是提示内容
        observe_formats (dict[str, str]):
            观察的格式。键是观察名称，值是格式方法名称
        sub_workflows (dict[str, 'Workflow']):
            工作流的子工作流。键是子工作流的名称，值是子工作流实例
    """
    # 基本信息
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_formats: dict[str, str]
    # 子工作流
    sub_workflows: dict[str, 'Workflow']
    
    @abstractmethod
    def register_agent(self, agent: Agent) -> None:
        """向工作流注册Agent
        
        参数:
            agent (Agent):
                要注册的Agent
        """
    
    @abstractmethod
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> Stateful:
        """运行工作流以修改有状态实体

        参数:
            target (Stateful): 
                运行工作流的有状态实体
            max_error_retry (int):
                目标出错时重试工作流的最大次数
            max_idle_thinking (int):
                工作流空闲思考的最大次数
            completion_config (CompletionConfig):
                工作流的完成配置
            **kwargs:
                运行工作流的额外关键字参数

        返回:
            Stateful: 
                运行工作流后的有状态实体
        """


class ReActFlow(Workflow):
    """ReActFlow 是一个基于推理和行动的工作流
    """
    
    @abstractmethod
    async def reason_act(
        self, 
        target: Stateful, 
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> tuple[Stateful, bool, bool]:
        """推理和行动步骤
        
        参数:
            target (Stateful):
                目标有状态实体
            completion_config (CompletionConfig):
                完成配置
            **kwargs:
                额外参数
                
        返回:
            tuple[Stateful, bool, bool]:
                修改后的目标、是否完成、是否继续
        """
    
    @abstractmethod
    async def reflect(
        self, 
        target: Stateful, 
        completion_config: CompletionConfig, 
        **kwargs, 
    ) -> tuple[Stateful, bool]:
        """反思步骤
        
        参数:
            target (Stateful):
                目标有状态实体
            completion_config (CompletionConfig):
                完成配置
            **kwargs:
                额外参数
                
        返回:
            tuple[Stateful, bool]:
                修改后的目标、是否完成
        """
    

class MemoryWorkflow(Workflow):
    """MemoryWorkflow 是一个基于记忆的工作流
    """
    
    @abstractmethod
    def get_memory_agent(self) -> MemoryAgent:
        """获取记忆Agent
        
        返回:
            MemoryAgent:
                记忆Agent
        """
        pass

    @abstractmethod
    async def extract_memory(
        self, 
        text: str, 
        **kwargs, 
    ) -> str:
        """从文本中提取记忆
        
        参数:
            text (str):
                文本
            **kwargs:
                额外参数
                
        返回:
            str:
                提取的记忆
        """


class EnvironmentStatus(Enum):
    """环境的状态
    
    - CREATED (int): 环境已创建
    - PLANNING (int): 环境正在规划
    - RUNNING (int): 环境正在运行
    - FINISHED (int): 环境已完成
    - ERROR (int): 环境出错
    - CANCELLED (int): 环境已取消
    """
    CREATED = 0
    PLANNING = 1
    RUNNING = 2
    FINISHED = 3
    ERROR = 4
    CANCELLED = 5


class Environment(Stateful, ToolsCaller, Scheduler):
    """环境是一个包含工作流的有状态对象。工作流可用于思考如何修改环境。
    工具可用于修改环境。
    
    属性:
        uid (str):
            环境的唯一标识符
        name (str):
            环境的名称
        profile (str):
            环境的描述文件。描述环境的行为和目标
        prompts (dict[str, str]):
            环境的提示。键是提示名称，值是提示内容
        required_agents (list[Enum]):
            列表中的Agent必须注册到环境
        agents (dict[str, Agent]):
            环境中的Agent。键是Agent名称，值是Agent
        agent_type_map (dict[Enum, list[str]]):
            Agent类型到Agent名称的映射。键是Agent类型，值是Agent名称列表
        agent_type_semaphore (dict[Enum, Semaphore]):
            Agent类型的信号量。键是Agent类型，值是信号量
    """
    uid: str
    name: str
    profile: str
    prompts: dict[str, str]
    required_agents: list[Enum]
    # Agent和并发控制
    agents: dict[str, Agent]
    agent_type_map: dict[Enum, list[str]]
    agent_type_semaphore: dict[Enum, Semaphore]
    
    @abstractmethod
    def register_agent(self, agent: Agent) -> None:
        """向环境注册Agent
        
        参数:
            agent (Agent):
                要注册的Agent
        """
    
    @abstractmethod
    async def call_agent(
        self, 
        agent_type: Enum, 
        target: Stateful, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig, 
        designated_agent: str, 
        **kwargs, 
    ) -> AssistantMessage:
        """调用Agent
        
        参数:
            agent_type (Enum):
                Agent类型
            target (Stateful):
                目标有状态实体
            max_error_retry (int):
                最大错误重试次数
            max_idle_thinking (int):
                最大空闲思考次数
            completion_config (CompletionConfig):
                完成配置
            designated_agent (str):
                指定的Agent名称
            **kwargs:
                额外参数
                
        返回:
            AssistantMessage:
                Agent返回的助手消息
        """
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """运行环境
        
        参数:
            *args:
                位置参数
            **kwargs:
                关键字参数
                
        返回:
            Any:
                运行结果
        """
