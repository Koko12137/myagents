import sys
from typing import Union, Any

from fastmcp.client import Client as MCPClient
from pydantic import BaseModel, Field

from myagents.core.configs.agents import CounterConfig, AgentConfig
from myagents.core.configs.llms import LLMConfig
from myagents.core.configs.mcps import MCPConfig
from myagents.core.configs.envs import EnvironmentConfig
from myagents.core.envs.orchestrate import Orchestrate
from myagents.core.interface import LLM, StepCounter, Agent, Workflow, Environment
from myagents.core.llms import OpenAiLLM
from myagents.core.agents import AgentType, ReActAgent, TreeReActAgent, OrchestrateAgent, PlanAndExecAgent
from myagents.core.envs import ComplexQuery, EnvironmentType
from myagents.core.utils.step_counters import BaseStepCounter, MaxStepCounter, TokenStepCounter
from myagents.core.utils.logger import init_logger
from myagents.core.utils.name_generator import generate_name


class AutoAgentConfig(BaseModel):
    """AutoAgent 的配置类
    
    属性:
        environment (EnvironmentConfig):
            环境的配置
        debug (bool, 可选, 默认为 False):
            是否启用自动代理的调试模式
        log_level (str, 可选, 默认为 "INFO"):
            自动代理的日志级别
        save_dir (str, 可选, 默认为 "stdout"):
            保存自动代理日志的目录
    """
    environment: EnvironmentConfig = Field(description="环境的配置")
    # 调试和日志设置
    debug: bool = Field(default=False, description="是否启用自动代理的调试模式")
    log_level: str = Field(default="INFO", description="自动代理的日志级别")
    save_dir: str = Field(default="stdout", description="保存自动代理日志的目录")
    
    
class AutoAgent:
    """AutoAgent 是一个用于创建代理并将其分配到环境和工作流的工厂类
    """
    
    def __init__(self):
        """初始化 AutoAgent 工厂"""
        self._existing_names = []  # 跟踪已创建的agent名字
    
    def build_counter(self, config: CounterConfig) -> StepCounter:
        """构建步骤计数器
        
        参数:
            config (CounterConfig):
                步骤计数器的配置
                
        返回:
            StepCounter:
                步骤计数器
        """
        name = config.name
        
        match name:
            case "base":
                return BaseStepCounter(limit=config.limit)
            case "max":
                return MaxStepCounter(limit=config.limit)
            case "token":
                return TokenStepCounter(limit=config.limit)
            case _:
                raise ValueError(f"无效的步骤计数器名称: {name}")
    
    def __build_llm(self, config: LLMConfig) -> LLM:
        """构建语言模型
        
        参数:
            config (LLMConfig):
                语言模型的配置
                
        返回:
            LLM:
                语言模型
        """
        provider = config.provider
        
        # 为语言模型创建参数字典
        kwargs: dict[str, Any] = {}
        if config.extra_body != {}:
            kwargs["extra_body"] = config.extra_body
        if config.api_key_field is not None:
            kwargs["api_key_field"] = config.api_key_field
        
        match provider:
            case "openai":
                return OpenAiLLM(
                    base_url=config.base_url,
                    model=config.model,
                    temperature=config.temperature, 
                    **kwargs, 
                )
            case _:
                raise ValueError(f"无效的语言模型名称: {provider}")
            
    def __build_mcp_client(self, config: MCPConfig) -> MCPClient:
        """构建 MCP 客户端
        
        参数:
            config (MCPConfig):
                MCP 客户端的配置
                
        返回:
            MCPClient:
                MCP 客户端
        """
        if config is None:
            return None
        
        # 创建 MCP 客户端
        mcp_client = MCPClient(
            server_name=config.server_name, 
            server_url=config.server_url, 
            server_port=config.server_port, 
            auth_token=config.auth_token, 
        )
        
        # 返回 MCP 客户端
        return mcp_client
        
        
    def __build_agent(self, config: AgentConfig, step_counters: list[StepCounter]) -> Agent:
        """构建代理
        
        参数:
            config (AgentConfig):
                代理的配置
            step_counters (list[StepCounter]):
                代理的步骤计数器。任何一个达到限制，代理就会停止
        
        返回:
            Agent:
                代理
        """
        agent_type = AgentType(config.type)
        
        # 构建语言模型
        llm = self.__build_llm(config.llm)
        # 构建 MCP 客户端
        mcp_client = self.__build_mcp_client(config.mcp_client)
        
        # 生成唯一的代理名字
        agent_name = generate_name(excluded_names=self._existing_names)
        # 添加到已存在名字列表
        self._existing_names.append(agent_name)
        
        match agent_type:
            case AgentType.REACT:
                AGENT = ReActAgent
            case AgentType.TREE_REACT:
                AGENT = TreeReActAgent
            case AgentType.ORCHESTRATE:
                AGENT = OrchestrateAgent
            case AgentType.PLAN_AND_EXECUTE:
                AGENT = PlanAndExecAgent
            case _:
                raise ValueError(f"无效的代理类型: {agent_type}")
        
        # 构建代理
        return AGENT(
            name=agent_name,
            llm=llm, 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
        )
            
    def __build_environment(self, config: EnvironmentConfig) -> Environment:
        """构建环境
        
        参数:
            config (EnvironmentConfig):
                环境的配置
        
        返回:
            Environment:
                环境
        """
        # 构建全局步骤计数器
        counters = [self.build_counter(counter) for counter in config.step_counters]
        # 构建代理
        agents = [self.__build_agent(agent, counters) for agent in config.agents]
        
        env_type = EnvironmentType(config.type)
        # 构建环境
        match env_type:
            case EnvironmentType.COMPLEX_QUERY:
                env = ComplexQuery
            case EnvironmentType.ORCHESTRATE:
                env = Orchestrate
            case _:
                raise ValueError(f"无效的环境类型: {env_type}")
        
        # 构建环境
        env = env()
        # 将代理注册到环境
        for agent in agents:
            # 将代理注册到环境
            env.register_agent(agent)
            # 将环境注册到代理
            agent.register_env(env)
        # 返回环境
        return env
            
    def auto_build(self, config: AutoAgentConfig) -> Union[Environment, Workflow]:
        """构建自动代理
        
        参数:
            config (AutoAgentConfig):
                自动代理的配置
                
        返回:
            Union[Environment, Workflow]:
                环境或工作流
                
        异常:
            ValueError:
                如果同时提供了环境和工作流，或者没有提供环境和工作流
        """
        # 设置调试和日志级别
        log_level = "DEBUG" if config.debug else "INFO"
        # 创建日志记录器
        custom_logger = init_logger(
            sink=config.save_dir, 
            level=log_level, 
            colorize=True, 
        )
        # 如果输出不是标准输出，添加额外的标准输出日志记录器
        if config.save_dir != "stdout":
            custom_logger.add(sys.stdout, level=log_level, colorize=True)
        
        # 构建环境
        environment = self.__build_environment(config.environment)
        # 返回环境
        return environment
