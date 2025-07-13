import sys
from typing import Union, Optional, Any

from fastmcp.client import Client as MCPClient
from pydantic import BaseModel, Field

from myagents.core.configs.agents import CounterConfig, AgentConfig
from myagents.core.configs.llms import LLMConfig
from myagents.core.configs.mcps import MCPConfig
from myagents.core.configs.envs import EnvironmentConfig
from myagents.core.interface import LLM, StepCounter, Agent, Workflow, Environment
from myagents.core.llms import OpenAiLLM
from myagents.core.agents import AgentType, ReActAgent, OrchestrateAgent, PlanAndExecAgent
from myagents.core.envs import Query, EnvironmentType
from myagents.core.utils.step_counters import BaseStepCounter, MaxStepCounter, TokenStepCounter
from myagents.core.utils.logger import init_logger
from myagents.core.utils.name_generator import generate_name


class AutoAgentConfig(BaseModel):
    """AutoAgentConfig is the configuration for the AutoAgent.
    
    Attributes:
        environment (EnvironmentConfig):
            The configuration for the environment.
        debug (bool, optional, defaults to False):
            Whether to enable the debug mode for the auto agent.
        log_level (str, optional, defaults to "INFO"):
            The log level for the auto agent.
        save_dir (str, optional, defaults to "stdout"):
            The directory to save the logs for the auto agent.
    """
    environment: EnvironmentConfig = Field(description="The configuration for the environment.")
    # Debug and logging settings
    debug: bool = Field(default=False, description="Whether to enable the debug mode for the auto agent.")
    log_level: str = Field(default="INFO", description="The log level for the auto agent.")
    save_dir: str = Field(default="stdout", description="The directory to save the logs for the auto agent.")
    
    
class AutoAgent:
    """AutoAgent is a factory for creating agents and allocating them to the environment and workflows.
    """
    
    def __init__(self):
        """Initialize the AutoAgent factory."""
        self._existing_names = []  # 跟踪已创建的agent名字
    
    def build_counter(self, config: CounterConfig) -> StepCounter:
        """Build a step counter.
        
        Args:
            config (CounterConfig):
                The configuration for the step counter.
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
                raise ValueError(f"Invalid step counter name: {name}")
    
    def __build_llm(self, config: LLMConfig) -> LLM:
        """Build a LLM.
        
        Args:
            config (LLMConfig):
                The configuration for the LLM.
                
        Returns:
            LLM:
                The LLM.
        """
        provider = config.provider
        
        # Create kwargs for the LLM
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
                raise ValueError(f"Invalid LLM name: {provider}")
            
    def __build_mcp_client(self, config: MCPConfig) -> MCPClient:
        """Build a MCP client.
        
        Args:
            config (MCPConfig):
                The configuration for the MCP client.
                
        Returns:
            MCPClient:
                The MCP client. 
        """
        if config is None:
            return None
        
        # Create the MCP client
        mcp_client = MCPClient(
            server_name=config.server_name, 
            server_url=config.server_url, 
            server_port=config.server_port, 
            auth_token=config.auth_token, 
        )
        
        # Return the MCP client
        return mcp_client
        
        
    def __build_agent(self, config: AgentConfig, step_counters: list[StepCounter]) -> Agent:
        """Build an agent.
        
        Args:
            config (AgentConfig):
                The configuration for the agent.
            step_counters (list[StepCounter]):
                The step counters for the agent. Any of one reach the limit, the agent will be stopped. 
        
        Returns:
            Agent:
                The agent.
        """
        agent_type = AgentType(config.type)
        
        # Build the LLM
        llm = self.__build_llm(config.llm)
        # Build the MCP client
        mcp_client = self.__build_mcp_client(config.mcp_client)
        
        # 生成唯一的agent名字
        agent_name = generate_name(excluded_names=self._existing_names)
        # 添加到已存在名字列表
        self._existing_names.append(agent_name)
        
        match agent_type:
            case AgentType.REACT:
                AGENT = ReActAgent
            case AgentType.ORCHESTRATE:
                AGENT = OrchestrateAgent
            case AgentType.PLAN_AND_EXECUTE:
                AGENT = PlanAndExecAgent
            case _:
                raise ValueError(f"Invalid agent type: {agent_type}")
        
        # Build the agent
        return AGENT(
            name=agent_name,
            llm=llm, 
            step_counters=step_counters, 
            mcp_client=mcp_client, 
        )
            
    def __build_environment(self, config: EnvironmentConfig) -> Environment:
        """Build an environment.
        
        Args:
            config (EnvironmentConfig):
                The configuration for the environment.
        
        Returns:
            Environment:
                The environment.
        """
        # Build Global Step Counter
        counters = [self.build_counter(counter) for counter in config.step_counters]
        # Build the agents
        agents = [self.__build_agent(agent, counters) for agent in config.agents]
        
        env_type = EnvironmentType(config.type)
        # Build the environment
        match env_type:
            case EnvironmentType.QUERY:
                env = Query()
                # Register the agents to the environment
                for agent in agents:
                    # Register the agent to the environment
                    env.register_agent(agent)
                    # Register the environment to the agent
                    agent.register_env(env)
                # Return the environment
                return env
            case _:
                raise ValueError(f"Invalid environment type: {env_type}")
            
    def auto_build(self, config: AutoAgentConfig) -> Union[Environment, Workflow]:
        """Build an auto agent.
        
        Args:
            config (AutoAgentConfig):
                The configuration for the auto agent.
                
        Returns:
            Union[Environment, Workflow]:
                The environment or workflow.
                
        Raises:
            ValueError:
                If both environment and workflow are provided. Or no environment or workflow is provided. 
        """
        # Set the debug and log level
        log_level = "DEBUG" if config.debug else "INFO"
        # Create a logger
        custom_logger = init_logger(
            sink=config.save_dir, 
            level=log_level, 
            colorize=True, 
        )
        # If sink is not stdout, add an additional stdout logger
        if config.save_dir != "stdout":
            custom_logger.add(sys.stdout, level=log_level, colorize=True)
        
        # Build the environment
        environment = self.__build_environment(config.environment)
        # Return the environment
        return environment
