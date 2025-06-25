import sys
from typing import Union

from loguru import logger
from fastmcp.client import Client as MCPClient
from pydantic import BaseModel, Field

from myagents.src.configs.agents import CounterConfig, AgentConfig
from myagents.src.configs.llms import LLMConfig
from myagents.src.configs.mcps import MCPConfig
from myagents.src.configs.workflows import WorkflowConfig
from myagents.src.configs.envs import EnvironmentConfig
from myagents.src.interface import LLM, StepCounter, Agent, Workflow, Logger, Environment
from myagents.src.llms import DummyLLM, OpenAiLLM, QueueLLM
from myagents.src.agents import DummyAgent, BaseAgent, BaseStepCounter, MaxStepCounter, TokenStepCounter
from myagents.src.workflows import ActionFlow, PlanFlow, ReActFlow
from myagents.src.envs import Query
from myagents.src.utils.logger import init_logger

    
class AutoAgentConfig(BaseModel):
    """AutoAgentConfig is the configuration for the AutoAgent.
    """
    step_counters: list[CounterConfig] = Field(default_factory=list[CounterConfig])
    environment: EnvironmentConfig = Field(default=None)
    workflow: WorkflowConfig = Field(default=None)
    
    # Debug and logging settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    save_dir: str = Field(default="stdout")
    
    
class AutoAgent:
    """AutoAgent is a factory for creating agents and allocating them to the environment and workflows.
    """
    
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
    
    def __build_llm(
        self, 
        config: LLMConfig, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> LLM:
        """Build a LLM.
        
        Args:
            config (LLMConfig):
                The configuration for the LLM.
            custom_logger (Logger, optional):
                The custom logger for the LLM. Defaults to loguru logger.
            debug (bool, optional):
                Whether to enable the debug mode for the LLM. Defaults to False.
                
        Returns:
            LLM:
                The LLM.
        """
        provider = config.provider
        
        # Create kwargs for the LLM
        kwargs = {}
        if config.extra_body is not {}:
            kwargs["extra_body"] = config.extra_body
        if config.api_key_field is not None:
            kwargs["api_key_field"] = config.api_key_field
        
        match provider:
            case "openai":
                return OpenAiLLM(
                    base_url=config.base_url,
                    model=config.model,
                    temperature=config.temperature, 
                    custom_logger=custom_logger,
                    debug=debug,
                    **kwargs, 
                )
            case "queue":
                return QueueLLM(
                    model=config.model,
                    base_url=config.base_url,
                    temperature=config.temperature,
                    custom_logger=custom_logger,
                    debug=debug,
                    **kwargs, 
                )
            case "dummy":
                return DummyLLM(
                    model=config.model,
                    base_url=config.base_url,
                    temperature=config.temperature,
                    custom_logger=custom_logger,
                    debug=debug,
                    **kwargs, 
                )
            case _:
                raise ValueError(f"Invalid LLM name: {provider}")
            
    def __build_mcp_client(self, config: MCPConfig) -> MCPClient:
        """Build a MCP client.
        
        Args:
            config (MCPConfig):
                The configuration for the MCP client.
        """
        raise NotImplementedError("MCP client is not implemented yet.")
        
    def __build_agent(
        self, 
        config: AgentConfig, 
        step_counters: list[StepCounter], 
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> Agent:
        """Build an agent.
        
        Args:
            config (AgentConfig):
                The configuration for the agent.
            step_counters (list[StepCounter]):
                The step counters for the agent. Any of one reach the limit, the agent will be stopped. 
            custom_logger (Logger, optional):
                The custom logger for the agent. Defaults to loguru logger.
            debug (bool, optional):
                Whether to enable the debug mode for the agent. Defaults to False.
        
        Returns:
            Agent:
                The agent.
        """
        name = config.name
        
        match name:
            case "base": 
                # Build the LLM
                llm = self.__build_llm(config.llm, custom_logger, debug)
                # Build the MCP client
                # mcp_client = self.build_mcp_client(config.mcp_client)
                # Build the agent
                return BaseAgent(llm=llm, step_counters=step_counters, custom_logger=custom_logger, debug=debug)
            case "dummy":
                # Build the LLM
                llm = self.__build_llm(config.llm, custom_logger, debug)
                # Build the agent
                return DummyAgent(llm=llm, step_counters=step_counters, custom_logger=custom_logger, debug=debug)
            case _:
                raise ValueError(f"Invalid agent name: {name}")
    
    def __build_workflow(
        self, 
        config: WorkflowConfig, 
        step_counters: list[StepCounter], 
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> Workflow:
        """Build a workflow.
        
        Args:
            config (WorkflowConfig):
                The configuration for the workflow. 
            step_counters (list[StepCounter]):
                The step counters for the workflow. Any of one reach the limit, the workflow will be stopped. 
            custom_logger (Logger, optional):
                The custom logger for the workflow. Defaults to loguru logger.
            debug (bool, optional):
                Whether to enable the debug mode for the workflow. Defaults to False.
        """
        flows: dict[str, Workflow] = {}
        # Build workflows recursively
        if len(config.workflows) > 0:
            for workflow in config.workflows:
                flows[workflow.name] = self.__build_workflow(workflow, step_counters, custom_logger, debug)
        
        # Build agents for action flow
        agent = self.__build_agent(config.agent, step_counters)
        
        match config.name:
            case "action":
                # Build the workflow
                return ActionFlow(agent=agent, custom_logger=custom_logger, debug=debug) 
            case "plan":
                # Build the workflow
                return PlanFlow(agent=agent, custom_logger=custom_logger, debug=debug)
            case "react":
                # Build the workflow
                return ReActFlow(
                    agent=agent, 
                    custom_logger=custom_logger, 
                    debug=debug, 
                    **flows,
                )
            case _:
                raise ValueError(f"Invalid workflow name: {config.name}")
            
    def __build_environment(
        self, 
        config: EnvironmentConfig, 
        step_counters: list[StepCounter], 
        custom_logger: Logger = logger, 
        debug: bool = False, 
    ) -> Environment:
        """Build an environment.
        
        Args:
            config (EnvironmentConfig):
                The configuration for the environment.
            step_counters (list[StepCounter]):
                The step counters for the environment. Any of one reach the limit, the environment will be stopped. 
            custom_logger (Logger, optional):
                The custom logger for the environment. Defaults to loguru logger.
            debug (bool, optional):
                Whether to enable the debug mode for the environment. Defaults to False.
        
        Returns:
            Environment:
                The environment.
        """
        agent = self.__build_agent(config.agent, step_counters)
        # Build the workflows
        workflows: dict[str, Workflow] = {}
        for workflow in config.workflows:
            workflows[workflow.name] = self.__build_workflow(workflow, step_counters, custom_logger, debug)
            
        name = config.name
        
        # Build the environment
        match name:
            case "query":
                return Query(agent=agent, react_flow=workflows["react"])
            case _:
                raise ValueError(f"Invalid environment name: {name}")
            
    def auto_build(self, config: AutoAgentConfig) -> Union[Environment, Workflow]:
        """Build an auto agent.
        
        Args:
            config (AutoAgentConfig):
                The configuration for the auto agent.
                
        Returns:
            Union[Environment, Workflow]:
                The environment or workflow.
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
        
        # Build the step counter
        step_counters = [self.build_counter(step_counter) for step_counter in config.step_counters]
        assert len(step_counters) > 0, "No step counters are provided."
        
        # Check if `environment` is provided
        if config.environment is not None:
            # Build the environment
            environment = self.__build_environment(config.environment, step_counters, custom_logger, config.debug)
            # Return the environment
            return environment
        elif config.workflow is not None:
            # Build the workflow
            workflow = self.__build_workflow(config.workflow, step_counters, custom_logger, config.debug)
            # Return the workflow
            return workflow
        
        # No environment or workflow is provided
        raise ValueError("No environment or workflow is provided.")
