import json 

from loguru import logger
from pydantic import BaseModel, Field
from fastmcp.client import Client as MCPClient

from myagents.src.interface import LLM, StepCounter, Agent, RunnableEnvironment, Workflow, OrchestratedFlows, Logger
from myagents.src.llms.openai import OpenAiLLM
from myagents.src.agents.base import BaseAgent, BaseStepCounter
from myagents.src.workflows.act import ActionFlow
from myagents.src.workflows.plan import PlanFlow
from myagents.src.workflows.react import ReActFlow



class CounterConfig(BaseModel):
    limit: int | float = Field(description="The limit of the step counter.")


class LLMConfig(BaseModel):
    provider: str = Field(description="The provider to use.")
    base_url: str = Field(description="The base URL to use.")
    model: str = Field(description="The model to use.")
    temperature: float = Field(description="The temperature to use.")
    api_key_field: str = Field(
        description="The environment variable name to use for the API key.", 
        default="OPENAI_KEY"
    )
    extra_body: dict = Field(
        description="The extra body to use for the request.", 
        default={}
    )
    
    
class MCPConfig(BaseModel):
    server_name: str = Field(description="The name of the MCP server.")
    server_url: str = Field(description="The URL of the MCP server.")
    server_port: int = Field(description="The port of the MCP server.")
    auth_token: str = Field(description="The authentication token to use.")
    
    
class AgentConfig(BaseModel):
    name: str = Field(description="The name of the agent.")
    llm: LLMConfig = Field(description="The configuration for the LLM.")
    # mcp_client: MCPConfig = Field(description="The configuration for the MCP client.")
    
    
class WorkflowConfig(BaseModel):
    name: str = Field(description="The name of the workflow.")
    agent: AgentConfig = Field(description="The configuration for the agent.")
    
    
class OrchestratedFlowsConfig(BaseModel):
    name: str = Field(description="The name of the orchestrated workflow.")
    agent: AgentConfig = Field(description="The configuration for the agent.")
    workflows: list[WorkflowConfig] = Field(description="The configurations for the workflows.")
    
    
class AutoAgent:
    """AutoAgent is a factory for creating agents and allocating them to the environment and workflows.
    """
    
    def build_counter(self, config: CounterConfig, name: str = "base") -> StepCounter:
        """Build a step counter.
        
        Args:
            config (CounterConfig):
                The configuration for the step counter.
            name (str, optional):
                The name of the step counter. Defaults to "base".
        """
        match name:
            case "base":
                return BaseStepCounter(limit=config.limit)
            case _:
                raise ValueError(f"Invalid step counter name: {name}")
    
    def build_llm(self, config: LLMConfig, provider: str = "openai") -> LLM:
        """Build a LLM.
        
        Args:
            config (LLMConfig):
                The configuration for the LLM.
            provider (str, optional):
                The provider of the LLM. Defaults to "openai".
        """
        match provider:
            case "openai":
                return OpenAiLLM(
                    provider=config.provider,
                    base_url=config.base_url,
                    model=config.model,
                    temperature=config.temperature,
                    extra_body=config.extra_body, 
                )
            case _:
                raise ValueError(f"Invalid LLM name: {provider}")
            
    def build_mcp_client(self, config: MCPConfig, name: str = "base") -> MCPClient:
        """Build a MCP client.
        
        Args:
            config (MCPConfig):
                The configuration for the MCP client.
        """
        raise NotImplementedError("MCP client is not implemented yet.")
        
    def build_agent(self, config: AgentConfig, step_counter: StepCounter, name: str = "base") -> Agent:
        """Build an agent.
        
        Args:
            config (AgentConfig):
                The configuration for the agent.
            step_counter (StepCounter):
                The global step counter for the agent.
            name (str, optional):
                The name of the agent. Defaults to "base".
        """
        match name:
            case "base": 
                # Build the LLM
                llm = self.build_llm(config.llm, provider=config.llm.provider)
                # Build the MCP client
                # mcp_client = self.build_mcp_client(config.mcp_client)
                # Build the agent
                return BaseAgent(llm=llm, step_counter=step_counter)
            case _:
                raise ValueError(f"Invalid agent name: {name}")
    
    def build_workflow(
        self, 
        config: WorkflowConfig, 
        step_counter: StepCounter, 
        custom_logger: Logger = logger, 
        debug: bool = False,
    ) -> Workflow:
        """Build a workflow.
        
        Args:
            config (WorkflowConfig):
                The configuration for the workflow. 
            step_counter (StepCounter):
                The step counter for the workflow.
            custom_logger (Logger, optional):
                The custom logger for the workflow. Defaults to loguru logger.
            debug (bool, optional):
                Whether to enable the debug mode for the workflow. Defaults to False.
        """
        # Build agents for action flow
        agent = self.build_agent(config.agent, step_counter)
        
        match config.name:
            case "act":
                # Build the workflow
                return ActionFlow(agent=agent, custom_logger=custom_logger, step_counter=step_counter, debug=debug) 
            case "plan":
                # Build the workflow
                return PlanFlow(agent=agent, custom_logger=custom_logger, step_counter=step_counter, debug=debug)
            case _:
                raise ValueError(f"Invalid workflow name: {config.name}")

    def build_orchestrated_workflows(
        self, 
        config: OrchestratedFlowsConfig, 
        step_counter: StepCounter,
        custom_logger: Logger = logger, 
        debug: bool = False,
    ) -> OrchestratedFlows:
        """Build an orchestrated workflow.
        
        Args:
            config (OrchestratedWorkflowConfig):
                The configuration for the orchestrated workflow.
            step_counter (StepCounter):
                The step counter for the orchestrated workflow.
            custom_logger (Logger, optional):
                The custom logger for the orchestrated workflow. Defaults to loguru logger.
            debug (bool, optional):
                Whether to enable the debug mode for the orchestrated workflow. Defaults to False.
        """
        # Build agents for orchestrated workflows
        agent = self.build_agent(config.agent, step_counter)
        
        match config.name:
            case "react":
                # Build agents for orchestrated workflows
                agents = {}
                for workflow in config.workflows:
                    agents[workflow.agent.name] = self.build_agent(workflow.agent, step_counter)
                # Build the workflow
                return ReActFlow(
                    agent=agent, 
                    plan_agent=agents["plan_agent"], 
                    action_agent=agents["action_agent"], 
                    custom_logger=custom_logger, 
                    debug=debug, 
                )
            case _:
                raise ValueError(f"Invalid orchestrated workflow name: {config.name}")
        
        
def build_runnable_environment() -> RunnableEnvironment:
    """Build a runnable environment.
    """
    raise NotImplementedError("Runnable environment is not implemented yet.")
