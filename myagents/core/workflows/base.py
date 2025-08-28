from enum import Enum

from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Workflow, Workspace, CallStack, PromptGroup
from myagents.core.tools_mixin import ToolsMixin


class BaseWorkflow(Workflow, ToolsMixin):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
        workspace (Workspace):
            The workspace of the workflow.
        call_stack (CallStack):
            The call stack of the workflow.
        
        agent (Agent):
            The agent that is used to work with the workflow.
        prompt_group (PromptGroup):
            The prompt group of the workflow.
        observe_formats (dict[str, str]):
            The format of the observation. The key is the observation name and the value is the format content. 
        sub_workflows (dict[str, Workflow]):
            The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
            sub-workflow instance. 
        run_stage (Enum):
            The run stage of the workflow.
    """
    
    # Tools
    tools: dict[str, FastMcpTool]
    # Workspace
    workspace: Workspace
    # Call stack information
    call_stack: CallStack
    # Agent
    agent: Agent
    # Prompt group
    prompt_group: PromptGroup
    # Observe formats
    observe_formats: dict[str, str]
    # Sub-workflows
    sub_workflows: dict[str, Workflow]
    # Run stage
    run_stage: Enum
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        prompt_group: PromptGroup, 
        observe_formats: dict[str, str] = None, 
        sub_workflows: dict[str, Workflow] = None, 
        **kwargs,
    ) -> None:
        """Initialize the BaseWorkflow.

        Args:
            call_stack (CallStack):
                The call stack of the workflow.
            workspace (Workspace):
                The workspace of the workflow.
            prompt_group (PromptGroup, optional):
                The prompt group of the workflow.
            observe_formats (dict[str, str], optional):
                The formats of the observation. The key is the observation name and the value is the format method name. 
            sub_workflows (dict[str, Workflow], optional):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
                sub-workflow instance. 
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(call_stack=call_stack, **kwargs)
        
        # Initialize the workflow components
        self.prompt_group = prompt_group
        self.observe_formats = observe_formats if observe_formats is not None else {}
        self.sub_workflows = sub_workflows if sub_workflows is not None else {}

        # Initialize the agent
        self.agent = None
        # Initialize the workspace
        self.workspace = workspace

        # Post initialize
        self.post_init()
        
    def get_run_stage(self) -> Enum:
        """Get the run stage of the workflow.
        
        Returns:
            Enum:
                The run stage of the workflow.
        """
        return self.run_stage
    
    def set_run_stage(self, run_stage: Enum) -> None:
        """Set the run stage of the workflow.
        
        Args:
            run_stage (Enum):
                The run stage to set.
        """
        self.run_stage = run_stage
        
    def get_prompt(self) -> str:
        """Get the prompt of the workflow.
        
        Returns:
            str:
                The prompt of the workflow.
        """
        return self.prompts[self.run_stage]
    
    def get_observe_format(self) -> str:
        """Get the observe format of the workflow.
        
        Returns:
            str:
                The observe format of the workflow.
        """
        return self.observe_formats[self.run_stage]
    
    def get_sub_workflow(self, sub_workflow_name: str) -> 'Workflow':
        """Get the sub-workflow of the workflow.
        
        Args:
            sub_workflow_name (str):
                The name of the sub-workflow.
                
        Returns:
            Workflow:
                The sub-workflow of the workflow.
        """
        return self.sub_workflows[sub_workflow_name]
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent to the workflow.
        
        Args:
            agent (Agent):
                The agent to register.
        """
        # Check if the agent is registered
        if self.agent is not None:
            return 
        
        # Register the agent to the workflow
        self.agent = agent
        # Register the agent to the sub-workflows
        for sub_workflow in self.sub_workflows.values():
            sub_workflow.register_agent(agent)
