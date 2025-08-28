from myagents.core.interface.prompt import PromptGroup
from myagents.prompts.workflows.react import ReactPromptGroup
from myagents.prompts.workflows.orchestrate import OrchestratePromptGroup


class PlanAndExecPromptGroup(ReactPromptGroup):
    """PlanAndExecPromptGroup is a prompt group for plan and exec workflow."""
    sub_groups: dict[str, PromptGroup]
    
    def __init__(
        self, 
        blueprint_system_prompt: str = "prompts/blueprint/system.md", 
        blueprint_reason_act_prompt: str = "prompts/blueprint/reason_act.md", 
        blueprint_reflect_prompt: str = "prompts/blueprint/reflect.md",
        plan_system_prompt: str = "prompts/plan/system.md", 
        plan_reason_act_prompt: str = "prompts/plan/reason_act.md", 
        plan_reflect_prompt: str = "prompts/plan/reflect.md",
        exec_system_prompt: str = "prompts/execute/system.md", 
        exec_reason_act_prompt: str = "prompts/execute/reason_act.md", 
        exec_reflect_prompt: str = "prompts/execute/reflect.md",
    ) -> None:
        """Initialize the PlanAndExecPromptGroup.
        
        Args:
            blueprint_system_prompt (str): The path to the blueprint system prompt file.
            blueprint_reason_act_prompt (str): The path to the blueprint reason and act prompt file.
            blueprint_reflect_prompt (str): The path to the blueprint reflect prompt file.
            plan_system_prompt (str): The path to the plan system prompt file.
            plan_reason_act_prompt (str): The path to the plan reason and act prompt file.
            plan_reflect_prompt (str): The path to the plan reflect prompt file.
            exec_system_prompt (str): The path to the execute system prompt file.
            exec_reason_act_prompt (str): The path to the execute reason and act prompt file.
            exec_reflect_prompt (str): The path to the execute reflect prompt file.
        """
        # Initialize the Orchestrate Prompt Group
        orch_prompt_group = OrchestratePromptGroup(
            blueprint_system_prompt=blueprint_system_prompt, 
            blueprint_reason_act_prompt=blueprint_reason_act_prompt, 
            blueprint_reflect_prompt=blueprint_reflect_prompt,
            plan_system_prompt=plan_system_prompt, 
            plan_reason_act_prompt=plan_reason_act_prompt, 
            plan_reflect_prompt=plan_reflect_prompt,
        )
        super().__init__(
            system_prompt=exec_system_prompt, 
            reason_act_prompt=exec_reason_act_prompt, 
            reflect_prompt=exec_reflect_prompt,
            sub_groups={
                "orchestrate": orch_prompt_group,
            },
        )
