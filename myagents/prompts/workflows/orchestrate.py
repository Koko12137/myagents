from myagents.core.interface import Prompt, PromptGroup
from myagents.prompts.workflows.plan import PlanPromptGroup


class BlueprintSystemPrompt(Prompt):
    """BlueprintSystemPrompt is a prompt for Blueprint workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()
    
    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "system_prompt"

    def string(self) -> str:
        return self.prompt
    

class BlueprintReasonActPrompt(Prompt):
    """BlueprintReasonActPrompt is a prompt for Blueprint workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "reason_act_prompt"

    def string(self) -> str:
        return self.prompt


class BlueprintReflectPrompt(Prompt):
    """BlueprintReflectPrompt is a prompt for Blueprint workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "reflect_prompt"

    def string(self) -> str:
        return self.prompt


class BlueprintPromptGroup(PromptGroup):
    """BlueprintPromptGroup is a prompt group for Blueprint workflow."""
    def __init__(
        self, 
        system_prompt: str = "prompts/blueprint/system.md", 
        reason_act_prompt: str = "prompts/blueprint/reason_act.md", 
        reflect_prompt: str = "prompts/blueprint/reflect.md",
        sub_groups: dict[str, PromptGroup] = None,
    ) -> None:
        """Initialize the BlueprintPromptGroup.
        
        Args:
            system_prompt (str): The path to the system prompt file.
            reason_act_prompt (str): The path to the reason and act prompt file.
            reflect_prompt (str): The path to the reflect prompt file.
        """
        # Initialize the prompts
        self.prompts = {
            "system_prompt": BlueprintSystemPrompt(system_prompt),
            "reason_act_prompt": BlueprintReasonActPrompt(reason_act_prompt),
            "reflect_prompt": BlueprintReflectPrompt(reflect_prompt),
        }
        # Initialize the sub-groups
        self.sub_groups = sub_groups if sub_groups is not None else {}

    def get_prompt(self, name: str) -> Prompt:
        return self.prompts[name]

    def get_sub_group(self, group_name: str) -> PromptGroup:
        return self.sub_groups[group_name]

    def has_prompts(self, names: list[str]) -> bool:
        return all(name in self.prompts for name in names)

    def has_sub_groups(self, group_names: list[str]) -> bool:
        return all(group_name in self.sub_groups for group_name in group_names)


class OrchestratePromptGroup(PlanPromptGroup):
    """OrchestratePromptGroup is a prompt group for Orchestrate workflow."""
    sub_groups: dict[str, PromptGroup]
    
    def __init__(
        self, 
        blueprint_system_prompt: str = "prompts/orchestrate/blueprint/system.md", 
        blueprint_reason_act_prompt: str = "prompts/orchestrate/blueprint/reason_act.md", 
        blueprint_reflect_prompt: str = "prompts/orchestrate/blueprint/reflect.md",
        plan_system_prompt: str = "prompts/orchestrate/plan/system.md", 
        plan_reason_act_prompt: str = "prompts/orchestrate/plan/reason_act.md", 
        plan_reflect_prompt: str = "prompts/orchestrate/plan/reflect.md",
    ) -> None:
        """Initialize the OrchestratePromptGroup.
        
        Args:
            blueprint_system_prompt (str): The path to the blueprint system prompt file.
            blueprint_reason_act_prompt (str): The path to the blueprint reason and act prompt file.
            blueprint_reflect_prompt (str): The path to the blueprint reflect prompt file.
            plan_system_prompt (str): The path to the plan system prompt file.
            plan_reason_act_prompt (str): The path to the plan reason and act prompt file.
            plan_reflect_prompt (str): The path to the plan reflect prompt file.
        """
        # Initialize the blueprint prompt group
        blueprint_prompt_group = BlueprintPromptGroup(
            system_prompt=blueprint_system_prompt, 
            reason_act_prompt=blueprint_reason_act_prompt, 
            reflect_prompt=blueprint_reflect_prompt,
        )
        # Initialize the plan prompt group
        super().__init__(
            system_prompt=plan_system_prompt, 
            reason_act_prompt=plan_reason_act_prompt, 
            reflect_prompt=plan_reflect_prompt,
            sub_groups={
                "blueprint": blueprint_prompt_group,
            },
        )
