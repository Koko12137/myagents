from myagents.core.interface.prompt import Prompt, PromptGroup


class PlanSystemPrompt(Prompt):
    """PlanSystemPrompt is a prompt for plan workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "system_prompt"

    def string(self) -> str:
        return self.prompt
    

class PlanReasonActPrompt(Prompt):
    """PlanReasonActPrompt is a prompt for plan workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "reason_act_prompt"

    def string(self) -> str:
        return self.prompt


class PlanReflectPrompt(Prompt):
    """PlanReflectPrompt is a prompt for plan workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "reflect_prompt"

    def string(self) -> str:
        return self.prompt


class PlanPromptGroup(PromptGroup):
    """PlanPromptGroup is a prompt group for plan workflow."""
    sub_groups: dict[str, PromptGroup]
    
    def __init__(
        self, 
        system_prompt: str = "prompts/plan/system.md", 
        reason_act_prompt: str = "prompts/plan/reason_act.md", 
        reflect_prompt: str = "prompts/plan/reflect.md",
        sub_groups: dict[str, PromptGroup] = None,
    ) -> None:
        """Initialize the PlanPromptGroup.
        
        Args:
            system_prompt (str): The path to the system prompt file.
            reason_act_prompt (str): The path to the reason and act prompt file.
            reflect_prompt (str): The path to the reflect prompt file.
            sub_groups (dict[str, PromptGroup]): The sub-groups of the plan workflow.
        """
        # Initialize the prompts
        self.prompts = {
            "system_prompt": PlanSystemPrompt(system_prompt),
            "reason_act_prompt": PlanReasonActPrompt(reason_act_prompt),
            "reflect_prompt": PlanReflectPrompt(reflect_prompt),
        }
        # Initialize the sub-groups
        self.sub_groups = sub_groups if sub_groups is not None else {}

    def get_prompt(self, name: str) -> Prompt:
        return self.prompts[name]

    def get_sub_group(self, group_name: str) -> PromptGroup:
        return self.sub_groups[group_name]
