from myagents.core.interface.prompt import Prompt, PromptGroup


class ReactSystemPrompt(Prompt):
    """ReactSystemPrompt is a prompt for react workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "system_prompt"

    def string(self) -> str:
        return self.prompt


class ReactReasonActPrompt(Prompt):

    """ReactReasonActPrompt is a prompt for react workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "reason_act_prompt"

    def string(self) -> str:
        return self.prompt


class ReactReflectPrompt(Prompt):
    """ReactReflectPrompt is a prompt for react workflow."""
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()

    def __str__(self) -> str:
        return self.prompt

    def get_name(self) -> str:
        return "reflect_prompt"

    def string(self) -> str:
        return self.prompt


class ReactPromptGroup(PromptGroup):
    """ReactPromptGroup is a prompt group for react workflow."""
    sub_groups: dict[str, PromptGroup]
    
    def __init__(
        self, 
        system_prompt: str = "prompts/execute/system.md", 
        reason_act_prompt: str = "prompts/execute/reason_act.md", 
        reflect_prompt: str = "prompts/execute/reflect.md",
        sub_groups: dict[str, PromptGroup] = None,
    ) -> None:
        """Initialize the ReactPromptGroup.
        
        Args:
            system_prompt (str): The path to the system prompt file.
            reason_act_prompt (str): The path to the reason and act prompt file.
            reflect_prompt (str): The path to the reflect prompt file.
            sub_groups (dict[str, PromptGroup]): The sub-groups of the react workflow.
        """
        # Initialize the prompts
        self.prompts = {
            "system_prompt": ReactSystemPrompt(system_prompt),
            "reason_act_prompt": ReactReasonActPrompt(reason_act_prompt),
            "reflect_prompt": ReactReflectPrompt(reflect_prompt),
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
