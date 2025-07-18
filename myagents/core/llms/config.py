from pydantic import BaseModel, Field


class CompletionConfig(BaseModel):
    """CompletionConfig is the configuration for the completion.
    
    Attributes:
        tool_choice (str):
            The tool choice to use for the agent.
        exclude_tools (list[str]):
            The tools to exclude from the tool choice.
    """
    tool_choice: str = Field(default="auto")
    exclude_tools: list[str] = Field(default=[])
