from pydantic import BaseModel, Field


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
