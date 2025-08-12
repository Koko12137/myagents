from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """语言模型配置类"""
    provider: str = Field(description="要使用的提供商")
    base_url: str = Field(description="要使用的基础URL")
    model: str = Field(description="要使用的模型")
    temperature: float = Field(description="要使用的温度参数")
    api_key_field: str = Field(
        description="用于API密钥的环境变量名称", 
        default="OPENAI_KEY"
    )
    extra_body: dict = Field(
        description="请求的额外参数", 
        default={}
    )
