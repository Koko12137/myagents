import json
import os

from myagents.src.envs.query import Query
from myagents.src.factory import AutoAgent, AutoAgentConfig


async def test_async_query():
    question = "请你讲一讲陀思妥耶夫斯基的罪与罚"
    description = "我需要知道陀思妥耶夫斯基的罪与罚结构化的内容"
    
    # Load API keys from json and convert to environment variables
    with open("configs/api_keys.json", "r") as f:
        api_keys = json.load(f)
        for key, value in api_keys.items():
            os.environ[key] = value
    
    # Create a list of agents according to the config file
    with open("configs/agents.json", "r") as f:
        config = AutoAgentConfig.model_validate(json.load(f))
    
    # Create Factory
    factory = AutoAgent()
    # Build ReActFlow
    query: Query = factory.auto_build(
        config=config, 
    )
    
    # Run the query
    answer = await query.run(question, description)
    # Print the answer
    print(answer)
