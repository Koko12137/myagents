import json
import os

from myagents.core.envs.complex_query import ComplexQuery, OutputType
from myagents.core.factory import AutoAgent, AutoAgentConfig


def load_imo25_data() -> list[str]:
    ds = json.load(open("datasets/IMO25/data.json"))
    data = [item["problem"] for item in ds]
    return data


async def test_async_query():
    data = load_imo25_data()
    
    # Load API keys from json and convert to environment variables
    with open("configs/api_keys.json", "r") as f:
        api_keys = json.load(f)
        for key, value in api_keys.items():
            os.environ[key] = value
    
    # Create a list of agents according to the config file
    with open("configs/agents_memory.json", "r") as f:
        config = AutoAgentConfig.model_validate(json.load(f))
        
    # Create Factory
    factory = AutoAgent()
    # Build ReActFlow
    query: ComplexQuery = await factory.auto_build(config=config)
    
    # Run the query
    answer = await query.run(
        question="请仔细分析这道数学题，给出正确的答案，所有过程需要用中文作答。", 
        description=data[0],
        sub_task_depth=1,
        output_type=OutputType.SUMMARY,
    )
    print("-" * 60)
    # Check the answer and the correct answer
    print("answer: ", answer)
    # print("correct answer: ", data[0]["answer"])
