import json
import os

from datasets import load_from_disk

from myagents.core.envs.complex_query import ComplexQuery, OutputType
from myagents.core.factory import AutoAgent, AutoAgentConfig


def load_imo25_data() -> list[str]:
    # 列出 datasets/IMO25 目录下的所有txt文件
    files = os.listdir("datasets/IMO25")
    files = [file for file in files if file.endswith(".txt")]
    # 读取每个txt文件
    data = []
    for file in files:
        with open(os.path.join("datasets/IMO25", file), "r") as f:
            data.append(f.read())
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
        sub_task_depth=2,
        output_type=OutputType.SUMMARY,
    )
    print("-" * 60)
    # Check the answer and the correct answer
    print("answer: ", answer)
    # print("correct answer: ", data[0]["answer"])
