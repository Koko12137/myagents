import asyncio
import json
import os
import random
from typing import Union
from traceback import format_exc

from loguru import logger
from datasets import load_from_disk
from openai import AsyncOpenAI

from myagents.core.envs.task import BaseTask, TaskContextView


PROMPT = """
# 阶段规范：任务分解规划阶段
**阶段核心任务**： 
- 创建完整的任务层级结构蓝图，拆分规划不同层级的任务
- 定义每个子任务的信息需求，包括任务描述、是否叶子节点、子任务列表等
- 标记叶子节点任务，当所有叶子节点都被执行后父节点才会被执行 

## 行动约束
- 本阶段必须完成**整个任务树**的顶层设计
- 需预见所有潜在子任务层级
- 你可以看看自己都有什么工具可以用，这些工具可以帮你完成任务。
- `Is Leaf=True` 仅当任务**不再可分**且**可直接执行**
- 非叶子任务必须包含`Sub-Tasks`属性
- 所有描述必须基于上下文需求，禁止虚构超出观察到的上下文范围的任务

## 输出格式
<think>
分析任务分解逻辑和层级设计依据
</think>
<orchestration>
这里写你的任务分解规划
</orchestration>

## 任务信息
{task_info}
"""


async def process_single_task(
    client: AsyncOpenAI,
    prompt: str, 
) -> list[dict[str, Union[str, dict]]]:
    """处理单个任务

    Args:
        client (AsyncOpenAI): 异步OpenAI客户端
        prompt (str): 提示词
    
    Returns:
        list[dict[str, Union[str, dict]]]: 
            处理结果，包含所有步骤的输出
    """
    task = BaseTask(question="请认真思考后回答以下问题：", description=prompt)
    # 转换Task为str
    task_str = TaskContextView(task).format(layer=1)
    # 构建提示词
    prompt = PROMPT.format(task_info=task_str)
    # 构建消息
    history = [
        {"role": "user", "content": prompt}
    ]
    # 调用OpenAI API
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-14B-AWQ",
        messages=history, 
        temperature=1.2,
    )
    history.append({
        "role": "assistant", 
        "content": response.choices[0].message.content
    })
    return history


async def run_sample(
    client: AsyncOpenAI,
    dataset_path: str,
    output_dir: str, 
    num_workers: int = 5, 
    max_samples: int = 300, 
) -> list[list[dict[str, Union[str, dict]]]]:
    """运行样本处理

    Args:
        client (AsyncOpenAI): 异步OpenAI客户端
        dataset_path (str): 数据集路径
        output_dir (str): 输出目录
        num_workers (int, optional): 工作线程数. 默认为2
        max_samples (int, optional): 最大样本数. 默认为500

    Returns:
        list[list[dict[str, Union[str, dict]]]]: 
            所有任务的处理结果，每个任务的处理结果是一个列表，列表中是每个步骤的输出
    """
    # 加载数据集
    dataset = load_from_disk(dataset_path)
    logger.info(f"加载数据集，数据集大小: {len(dataset)}")
    
    # 加载已经处理过的数据
    if os.path.exists(os.path.join(output_dir, "results.jsonl")):
        with open(os.path.join(output_dir, "results.jsonl"), "r", encoding="utf-8") as f:
            processed_data = [json.loads(line) for line in f]
            
        # 过滤掉已经处理过的数据
        dataset = dataset.filter(lambda x: x['question'] not in [item['question'] for item in processed_data])
        logger.info(f"过滤掉已处理过的数据，剩余数据: {len(dataset)}")
    
    # 随机采样
    if len(dataset) > max_samples and max_samples > 0:
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    
    # 创建信号量来控制并发
    semaphore = asyncio.Semaphore(num_workers)
    # 创建jsonl文件
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    
    async def process_with_semaphore(item):
        async with semaphore:
            try:
                result = await process_single_task(client, item['question'])
                if result is not None:
                    # 将结果写入jsonl文件
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        json.dump({
                            "question": item['question'],
                            "result": result
                        }, f, ensure_ascii=False)
                        f.write("\n")
                    
                    logger.info(f"已处理并保存: {item['question'][:50]}...")
                return result
            except Exception as e:
                logger.error(f'处理提示词时发生错误: {item["question"]}')
                logger.error(f'错误信息: {e}, 调用栈: {format_exc()}')
                return None
    
    # 创建所有任务
    tasks = [process_with_semaphore(item) for item in dataset]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    # 过滤掉None结果（错误的任务）
    return [r for r in results if r is not None]


async def main():
    # 加载API密钥
    with open("configs/api_keys.json", "r") as f:
        api_keys = json.load(f)
        for key, value in api_keys.items():
            os.environ[key] = value
    
    # 创建OpenAI客户端
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_KEY"], 
        base_url="http://192.168.152.104:45519/v1", 
    )
        
    # 设置数据集路径
    dataset_path = "datasets/GAOKAO-Bench/processed/train"
    
    # 创建输出目录
    output_dir = f"datasets/generated_rejected_gaokao_qwen3_14b_awq_5"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 运行样本处理
    results = await run_sample(client, dataset_path, output_dir, max_samples=-1)
    
    # 打印处理结果统计
    print(f"\n处理完成!")
    print(f"成功处理样本数: {len(results)}")
    print(f"结果已保存到jsonl文件中")


if __name__ == "__main__":
    asyncio.run(main())
    