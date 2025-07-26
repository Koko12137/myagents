import asyncio
import json
import os
import re
import copy
from typing import Dict, Any, Set
from tqdm import tqdm

from loguru import logger
from datasets import load_dataset
from openai import AsyncOpenAI


PROMPT = """请仔细分析这道数学题，尽快给出正确的答案。最终答案应只包含数字，不要有任何其他内容。

## 格式要求
<answer>数字</answer>

【警告】：禁止过度思考

{description}
"""


# 加载API密钥
with open("configs/api_keys.json", "r", encoding="utf-8") as f:
    api_keys = json.load(f)
    for key, value in api_keys.items():
        os.environ[key] = value


# 初始化AsyncOpenAI客户端
client = AsyncOpenAI(
    api_key=os.getenv("LOCAL_KEY"), 
    base_url="http://localhost:8889/v1"
)


def extract_answer_from_response(response: str) -> str:
    """
    只从<answer>标签中提取内容，并清洗出数字（支持整数和小数）。
    """
    match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        # 提取第一个数字（支持整数和小数，正负号）
        number_match = re.search(r'[-+]?\d*\.?\d+', answer_content)
        if number_match:
            return number_match.group(0)
        else:
            return ""
    # 没有<answer>标签，直接返回空字符串
    return ""


def ensure_output_directory(output_file: str) -> None:
    """
    确保输出文件的目录存在
    
    Args:
        output_file (str): 输出文件路径
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")


def load_existing_results(output_file: str) -> Set[str]:
    """
    加载已存在的结果，返回已处理的问题集合
    
    Args:
        output_file (str): 结果文件路径
        
    Returns:
        Set[str]: 已处理的问题集合
    """
    processed_questions = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        processed_questions.add(data["question"])
            logger.info(f"加载已处理的问题: {len(processed_questions)} 个")
        except Exception as e:
            logger.warning(f"加载已存在结果时出错: {e}")
    else:
        logger.info(f"结果文件不存在，将创建新文件: {output_file}")
    
    return processed_questions


async def write_single_result(result: Dict[str, Any], output_file: str) -> None:
    """
    异步写入单个结果到jsonl文件
    
    Args:
        result (Dict[str, Any]): 单个结果
        output_file (str): 输出文件路径
    """
    # 准备记录数据
    record = {
        "question": result["question"],
        "ground_truth": result["ground_truth"],
        "predicted_answer": result["predicted_answer"], 
        "is_correct": result["is_correct"],
        "stop_reason": result["stop_reason"], 
    }
    
    # 异步写入文件
    await asyncio.get_event_loop().run_in_executor(
        None, 
        lambda: write_to_file_sync(output_file, record)
    )


def write_to_file_sync(output_file: str, record: Dict[str, Any]) -> None:
    """
    同步写入文件（在线程池中执行）
    
    Args:
        output_file (str): 输出文件路径
        record (Dict[str, Any]): 要写入的记录
    """
    # 确保目录存在
    ensure_output_directory(output_file)
    
    # 追加模式写入，不会覆盖现有内容
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()  # 立即刷新缓冲区


async def process_single_question(
    description: str, 
    ground_truth: str, 
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    处理单个问题的并发函数，支持流式输出
    
    Args:
        description (str): 问题描述
        ground_truth (str): 真实答案
        semaphore (asyncio.Semaphore): 信号量控制并发
        
    Returns:
        Dict[str, Any]: 处理结果
    """
    # 获取信号量
    await semaphore.acquire()
    try:
        # 构建提示词
        prompt = copy.deepcopy(PROMPT).format(description=description)
        # 开启流式输出
        response_stream = await client.chat.completions.create(
            model="Qwen/Qwen3-14B-AWQ", 
            max_completion_tokens=1024*8, 
            temperature=0.8, 
            presence_penalty=1.0, 
            messages=[{"role": "user", "content": prompt}],
            stream=True, 
            extra_body={
                "enable_thinking": False, # 禁用思考控制
            }
        )
        # 流式拼接内容
        full_content = ""
        async for chunk in response_stream:
            delta = chunk.choices[0].delta.content if chunk.choices[0].delta else ""
            if delta:
                print(delta, end="", flush=True)  # 实时输出到终端
                full_content += delta
        print()  # 换行

        # 提取答案
        predicted_answer = extract_answer_from_response(full_content)
        stop_reason = getattr(chunk.choices[0], "finish_reason", "stream_end")

        return {
            "question": description,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "is_correct": predicted_answer == ground_truth, 
            "stop_reason": stop_reason
        }
        
    except Exception as e:
        logger.error(f"处理问题时出错: {description[:50]}... 错误: {str(e)}")
        return {
            "question": description,
            "ground_truth": ground_truth,
            "predicted_answer": "",
            "is_correct": False,
            "stop_reason": "error"
        }
    finally:
        semaphore.release()


def calculate_accuracy_from_jsonl(jsonl_file: str) -> Dict[str, float]:
    """
    从jsonl文件计算准确率
    
    Args:
        jsonl_file (str): jsonl文件路径
        
    Returns:
        Dict[str, float]: 准确率指标
    """
    predictions = []
    ground_truths = []
    
    if not os.path.exists(jsonl_file):
        return {"accuracy": 0.0, "correct": 0, "total": 0, "percentage": 0.0}
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                predictions.append(data["predicted_answer"])
                ground_truths.append(data["ground_truth"])
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    total = len(predictions)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "percentage": accuracy * 100
    }


async def run_benchmark(
    dataset_path: str,
    output_file: str,
    max_samples: int = -1,
    concurrency: int = 4,
    input_field: str = "question",
    output_field: str = "answer",
) -> Dict[str, Any]:
    """
    运行基准测试
    
    Args:
        dataset_path (str): 数据集路径
        output_file (str): 结果输出文件
        max_samples (int): 最大测试样本数
        concurrency (int): 并发数量
        input_field (str): 输入字段
        output_field (str): 输出字段
        
    Returns:
        Dict[str, Any]: 测试结果
    """
    # 确保输出目录存在
    ensure_output_directory(output_file)
    
    # 加载数据集
    logger.info(f"加载数据集: {dataset_path}")
    dataset = load_dataset(dataset_path)
    if "train" in dataset:
        dataset = dataset["train"]
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 加载已处理的问题
    processed_questions = load_existing_results(output_file)
    
    # 过滤掉已处理的问题
    filtered_dataset = dataset.filter(lambda x: x[input_field] not in processed_questions)
    logger.info(f"过滤后数据集大小: {len(filtered_dataset)}")
    
    # 限制样本数量
    if max_samples > 0 and len(filtered_dataset) > max_samples:
        filtered_dataset = filtered_dataset.select(range(max_samples))
        logger.info(f"限制测试样本数为: {max_samples}")
    
    if len(filtered_dataset) == 0:
        logger.info("没有新的问题需要处理")
        # 计算现有结果的准确率
        accuracy_metrics = calculate_accuracy_from_jsonl(output_file)
        return {
            "accuracy_metrics": accuracy_metrics,
            "total_samples": accuracy_metrics["total"],
            "correct_samples": accuracy_metrics["correct"],
            "error_samples": accuracy_metrics["total"] - accuracy_metrics["correct"],
            "error_rate": (accuracy_metrics["total"] - accuracy_metrics["correct"]) / accuracy_metrics["total"] if accuracy_metrics["total"] > 0 else 0.0,
            "concurrency": concurrency,
            "output_file": output_file,
            "new_processed": 0
        }
    
    # 准备测试数据
    questions = filtered_dataset[input_field]
    answers = filtered_dataset[output_field]
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(concurrency)
    
    logger.info(f"开始运行基准测试，并发数: {concurrency}")
    
    # 使用tqdm创建进度条
    correct_count = 0
    processed_count = 0
    
    # 获取初始准确率
    initial_accuracy = calculate_accuracy_from_jsonl(output_file)
    total_correct = initial_accuracy["correct"]
    total_processed = initial_accuracy["total"]
    
    with tqdm(total=len(questions), desc="处理进度", unit="题", mininterval=0.1, dynamic_ncols=True) as pbar:
        # 创建所有任务
        tasks = []
        for question, answer in zip(questions, answers):
            task = process_single_question(question, str(answer), semaphore)
            tasks.append(task)
        
        # 使用asyncio.as_completed来处理完成的任务
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                
                # 异步写入结果
                await write_single_result(result, output_file)
                
                # 更新计数
                processed_count += 1
                if result["is_correct"]:
                    correct_count += 1
                    total_correct += 1
                total_processed += 1
                
                # 更新进度条
                current_accuracy = (total_correct / total_processed) * 100
                pbar.set_postfix({
                    "正确率": f"{current_accuracy:.1f}%",
                    "正确": total_correct,
                    "总数": total_processed,
                    "新增": processed_count
                })
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"任务执行异常: {e}")
                pbar.update(1)
    
    # 计算最终准确率
    accuracy_metrics = calculate_accuracy_from_jsonl(output_file)
    
    # 汇总结果
    summary = {
        "accuracy_metrics": accuracy_metrics,
        "total_samples": accuracy_metrics["total"],
        "correct_samples": accuracy_metrics["correct"],
        "error_samples": accuracy_metrics["total"] - accuracy_metrics["correct"],
        "error_rate": (accuracy_metrics["total"] - accuracy_metrics["correct"]) / accuracy_metrics["total"] if accuracy_metrics["total"] > 0 else 0.0,
        "concurrency": concurrency,
        "output_file": output_file,
        "new_processed": processed_count
    }
    
    logger.info(f"测试完成！结果已保存到: {output_file}")
    logger.info(f"准确率: {accuracy_metrics['percentage']:.2f}% ({accuracy_metrics['correct']}/{accuracy_metrics['total']})")
    logger.info(f"本次新增处理: {processed_count} 个问题")
    
    return summary


async def main(
    dataset_path: str,
    output_file: str,
    input_field: str,
    output_field: str, 
    concurrency: int = 1,
):
    """主函数"""
    # 运行基准测试
    results = await run_benchmark(
        dataset_path=dataset_path,
        output_file=output_file,
        max_samples=-1,  # 可以调整测试样本数
        concurrency=concurrency,  # 并发数量
        input_field=input_field,
        output_field=output_field
    )
    
    # 打印详细结果
    print("\n" + "="*50)
    print("基准测试结果")
    print("="*50)
    print(f"总样本数: {results['total_samples']}")
    print(f"正确样本数: {results['correct_samples']}")
    print(f"错误样本数: {results['error_samples']}")
    print(f"准确率: {results['accuracy_metrics']['percentage']:.2f}%")
    print(f"错误率: {results['error_rate']*100:.2f}%")
    print(f"并发数: {results['concurrency']}")
    print(f"本次新增处理: {results['new_processed']} 个问题")
    print(f"结果文件: {results['output_file']}")


if __name__ == "__main__":
    asyncio.run(main(
        dataset_path="Maxwell-Jia/AIME_2024",
        output_file="benchmark/raw_llm_aime_nothink.jsonl",
        input_field="Problem",
        output_field="Answer",
        concurrency=1
    ))

