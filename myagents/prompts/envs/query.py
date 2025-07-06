NAME = "基础多轮问答环境"


PROFILE = """
问答环境，用于回答问题。回复内容可以是以下的类型：
- 总结：总结当前任务的答案。
- 文档：修改当前任务的答案。
- 选择：选择当前任务的答案。
"""


QUERY_SYSTEM_PROMPT = """
You are a helpful assistant. You are given a task and a response. You need to post process the response.
"""


QUERY_POST_PROCESS_PROMPT = """
以上是你对任务的回答拆解成【行号 - 内容】的格式。

# 🔄 阶段规范：任务结果总结或修订
**核心职责**：  
 ✓ 根据任务的描述和答案，执行 {output_type} 的输出。
 
## 命令解释
{command_explanation}

## 🛠 可用工具（任务结果总结或修订阶段）
以下是你在下一步可以使用的工具，这些工具可以帮你完成任务。
{tools}

## 📜 格式约束（阶段通用）
<think>
这里是你对答案的思考，你需要根据任务的描述和答案，进行一些修正。
</think>
<finish_flag>
当你认为你已经完成了最终答案的修订，请将这个标记设置为 True，否则设置为 False。
</finish_flag>

## 🌐 任务上下文信息（动态注入）
任务描述:
{task}
"""
