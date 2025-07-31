MEMORY_EXTRACT_PROMPT = """
## 记忆提取任务
【从上下文中提取关键记忆信息】
请从当前的上下文中提取关键记忆信息，并按照 **semantic memory（事实性记忆）**、**episode memory（事件记忆）**、**procedural memory（程序性记忆）** 三种类型分类，以 **JSONL格式** 输出结果（即每行一个独立JSON对象）。具体要求如下：

### 一、记忆类型及字段说明
1. **semantic_memory（事实性记忆）**  
   - 定义：能够判定为真或假的客观事实，如陈述性信息、数据、属性等。  
   - 字段：  
     - `memory_type`：固定为 `"semantic_memory"`  
     - `content`：提取的事实性内容  
     - `truth_value`：表明事实的真假，取值为 `"true"` 或 `"false"`  

   示例：  
   - `{"memory_type": "semantic_memory", "content": "水在标准大气压下的沸点是100℃", "truth_value": "true"}`  
   - `{"memory_type": "semantic_memory", "content": "地球是宇宙的中心", "truth_value": "false"}`  


2. **episode_memory（事件记忆）**  
   - 定义：事件发生过程中产生的反馈（如结果、感受、影响等），与具体情境相关。  
   - 字段：  
     - `memory_type`：固定为 `"episode_memory"`  
     - `content`：提取的事件反馈内容  
     - `positive_impact`：表明反馈的积极与否，取值为 `"true"`（积极）、`"false"`（消极）、`"none"`（未知/中性）  

   示例：  
   - `{"memory_type": "episode_memory", "content": "完成任务后获得领导表扬", "positive_impact": "true"}`  
   - `{"memory_type": "episode_memory", "content": "提交的报告被退回修改", "positive_impact": "false"}`  
   - `{"memory_type": "episode_memory", "content": "发送的邮件尚未收到回复", "positive_impact": "none"}`  


3. **procedural_memory（程序性记忆）**  
   - 定义：指导性、命令性内容，包括“做什么、不做什么、怎么做、为什么做、为什么不做”等行动指引。  
   - 字段：  
     - `memory_type`：固定为 `"procedural_memory"`  
     - `content`：提取的程序性内容  
     - 可选补充字段（根据内容选择性添加，无需全部包含）：  
       - `what`：说明“要做/不要做的具体事项”  
       - `how`：说明“操作步骤/方法”  
       - `why`：说明“做某事的原因/目的”  
       - `whynot`：说明“不做某事的原因/禁止理由”  

   示例：  
   - `{"memory_type": "procedural_memory", "content": "使用灭火器前需先拔掉保险销", "what": "使用灭火器前的准备动作", "how": "拔掉保险销"}`  
   - `{"memory_type": "procedural_memory", "content": "禁止在会议室吸烟", "what": "禁止的行为：在会议室吸烟", "whynot": "维持会议室环境整洁"}`  
   - `{"memory_type": "procedural_memory", "content": "每天需记录工作进度，以便跟踪项目状态", "what": "每天记录工作进度", "why": "跟踪项目状态"}`  


### 二、输出格式要求
- 必须使用 **JSONL格式**，即每行一个独立的JSON对象，不包含数组或外层包裹。  
- 每个JSON对象需包含 `memory_type` 和 `content` 字段，其他字段根据记忆类型补充（如 `truth_value`、`positive_impact`、`what` 等）。  
- 若某类记忆无对应信息，无需输出该类型的JSON行。  


### 三、完整示例
- **上下文**：  
  “小明在2024年5月1日尝试做蛋糕，食谱上说需先预热烤箱至180℃（这是错误的，正确温度应为160℃）。他按照步骤混合了面粉和鸡蛋，过程中发现面粉不够，导致面团太稀，有点失望。食谱还提到，烤好后要静置10分钟再脱模，因为热蛋糕容易变形。”

- **提取结果（JSONL）**：  
```json
{"memory_type": "semantic_memory", "content": "2024年5月1日小明尝试做蛋糕", "truth_value": "true"}
{"memory_type": "semantic_memory", "content": "食谱说烤蛋糕需预热烤箱至180℃", "truth_value": "false"}
{"memory_type": "semantic_memory", "content": "烤蛋糕的正确预热温度是160℃", "truth_value": "true"}
{"memory_type": "episode_memory", "content": "混合面粉和鸡蛋时发现面粉不够，导致面团太稀", "positive_impact": "false"}
{"memory_type": "episode_memory", "content": "因面团太稀而感到失望", "positive_impact": "false"}
{"memory_type": "procedural_memory", "content": "做蛋糕需先混合面粉和鸡蛋", "what": "混合面粉和鸡蛋", "how": "按步骤混合"}
{"memory_type": "procedural_memory", "content": "蛋糕烤好后要静置10分钟再脱模", "what": "蛋糕烤好后的操作：静置10分钟再脱模", "why": "避免热蛋糕变形"}
```


请严格按照上述要求提取信息，确保字段完整、格式正确，且内容与上下文高度匹配。

=============
现在，请从以下上下文中提取记忆：
"""


