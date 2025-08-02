SEMANTIC_MEMORY_EXTRACT_PROMPT = """
## 语义记忆提取任务
【从上下文中提取事实性记忆信息，并判断操作类型】

请从当前的上下文中提取**语义记忆（semantic memory）**，即能够判定为真或假的客观事实，如陈述性信息、数据、属性等。同时需要判断与历史记忆是否存在冲突，并决定相应的操作类型。

### 记忆定义
**语义记忆**：能够判定为真或假的客观事实，包括：
- 陈述性信息（如"小明在2024年5月1日尝试做蛋糕"）
- 数据信息（如"水的沸点是100℃"）
- 属性信息（如"地球是圆的"）
- 时间、地点、人物等客观事实

### 冲突检测规则
请对比历史记忆与当前上下文，判断是否存在冲突：

1. **内容冲突**：相同主题但内容不一致（如历史记忆"水的沸点是100℃"，当前信息"水的沸点是95℃"）
2. **真假冲突**：相同内容但真假判断相反（如历史记忆"地球是平的"为true，当前信息"地球是平的"为false）
3. **信息过时**：历史记忆已被新信息替代或修正

### 操作类型判断
- `"add"`：新信息，历史记忆中不存在相关内容
- `"update"`：存在冲突，需要用新信息更新历史记忆
- `"delete"`：历史记忆已被证明错误或过时，需要删除

### 输出格式
请以 **JSONL格式** 输出结果（每行一个独立JSON对象），包含以下字段：

- `operation`：操作类型，取值为 `"add"`、`"update"` 或 `"delete"`
- 当 `operation` 为 `"add"` 或 `"update"` 时，需要包含 `memory` 字段：
  - `memory`：记忆对象，包含以下字段：
    - `memory_type`：固定为 `"semantic_memory"`
    - `content`：提取的事实性内容
    - `truth_value`：表明事实的真假，取值为 `"true"` 或 `"false"`
- 当 `operation` 为 `"delete"` 时，只需要包含 `memory_id` 字段：
  - `memory_id`：要删除的记忆ID（从相似记忆中获取）

### 示例
**历史记忆**：
```json
{"memory_id": 12345, "memory_type": "semantic_memory", "content": "水的沸点是100℃", "truth_value": "true"}
{"memory_id": 12346, "memory_type": "semantic_memory", "content": "地球是平的", "truth_value": "true"}
{"memory_id": 12347, "memory_type": "semantic_memory", "content": "月球是地球的卫星", "truth_value": "true"}
```

**当前上下文**："小明在2024年5月1日尝试做蛋糕。水的沸点实际上是95℃。地球是圆的。月球不是地球的卫星。"

**提取结果**：
```json
{"operation": "add", "memory": {"memory_type": "semantic_memory", "content": "2024年5月1日小明尝试做蛋糕", "truth_value": "true"}}
{"operation": "update", "memory": {"memory_type": "semantic_memory", "content": "水的沸点是95℃", "truth_value": "true"}}
{"operation": "update", "memory": {"memory_type": "semantic_memory", "content": "地球是圆的", "truth_value": "true"}}
{"operation": "delete", "memory_id": 12347}
```

### 注意事项
- 仔细对比历史记忆与当前信息，准确判断冲突情况
- 确保提取的内容与上下文高度匹配
- 若没有相关语义记忆，无需输出任何内容
- 操作类型必须准确反映与历史记忆的关系

=============

## 相似记忆
{sim_memories}

=============
现在，请从以下上下文中提取语义记忆，并判断操作类型：
"""

# 单个语义记忆项的格式化模板
SEMANTIC_ITEM_FORMAT = """
**语义记忆 {memory_id}**：
{content}

**真假判断**：{truth_value}

---
"""

# 总体语义记忆组装模板
SEMANTIC_FORMAT = """
## 历史语义记忆回顾
【基于历史事实信息指导当前决策】

以下是相关的历史语义记忆，请参考这些事实信息来指导当前的任务执行：

{memory_items}

### 记忆使用指导
- **真实信息记忆**：参考历史中已验证的真实事实和数据
- **错误信息记忆**：注意历史中已被证明错误的信息，避免重复使用
- **客观事实记忆**：基于历史中的客观事实进行决策
- **数据信息记忆**：参考历史中的具体数据和数值信息

### 决策建议
1. **基于真实事实**：优先使用历史中已验证的真实信息
2. **避免错误信息**：注意识别和避免历史中的错误信息
3. **参考客观数据**：利用历史中的客观事实和数据
4. **保持信息更新**：注意信息的时效性和准确性

请基于以上历史语义记忆，为当前任务提供准确的事实依据和信息参考。
"""
