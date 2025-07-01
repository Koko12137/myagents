def normalize_string(s: str) -> str:
    """标准化字符串，处理转义字符和特殊字符"""
    if not s:
        return s
    
    # 处理常见的转义字符
    # 将反斜杠转义为普通字符
    s = s.replace('\\', '\\\\')
    # 处理其他可能的转义字符
    s = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    s = s.replace('\'', '\\\'').replace('"', '\\"')
    
    # 移除首尾空白字符
    s = s.strip()
    
    return s


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein距离）"""
    # 标准化字符串
    s1 = normalize_string(s1)
    s2 = normalize_string(s2)
    
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_best_match(target: str, candidates: list[str]) -> str:
    """在候选任务中找到与目标字符串编辑距离最小的任务
    
    Args:
        target (str):
            目标字符串
        candidates (list[str]):
            候选字符串列表
            
    Returns:
        str:
            与目标字符串编辑距离最小的字符串
            
    Raises: 
        KeyError:
            如果候选字符串列表为空。
    """
    if not candidates:
        raise KeyError(f"No sub-tasks found to match with '{target}'")
    
    best_match = None
    min_distance = float('inf')
    target_normalized = normalize_string(target)
    
    # 记录所有匹配结果用于调试
    match_results = []
    
    for question in candidates:
        question_normalized = normalize_string(question)
        distance = levenshtein_distance(target_normalized, question_normalized)
        
        match_results.append({
            'original': question,
            'normalized': question_normalized,
            'distance': distance
        })
        
        if distance < min_distance:
            min_distance = distance
            best_match = question
    
    # 如果没有找到匹配，记录调试信息
    if best_match is None:
        raise KeyError(f"No suitable match found for '{target}'")
    
    return best_match


def safe_string_compare(s1: str, s2: str) -> bool:
    """安全地比较两个字符串，处理转义字符"""
    try:
        return normalize_string(s1) == normalize_string(s2)
    except Exception as e:
        return s1 == s2
