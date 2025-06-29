#!/usr/bin/env python3
"""
测试BaseDocument类的功能
"""

from .docs import BaseDocument, DocumentLogAction, DocumentLog, FormatType


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建原始文档
    original_content = """第一行
第二行
第三行
第四行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    print(f"当前内容:\n{doc.format(FormatType.ARTICLE)}")
    print(f"原始行数: {len(doc.original_lines)}")
    print(f"当前行数: {len(doc.current_lines)}")
    print()


def test_insert_operation():
    """测试插入操作"""
    print("=== 测试插入操作 ===")
    
    original_content = """第一行
第二行
第三行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 在第2行后插入新行
    logs = [DocumentLog(action=DocumentLogAction.INSERT, line=2, content="插入的新行\n")]
    doc.modify(logs)
    print(f"插入后的内容:\n{doc.format(FormatType.ARTICLE)}")
    print(f"日志数量: {len(doc.logs)}")
    print()


def test_replace_operation():
    """测试替换操作"""
    print("=== 测试替换操作 ===")
    
    original_content = """第一行
第二行
第三行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 替换第2行
    logs = [DocumentLog(action=DocumentLogAction.REPLACE, line=2, content="替换后的第二行\n")]
    doc.modify(logs)
    print(f"替换后的内容:\n{doc.format(FormatType.ARTICLE)}")
    print()


def test_delete_operation():
    """测试删除操作"""
    print("=== 测试删除操作 ===")
    
    original_content = """第一行
第二行
第三行
第四行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 删除第2行
    logs = [DocumentLog(action=DocumentLogAction.DELETE, line=2, content="")]
    doc.modify(logs)
    print(f"删除后的内容:\n{doc.format(FormatType.ARTICLE)}")
    print()


def test_multiple_operations():
    """测试多个操作"""
    print("=== 测试多个操作 ===")
    
    original_content = """第一行
第二行
第三行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 执行多个操作
    logs = [
        DocumentLog(action=DocumentLogAction.INSERT, line=2, content="插入行1\n"),
        DocumentLog(action=DocumentLogAction.REPLACE, line=3, content="替换后的第三行\n"),
        DocumentLog(action=DocumentLogAction.INSERT, line=5, content="插入行2\n")
    ]
    doc.modify(logs)
    
    print(f"多次操作后的内容:\n{doc.format(FormatType.ARTICLE)}")
    print(f"日志数量: {len(doc.logs)}")
    
    # 显示所有日志
    print("所有日志:")
    for i, log in enumerate(doc.logs, 1):
        print(f"  {i}. {log.action.value} 第{log.line}行: {repr(log.content)}")
    print()


def test_apply_diff():
    """测试应用diff"""
    print("=== 测试应用diff ===")
    
    original_content = """原始第一行
原始第二行
原始第三行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 创建diff日志
    diff_logs = [
        DocumentLog(action=DocumentLogAction.REPLACE, line=1, content="修改后的第一行\n"),
        DocumentLog(action=DocumentLogAction.INSERT, line=3, content="插入的新行\n"),
        DocumentLog(action=DocumentLogAction.DELETE, line=4, content="")
    ]
    
    # 应用diff
    doc.apply_diff(diff_logs)
    print(f"应用diff后的内容:\n{doc.format(FormatType.ARTICLE)}")
    print()


def test_reset_functionality():
    """测试重置功能"""
    print("=== 测试重置功能 ===")
    
    original_content = """第一行
第二行
第三行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 进行一些修改
    logs = [
        DocumentLog(action=DocumentLogAction.REPLACE, line=2, content="修改后的第二行\n"),
        DocumentLog(action=DocumentLogAction.INSERT, line=3, content="插入的新行\n")
    ]
    doc.modify(logs)
    print(f"修改后的内容:\n{doc.format(FormatType.ARTICLE)}")
    
    # 重置到原始状态
    doc.reset_to_original()
    print(f"重置后的内容:\n{doc.format(FormatType.ARTICLE)}")
    print(f"日志数量: {len(doc.logs)}")
    print()


def test_format_functionality():
    """测试格式化功能"""
    print("=== 测试格式化功能 ===")
    
    original_content = """第一行内容
第二行内容
第三行内容"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 进行一些修改
    logs = [
        DocumentLog(action=DocumentLogAction.INSERT, line=2, content="插入的新行\n"),
        DocumentLog(action=DocumentLogAction.REPLACE, line=4, content="替换后的第四行\n")
    ]
    doc.modify(logs)
    
    print("修改后的内容:")
    print(f"文章格式:\n{doc.format(FormatType.ARTICLE)}")
    print()
    print(f"行号格式:\n{doc.format(FormatType.LINE_NUMBER)}")
    print()
    
    # 测试默认格式（文章格式）
    print(f"默认格式:\n{doc.format()}")
    print()


def test_format_with_empty_lines():
    """测试包含空行的格式化"""
    print("=== 测试包含空行的格式化 ===")
    
    original_content = """第一行

第三行

第五行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{repr(doc.format(FormatType.ORIGINAL))}")
    
    print("行号格式:")
    print(doc.format(FormatType.LINE_NUMBER))
    print()
    
    print("文章格式:")
    print(doc.format(FormatType.ARTICLE))
    print()


def test_format_with_complex_modifications():
    """测试复杂修改后的格式化"""
    print("=== 测试复杂修改后的格式化 ===")
    
    original_content = """原始第一行
原始第二行
原始第三行"""
    
    doc = BaseDocument(original_content)
    
    # 复杂的修改序列
    logs = [
        DocumentLog(action=DocumentLogAction.REPLACE, line=1, content="修改后的第一行\n"),
        DocumentLog(action=DocumentLogAction.INSERT, line=2, content="插入行A\n"),
        DocumentLog(action=DocumentLogAction.INSERT, line=3, content="插入行B\n"),
        DocumentLog(action=DocumentLogAction.DELETE, line=5, content=""),
        DocumentLog(action=DocumentLogAction.REPLACE, line=4, content="替换后的第四行\n")
    ]
    doc.modify(logs)
    
    print("复杂修改后的结果:")
    print("行号格式:")
    print(doc.format(FormatType.LINE_NUMBER))
    print()
    
    print("文章格式:")
    print(doc.format(FormatType.ARTICLE))
    print()
    
    print(f"日志数量: {len(doc.logs)}")
    print("所有日志:")
    for i, log in enumerate(doc.logs, 1):
        print(f"  {i}. {log.action.value} 第{log.line}行: {repr(log.content)}")
    print()


def test_all_format_types():
    """测试所有格式化类型"""
    print("=== 测试所有格式化类型 ===")
    
    original_content = """原始第一行
原始第二行
原始第三行"""
    
    doc = BaseDocument(original_content)
    
    # 进行修改
    logs = [
        DocumentLog(action=DocumentLogAction.REPLACE, line=1, content="修改后的第一行\n"),
        DocumentLog(action=DocumentLogAction.INSERT, line=2, content="插入的新行\n")
    ]
    doc.modify(logs)
    
    print("原始格式:")
    print(doc.format(FormatType.ORIGINAL))
    print()
    
    print("文章格式:")
    print(doc.format(FormatType.ARTICLE))
    print()
    
    print("行号格式:")
    print(doc.format(FormatType.LINE_NUMBER))
    print()
    
    print("默认格式:")
    print(doc.format())
    print()


def test_batch_modify():
    """测试批量修改功能"""
    print("=== 测试批量修改功能 ===")
    
    original_content = """第一行
第二行
第三行
第四行"""
    
    doc = BaseDocument(original_content)
    print(f"原始内容:\n{doc.format(FormatType.ORIGINAL)}")
    
    # 批量修改：插入、替换、删除
    batch_logs = [
        DocumentLog(action=DocumentLogAction.INSERT, line=1, content="插入到开头\n"),
        DocumentLog(action=DocumentLogAction.REPLACE, line=3, content="替换第三行\n"),
        DocumentLog(action=DocumentLogAction.DELETE, line=5, content=""),
        DocumentLog(action=DocumentLogAction.INSERT, line=5, content="插入到末尾\n")
    ]
    
    doc.modify(batch_logs)
    print(f"批量修改后的内容:\n{doc.format(FormatType.ARTICLE)}")
    print(f"行号格式:\n{doc.format(FormatType.LINE_NUMBER)}")
    print(f"日志数量: {len(doc.logs)}")
    print()


if __name__ == "__main__":
    test_basic_functionality()
    test_insert_operation()
    test_replace_operation()
    test_delete_operation()
    test_multiple_operations()
    test_apply_diff()
    test_reset_functionality()
    test_format_functionality()
    test_format_with_empty_lines()
    test_format_with_complex_modifications()
    test_all_format_types()
    test_batch_modify() 