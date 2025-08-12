#!/usr/bin/env python3
"""
简单的测试运行脚本，用于验证Milvus记忆模块的测试
"""

import os
import sys
import subprocess

def main():
    """运行Milvus记忆模块测试"""
    
    # 添加项目根目录到Python路径
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, project_root)
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = project_root + ':' + os.environ.get('PYTHONPATH', '')
    
    print("=" * 60)
    print("开始运行Milvus记忆模块测试")
    print("=" * 60)
    
    # 运行测试
    test_file = os.path.join(os.path.dirname(__file__), 'test_milvus_memory.py')
    
    try:
        # 使用pytest运行测试
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_file, 
            '-v', 
            '--tb=short'
        ], capture_output=True, text=True, cwd=project_root)
        
        print("测试输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ 所有测试通过!")
        else:
            print(f"\n❌ 测试失败，退出码: {result.returncode}")
            
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 