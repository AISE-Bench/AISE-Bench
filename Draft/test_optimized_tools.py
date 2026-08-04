"""
测试优化工具信息的使用
"""
import json
import os
from caller import TaskExecutor

def test_optimized_tools_loading():
    """测试从 optimized_tools.json 加载优化后的工具信息"""
    
    print("=" * 60)
    print("测试 1: 测试从 optimized_tools.json 加载优化工具")
    print("=" * 60)
    
    # 检查 optimized_tools.json 是否存在
    if not os.path.exists("optimized_tools.json"):
        print("❌ optimized_tools.json 不存在，跳过此测试")
        return False
    
    # 创建一个简单的任务列表
    tasks = [
        {
            "name": "search_paper_id",
            "params": {"keywords": ["machine learning"]},
            "order": 1,
            "rely": []
        }
    ]
    
    # 创建 TaskExecutor 实例
    executor = TaskExecutor(tasks)
    
    # 调用 optimize_tools 方法（应该从文件加载优化信息）
    optimized_tools = executor.optimize_tools()
    
    print(f"\n✅ 成功加载 {len(optimized_tools)} 个优化工具")
    print(f"✅ 内存中的优化工具数量: {len(executor.optimized_tools)}")
    
    # 测试 get_tool_info 方法是否使用优化后的信息
    tool_info = executor.get_tool_info("search_paper_id")
    
    print(f"\n=== 测试 get_tool_info 方法 ===")
    print(f"工具名称: {tool_info.get('name')}")
    print(f"工具描述: {tool_info.get('description')}")
    
    # 检查是否使用了优化后的信息
    if "optimization_rounds" in tool_info:
        print(f"✅ 成功使用优化后的工具信息（包含 optimization_rounds 字段）")
        return True
    else:
        print(f"⚠️ 使用的是默认工具信息（不包含 optimization_rounds 字段）")
        return False

def test_batch_optimized_tools_loading():
    """测试从 batch_optimized_tools.json 加载优化后的工具信息"""
    
    print("\n" + "=" * 60)
    print("测试 2: 测试从 batch_optimized_tools.json 加载优化工具")
    print("=" * 60)
    
    # 检查 batch_optimized_tools.json 是否存在
    if not os.path.exists("batch_optimized_tools.json"):
        print("❌ batch_optimized_tools.json 不存在，跳过此测试")
        return False
    
    # 创建一个简单的任务列表
    tasks = [
        {
            "name": "search_paper_id",
            "params": {"keywords": ["machine learning"]},
            "order": 1,
            "rely": []
        }
    ]
    
    # 创建 TaskExecutor 实例
    executor = TaskExecutor(tasks)
    
    # 调用 optimize_tools 方法（应该从文件加载优化信息）
    optimized_tools = executor.optimize_tools()
    
    print(f"\n✅ 成功加载 {len(optimized_tools)} 个优化工具")
    print(f"✅ 内存中的优化工具数量: {len(executor.optimized_tools)}")
    
    # 测试 get_tool_info 方法是否使用优化后的信息
    tool_info = executor.get_tool_info("search_paper_id")
    
    print(f"\n=== 测试 get_tool_info 方法 ===")
    print(f"工具名称: {tool_info.get('name')}")
    print(f"工具描述: {tool_info.get('description')}")
    
    # 检查是否使用了优化后的信息
    if "optimization_rounds" in tool_info:
        print(f"✅ 成功使用优化后的工具信息（包含 optimization_rounds 字段）")
        return True
    else:
        print(f"⚠️ 使用的是默认工具信息（不包含 optimization_rounds 字段）")
        return False

def test_default_tool_info():
    """测试默认工具信息（当没有优化文件时）"""
    
    print("\n" + "=" * 60)
    print("测试 3: 测试默认工具信息（无优化文件时）")
    print("=" * 60)
    
    # 创建一个简单的任务列表
    tasks = [
        {
            "name": "search_paper_id",
            "params": {"keywords": ["machine learning"]},
            "order": 1,
            "rely": []
        }
    ]
    
    # 创建 TaskExecutor 实例
    executor = TaskExecutor(tasks)
    
    # 测试 get_tool_info 方法（应该使用默认信息）
    tool_info = executor.get_tool_info("search_paper_id")
    
    print(f"\n=== 测试 get_tool_info 方法 ===")
    print(f"工具名称: {tool_info.get('name')}")
    print(f"工具描述: {tool_info.get('description')}")
    print(f"必需参数: {tool_info.get('required_parameters')}")
    print(f"可选参数: {list(tool_info.get('optional_parameters', {}).keys())[:5]}...")
    
    # 检查是否使用了默认信息
    if "optimization_rounds" not in tool_info:
        print(f"✅ 成功使用默认工具信息")
        return True
    else:
        print(f"⚠️ 意外使用了优化后的工具信息")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试优化工具信息的使用")
    print("=" * 60)
    
    results = []
    
    # 测试 1: 从 optimized_tools.json 加载
    results.append(("optimized_tools.json 加载测试", test_optimized_tools_loading()))
    
    # 测试 2: 从 batch_optimized_tools.json 加载
    results.append(("batch_optimized_tools.json 加载测试", test_batch_optimized_tools_loading()))
    
    # 测试 3: 默认工具信息
    results.append(("默认工具信息测试", test_default_tool_info()))
    
    # 输出测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n总计: {passed_count}/{total_count} 个测试通过")
    
    if passed_count == total_count:
        print("🎉 所有测试通过！")
    else:
        print("⚠️ 部分测试失败，请检查代码")
