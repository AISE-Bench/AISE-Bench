#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Draft优化器
"""

import sys
import os
from draft_optimizer import DraftOptimizer

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_single_tool_optimization():
    """
    测试单个工具的优化
    """
    print("=== 测试单个工具优化 ===")
    
    # 创建优化器实例
    optimizer = DraftOptimizer(
        model="glm-4.7",
        temperature=0.3,
        top_p=0.9,
        max_tokens=1000,
        episodes=3,
        # 请替换为真实的API密钥
        api_key=""
    )
    
    # 测试工具信息
    test_tool = {
        "name": "search_paper_id",
        "description": "搜索论文ID的工具",
        "required_parameters": {
            "keywords": "搜索关键词",
            "year": "发表年份"
        },
        "optional_parameters": {
            "sort": "排序方式",
            "size": "返回数量"
        }
    }
    
    print("优化前工具信息:")
    print(f"名称: {test_tool['name']}")
    print(f"描述: {test_tool['description']}")
    print(f"必填参数: {test_tool['required_parameters']}")
    print(f"可选参数: {test_tool['optional_parameters']}")
    
    # 执行优化
    optimized_tool = optimizer.optimize_tool_documentation(test_tool)
    
    print("\n优化后工具信息:")
    print(f"名称: {optimized_tool['name']}")
    print(f"描述: {optimized_tool['description']}")
    print(f"必填参数: {optimized_tool['required_parameters']}")
    print(f"可选参数: {optimized_tool['optional_parameters']}")
    
    return optimized_tool

def test_multi_tool_optimization():
    """
    测试多个工具的优化
    """
    print("\n=== 测试多个工具优化 ===")
    
    # 创建优化器实例
    optimizer = DraftOptimizer(
        model="glm-4.7",
        temperature=0.3,
        top_p=0.9,
        max_tokens=1000,
        episodes=2,
        # 请替换为真实的API密钥
        api_key=""
    )
    
    # 测试工具信息
    test_tools = {
        "tool_name": "paper_search_tool",
        "tool_guidelines": {
            "search_paper_id": {
                "description": "搜索论文ID的工具",
                "required_parameters": {
                    "keywords": "搜索关键词"
                }
            },
            "search_paper_detail": {
                "description": "获取论文详细信息的工具",
                "required_parameters": {
                    "paper_ids": "论文ID列表"
                }
            }
        }
    }
    
    print("优化前工具信息:")
    for api_name, api_info in test_tools["tool_guidelines"].items():
        print(f"{api_name}: {api_info['description']}")
    
    # 执行优化
    optimized_tools = optimizer.optimize_tool_documentation(test_tools)
    
    print("\n优化后工具信息:")
    for api_name, api_info in optimized_tools["tool_guidelines"].items():
        print(f"{api_name}: {api_info['description']}")
    
    if "tool_description" in optimized_tools:
        print("\n全局工具描述:")
        print(optimized_tools["tool_description"])
    
    return optimized_tools

if __name__ == "__main__":
    print("开始测试Draft优化器...")
    
    # 测试单个工具优化
    try:
        test_single_tool_optimization()
        print("\n单个工具优化测试完成！")
    except Exception as e:
        print(f"\n单个工具优化测试失败: {str(e)}")
    
    # 测试多个工具优化
    try:
        test_multi_tool_optimization()
        print("\n多个工具优化测试完成！")
    except Exception as e:
        print(f"\n多个工具优化测试失败: {str(e)}")
    
    print("\n所有测试完成！")
