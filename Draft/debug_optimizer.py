#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试Draft优化器
"""

import sys
import os
import traceback
from draft_optimizer import DraftOptimizer

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_api_call():
    """
    调试API调用
    """
    print("=== 调试API调用 ===")
    
    # 创建优化器实例
    optimizer = DraftOptimizer(
        model="glm-4.7",
        temperature=0.3,
        top_p=0.9,
        max_tokens=1000,
        episodes=1,
        api_key=""
    )
    
    # 测试简单的API调用
    test_messages = [
        {"role": "system", "content": "你是一个工具探索专家，根据工具描述生成用户查询和参数。"},
        {"role": "user", "content": "工具描述：搜索论文ID的工具"}
    ]
    
    print("发送测试消息...")
    try:
        response = optimizer.glm_response(test_messages)
        print(f"响应类型: {type(response)}")
        print(f"响应内容: {response}")
        print("API调用成功！")
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        traceback.print_exc()

def debug_prompt_loading():
    """
    调试提示加载
    """
    print("\n=== 调试提示加载 ===")
    
    # 创建优化器实例
    optimizer = DraftOptimizer(
        model="glm-4.7",
        temperature=0.3,
        top_p=0.9,
        max_tokens=1000,
        episodes=1,
        api_key=""
    )
    
    print("加载的提示:")
    for key, value in optimizer.prompts.items():
        print(f"{key}: {value[:100]}...")

def debug_single_step():
    """
    调试单步执行
    """
    print("\n=== 调试单步执行 ===")
    
    # 创建优化器实例
    optimizer = DraftOptimizer(
        model="glm-4.7",
        temperature=0.3,
        top_p=0.9,
        max_tokens=1000,
        episodes=1,
        api_key=""
    )
    
    # 测试工具信息
    test_tool = {
        "name": "search_paper_id",
        "description": "搜索论文ID的工具",
        "required_parameters": {
            "keywords": "搜索关键词"
        }
    }
    
    print("测试工具信息:")
    print(test_tool)
    
    # 测试探索器步骤
    print("\n测试探索器步骤...")
    try:
        last_description = test_tool["description"]
        explorer_prompt = optimizer.prompts['explorer'].format(**{"Tool Description": last_description})
        print(f"探索器提示: {explorer_prompt}")
        
        explorer_messages = [
            {"role": "system", "content": "你是一个工具探索专家，根据工具描述生成用户查询和参数。"},
            {"role": "user", "content": explorer_prompt}
        ]
        
        explorer_response = optimizer.glm_response(explorer_messages)
        print(f"探索器响应: {explorer_response}")
        print("探索器步骤成功！")
    except Exception as e:
        print(f"探索器步骤失败: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("开始调试Draft优化器...")
    
    # 调试API调用
    debug_api_call()
    
    # 调试提示加载
    debug_prompt_loading()
    
    # 调试单步执行
    debug_single_step()
    
    print("\n调试完成！")
