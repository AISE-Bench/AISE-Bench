#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DRAFT优化流程
"""

import json
import sys
from from_plan_to_result import optimize_tool_documentation, openai_embedding, compute_similarity_and_bleu


def test_openai_api():
    """
    测试OpenAI API是否正常工作
    """
    print("\n=== 测试OpenAI API ===")
    try:
        # 测试Embedding API
        embedding = openai_embedding("test")
        print(f"✓ Embedding API 测试成功，返回嵌入向量长度: {len(embedding)}")
        
        # 测试相似度计算
        similarity_score = compute_similarity_and_bleu("test sentence", "test sentence")
        print(f"✓ 相似度计算测试成功，相似度分数: {similarity_score}")
        
        return True
    except Exception as e:
        print(f"✗ API测试失败: {str(e)}")
        return False


def test_draft_optimization():
    """
    测试DRAFT优化流程
    """
    print("\n=== 测试DRAFT优化流程 ===")
    
    # 创建一个测试工具信息
    test_tool = {
        "category": "academic",
        "name": "search_paper_id",
        "description": "根据关键词搜索论文ID",
        "required_parameters": {
            "keywords": "论文关键词列表"
        },
        "optional_parameters": {
            "years": "发表年份列表",
            "size": "返回结果数量"
        }
    }
    
    print(f"原始工具描述: {test_tool['description']}")
    
    try:
        # 运行优化流程
        optimized_tool = optimize_tool_documentation(test_tool, episodes=1)
        print(f"优化后工具描述: {optimized_tool['description']}")
        print("✓ DRAFT优化流程测试成功")
        return True
    except Exception as e:
        print(f"✗ 优化流程测试失败: {str(e)}")
        return False


def main():
    """
    主测试函数
    """
    print("开始测试DRAFT优化流程...")
    
    # 测试OpenAI API
    api_success = test_openai_api()
    
    # 测试DRAFT优化流程
    draft_success = test_draft_optimization()
    
    print("\n=== 测试结果汇总 ===")
    print(f"OpenAI API测试: {'通过' if api_success else '失败'}")
    print(f"DRAFT优化流程测试: {'通过' if draft_success else '失败'}")
    
    if api_success and draft_success:
        print("\n🎉 所有测试通过！DRAFT优化流程已成功移植。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查配置和代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
