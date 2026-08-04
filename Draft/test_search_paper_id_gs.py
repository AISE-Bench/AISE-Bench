"""
测试 search_paper_id_gs 功能
"""
import asyncio
import time
from new_tool_async import search_paper_id_gs

def test_search_paper_id_gs():
    """测试 search_paper_id_gs 功能"""
    print("=" * 60)
    print("开始测试 search_paper_id_gs 功能")
    print("=" * 60)
    
    # 测试查询
    test_queries = [
        "盾构技术 中国式现代化 作用 影响",
        "大模型 推理加速 方法",
        "reinforcement learning terminal reward design"
    ]
    
    success_count = 0
    total_count = len(test_queries)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i}/{len(test_queries)} ---")
        print(f"查询内容: {query}")
        
        try:
            # 添加超时控制
            start_time = time.time()
            # 调用 search_paper_id_gs
            result = asyncio.run(search_paper_id_gs(query))
            end_time = time.time()
            
            print(f"调用耗时: {end_time - start_time:.2f} 秒")
            
            # 检查结果格式
            if isinstance(result, tuple) and len(result) == 2:
                paper_ids = result[0]
                search_context = result[1]
                
                print(f"✅ 调用成功！")
                print(f"找到的论文ID数量: {len(paper_ids)}")
                print(f"论文ID列表: {paper_ids}")
                print(f"搜索上下文长度: {len(search_context)} 字符")
                
                if paper_ids:
                    print(f"✅ 成功找到相关论文！")
                    success_count += 1
                else:
                    print(f"⚠️ 未找到相关论文")
            else:
                print(f"❌ 调用失败：返回格式不正确")
                print(f"返回结果: {result}")
                
        except asyncio.TimeoutError:
            print(f"❌ 调用超时")
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            # 只打印错误类型，不打印完整堆栈
            print(f"错误类型: {type(e).__name__}")
        
        # 每测试一个查询后休息一下，避免API限流
        if i < len(test_queries):
            print("⏳ 休息3秒，避免API限流...")
            time.sleep(3)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print(f"测试结果: {success_count}/{total_count} 个查询成功")
    
    if success_count > 0:
        print("🎉 测试通过！search_paper_id_gs 功能正常")
    else:
        print("⚠️ 所有测试都失败了，可能是网络问题或API配置问题")

if __name__ == "__main__":
    test_search_paper_id_gs()
