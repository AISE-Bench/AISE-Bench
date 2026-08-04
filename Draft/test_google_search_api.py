"""
测试谷歌搜论文 API
"""
from google_search import google_search_tool, google_search_tool_2

def test_google_search_apis():
    """测试谷歌搜论文 API"""
    print("=" * 60)
    print("开始测试谷歌搜论文 API")
    print("=" * 60)
    
    # 测试查询
    test_queries = [
        "盾构技术 中国式现代化 作用 影响",
        "大模型 推理加速 方法",
        "reinforcement learning terminal reward design"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i}/{len(test_queries)} ---")
        print(f"查询内容: {query}")
        
        # 测试 google_search_tool (SerpApi)
        print(f"\n1. 测试 google_search_tool (SerpApi):")
        try:
            results = google_search_tool(query)
            print(f"   ✅ 调用成功！")
            print(f"   找到的结果数量: {len(results)}")
            
            if results:
                print(f"   第一个结果:")
                print(f"   标题: {results[0].get('title', 'N/A')}")
                print(f"   链接: {results[0].get('link', 'N/A')}")
                print(f"   摘要: {results[0].get('snippet', 'N/A')[:100]}...")
            else:
                print(f"   ⚠️ 未找到相关结果")
                
        except Exception as e:
            print(f"   ❌ 调用失败: {e}")
        
        # 测试 google_search_tool_2 (备用方法)
        print(f"\n2. 测试 google_search_tool_2 (备用方法):")
        try:
            results = google_search_tool_2(query)
            print(f"   ✅ 调用成功！")
            print(f"   找到的结果数量: {len(results)}")
            
            if results:
                print(f"   第一个结果:")
                print(f"   标题: {results[0].get('title', 'N/A')}")
                print(f"   链接: {results[0].get('link', 'N/A')}")
                print(f"   摘要: {results[0].get('snippet', 'N/A')[:100]}...")
            else:
                print(f"   ⚠️ 未找到相关结果")
                
        except Exception as e:
            print(f"   ❌ 调用失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_google_search_apis()
