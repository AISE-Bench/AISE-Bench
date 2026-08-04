"""
测试修复后的谷歌搜论文功能
"""
from google_search import google_search_tool
import time

def test_google_search_fix():
    """测试修复后的谷歌搜论文功能"""
    print("=" * 60)
    print("开始测试修复后的谷歌搜论文功能")
    print("=" * 60)
    
    # 测试查询
    test_queries = [
        "在全球化时代的数字化浪潮中，数字技术以其巨大的变革潜能，正引领人类社会步入一个全新的数字化时代",
        "盾构技术 中国式现代化 作用 影响",
        "大模型 推理加速 方法"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i}/{len(test_queries)} ---")
        print(f"查询内容: {query}")
        
        # 测试修复后的 google_search_tool
        print(f"\n测试 google_search_tool (带重试和备用方案):")
        start_time = time.time()
        try:
            results = google_search_tool(query)
            end_time = time.time()
            print(f"   ✅ 调用成功！耗时: {end_time - start_time:.2f}秒")
            print(f"   找到的结果数量: {len(results)}")
            
            if results:
                print(f"   前3个结果:")
                for j, result in enumerate(results[:3], 1):
                    print(f"   {j}. 标题: {result.get('title', 'N/A')[:80]}...")
                    print(f"      链接: {result.get('link', 'N/A')}")
            else:
                print(f"   ⚠️ 未找到相关结果")
                
        except Exception as e:
            end_time = time.time()
            print(f"   ❌ 调用失败: {e} (耗时: {end_time - start_time:.2f}秒)")
        
        # 避免过快请求
        if i < len(test_queries):
            print("\n⏳ 休息5秒，避免API限流...")
            time.sleep(5)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_google_search_fix()
