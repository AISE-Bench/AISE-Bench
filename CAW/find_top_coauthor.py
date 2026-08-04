import json
import os
import asyncio
from caller import TaskExecutor

# 定义查询任务
query = "清华大学的唐杰的合作者中发文量最多的是谁？"

# 构建多步骤任务计划
tasks = [
    {
        "name": "search_author_id",
        "rely": [],
        "order": 1,
        "params": {
            "name": "Tang Jie",
            "org": "Tsinghua University",
            "size": 10
        }
    },
    {
        "name": "search_paper_id",
        "rely": ["search_author_id"],
        "order": 2,
        "params": {
            "use_topic": True,
            "author_terms": ["Tang Jie", "Jie Tang"],
            "org_terms": ["Tsinghua University"],
            "size": 10
        }
    },
    {
        "name": "search_paper_detail",
        "rely": ["search_paper_id"],
        "order": 3,
        "params": {
            "ids": []
        }
    },
    {
        "name": "search_author_detail",
        "rely": ["search_paper_detail"],
        "order": 4,
        "params": {
            "ids": []
        }
    }
]


async def main():
    executor = TaskExecutor(tasks)
    inputs, outputs = await executor.run()
    
    print("=== 执行结果 ===")
    print(f"输入: {inputs}")
    print(f"输出: {outputs}")
    
    # 分析结果
    if "search_paper_detail" in outputs:
        paper_detail_result = outputs["search_paper_detail"]
        # 确保结果是列表
        if isinstance(paper_detail_result, list):
            # 提取所有合作者
            coauthors = {}
            for paper in paper_detail_result:
                if "authors" in paper:
                    for author in paper["authors"]:
                        if isinstance(author, dict) and "name" in author:
                            author_name = author["name"]
                            if author_name not in coauthors:
                                coauthors[author_name] = 0
                            coauthors[author_name] += 1
            
            print("\n=== 合作者统计 ===")
            for name, count in coauthors.items():
                print(f"{name}: {count}篇合作论文")
        
        # 提取合作者ID并获取详细信息
        if "search_author_detail" in outputs:
            author_detail_result = outputs["search_author_detail"]
            if isinstance(author_detail_result, list):
                print("\n=== 合作者论文发表量 ===")
                max_pub_num = 0
                max_pub_author = None
                
                for author_detail in author_detail_result:
                    if isinstance(author_detail, dict) and "name" in author_detail and "pub_num" in author_detail:
                        name = author_detail["name"]
                        pub_num = author_detail["pub_num"]
                        print(f"{name}: {pub_num}篇论文")
                        
                        if pub_num > max_pub_num:
                            max_pub_num = pub_num
                            max_pub_author = name
                
                if max_pub_author:
                    print(f"\n=== 结果 ===")
                    print(f"清华大学的唐杰的合作者中发文量最多的是: {max_pub_author}，共发表{max_pub_num}篇论文")


if __name__ == "__main__":
    asyncio.run(main())