import json
import asyncio
import requests
import concurrent.futures
from difflib import SequenceMatcher

from typing_extensions import Annotated, Doc
from openai import OpenAI
from google_search import google_search_tool

from config import (
    CHATGLM_API_KEY,
    CHATGLM_API_BASE,
    DEEPSEEK_API_KEY,
)
from apis import search_paper_id
from language import translate_to_english
from llm import llm_client

def search_paper_title_via_aminer(title):
    params = {"query": title, "needDetails":True, "page":0, "size":20, 'filters': []}
    try:
        res = requests.post("https://searchtest.aminer.cn/aminer-search/search/publication", json=params, timeout=60)
        res_dict = json.loads(res.text)
        papers = res_dict["data"].get("hitList", [])
    except:
        papers = []
    return papers

def parse_user_query_to_structured_params(user_query: str):
    """
    先调用 Google 获取搜索结果，再用大模型总结参数
    """

    user_query = translate_to_english(user_query)

    # 第一步：Google搜索
    google_results = google_search_tool(user_query)
    search_context = "\n".join([
        f"网页标题: {r['title']}\n网页摘要: {r['snippet']}" for r in google_results
    ][:20])  # 限制前20项，避免prompt过长

    # 第二步：提示构造
    system_prompt = """你是一个智能的学术搜索参数提取助手。
用户会输入一个学术查询，你需要根据用户问题 + Google 搜索结果，
提取出可供 search_paper_id 调用的参数。

你的输出为
{{titles: Annotated[list, Doc("论文标题列表")] = [str,str,...]}}

请注意，不要补全标题，网页上是什么就输入什么（删除省略号，不要擅自添加单词）
输出格式为纯JSON，无解释。
"""

    user_prompt = f"""
用户问题如下：
{user_query}

相关的Google搜索摘要：
{search_context}

请基于以上内容，输出JSON.
"""
    
    reply = llm_client(system_prompt, user_prompt)
    print("大模型返回：", reply)
    try:
        clean_reply = reply.strip()

        # 移除 Markdown 代码块符号（例如 ```json 或 ```）
        if clean_reply.startswith("```"):
            clean_reply = clean_reply.strip("`")
            # 如果包含指定语言标记
            if clean_reply.lower().startswith("json"):
                clean_reply = clean_reply[4:]
            # 再去一次首尾的 ```, 空格, 换行
            clean_reply = clean_reply.replace("```", "").strip()

        params = json.loads(clean_reply)

    except json.JSONDecodeError:
        print("⚠️ JSON解析失败，原始返回：", reply)
        params = {}

    return params,search_context

def string_similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度（0~100之间）"""
    return SequenceMatcher(None, a, b).ratio() * 100

def search_paper_id_gs_tool(query: str):
    """
    根据用户查询 query 搜索论文，
    并只保留标题与网页标题相似度>=80%的论文ID。
    - search_paper_title_via_aminer(title) 返回 list，每个元素为 dict。
    - 只取每个 list 的第一个 paper。
    - 等所有并发任务返回后统一过滤。
    """

    params, search_context = parse_user_query_to_structured_params(query)

    idlist = []
    titles_from_context = params.get("titles", []) if isinstance(params, dict) else params

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(search_paper_title_via_aminer, title) for title in titles_from_context]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]  

    for result in results:
        # 每个 result 预期是一个 list
        if not isinstance(result, list) or len(result) == 0:
            continue

        # 只取第一个返回的论文 dict
        paper = result[0]
        if not isinstance(paper, dict):
            continue

        paper_title = paper.get("title", "").strip()
        if not paper_title:
            continue

        for ctx_title in titles_from_context:
            similarity = string_similarity(paper_title.lower(), ctx_title.lower())

            if similarity >= 80:
                idlist.append(paper["id"])
                break  # 匹配到一个即可跳出

    return idlist, search_context

if __name__ == "__main__":
    # print(search_paper_id_gs("请帮我找近5年MIT发表的关于大语言模型的SCI论文，英文文献，按年份排序。"))
    print(search_paper_id_gs_tool("关于智慧博物馆设计方向的文献"))




