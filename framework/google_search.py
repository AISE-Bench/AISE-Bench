import requests
import json
import time
import streamlit as st
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import re
from config import GOOGLE_API_KEY
import serpapi

SERPAPI_KEY = ""  # ← 这里放你的 API Key

def google_search_tool(query: str) -> list:
    """
    使用 SerpApi 的 Google Scholar 搜索接口，返回结构化的 [{title, link, snippet}, ...] 列表

    参数：
        query (str): 搜索关键词

    返回：
        list: 包含搜索结果的列表，每个元素是一个字典
              [
                  {"title": "...", "link": "...", "snippet": "..."},
                  ...
              ]
    """
    try:
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": SERPAPI_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        output = []
        for item in organic_results:
            info = {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            }
            output.append(info)
        return output

    except Exception as e:
        print(f"❌ SerpApi 调用失败: {e}")
        return []
    
def google_search_tool_old_1124(query: str) -> list:
    """
    使用 Serper.dev API 搜索 Google 并返回结构化的 [{title, link, snippet}, ...] 列表
    
    参数：
        query (str): 搜索关键词
        api_key (str): 你的 Serper.dev API Key
    
    返回：
        list: 包含搜索结果的列表，每个元素是一个字典
              例如：
              [
                  {"title": "Google Scholar", "link": "https://scholar.google.com/", "snippet": "..."},
                  ...
              ]
    """
    headers = {
        'X-API-KEY': GOOGLE_API_KEY,
        'Content-Type': 'application/json',
    }
    
    data = {"q": query}
    response = requests.post('https://google.serper.dev/search', headers=headers, json=data)

    if response.status_code != 200:
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print("响应内容：", response.text)
        return []
    
    result = response.json()
    organic_results = result.get("organic", [])
    
    output = []
    for item in organic_results:
        info = {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        }
        output.append(info)

    return output

def google_search_tool_2(query: str) -> list:
 
    headers = {
        "Content-Type": "application/json",
        "X-VE-Source": "google_search",
        "Accept-Encoding": "identity"
    }

    url = f"https://scholar.google.com/scholar?hl=zh-CN&as_sdt=0,5&q={query.replace(' ', '+')}"

    data = {
        "source": "google",
        'url': url,
        "parse": False  # true
    }

    res = requests.post("http://serp.glm.moe:8001/v1/queries", headers=headers, data=json.dumps(data), verify=False)

    # ✅ 判断请求是否成功
    if res.status_code == 200:

        html = res.text  # 你返回的 HTML 内容
        html_json = json.loads(html)
        html_1 = html_json["results"][0]["content"]
        soup = BeautifulSoup(html_1, "lxml")

        # 存储结果
        results = []

        # 每个搜索结果在 Google Scholar HTML 中是 div.gs_r
        for item in soup.select("div.gs_r.gs_or"):
            data = []

            title_tag = item.select_one("h3.gs_rt a")
            pdf_tag = item.select_one(".gs_ggs a")
            abs_tag = item.select_one("div.gs_rs")

            info = {
                "title": title_tag.get_text(" ", strip=True) if title_tag else "",
                "link": pdf_tag.get("href") if pdf_tag else "",
                "snippet": abs_tag.get_text(" ", strip=True) if abs_tag else ""
            }

            # data["title"] = title_tag.get_text(" ", strip=True) if title_tag else ""
            # data["link"] = title_tag.get("href") if title_tag else ""

            # meta_tag = item.select_one("div.gs_a")
            # data["authors_info"] = meta_tag.get_text(" ", strip=True) if meta_tag else ""

            # abs_tag = item.select_one("div.gs_rs")
            # data["abstract"] = abs_tag.get_text(" ", strip=True) if abs_tag else ""

            # pdf_tag = item.select_one(".gs_ggs a")
            # data["pdf_url"] = pdf_tag.get("href") if pdf_tag else ""

            results.append(info)

        return results


# 示例使用
if __name__ == "__main__":

    query = "jie tang "
    
    results = google_search_tool(query)

    # 打印结果（易读格式）
    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # 也可以保存为文件
    with open("1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)