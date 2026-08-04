import requests
import json
import time
import streamlit as st
from bs4 import BeautifulSoup
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
    max_retries = 3
    base_wait_time = 2
    
    for attempt in range(max_retries):
        try:
            print(f"🔍 尝试使用 SerpApi 搜索 (尝试 {attempt+1}/{max_retries}): {query[:50]}...")
            # 使用 serpapi.Client 类（兼容旧版本）
            client = serpapi.Client(api_key=SERPAPI_KEY)
            results = client.search({
                "engine": "google_scholar",
                "q": query
            })
            organic_results = results.get("organic_results", [])

            output = []
            for item in organic_results:
                info = {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                }
                output.append(info)
            
            print(f"✅ SerpApi 搜索成功，找到 {len(output)} 条结果")
            return output

        except Exception as e:
            error_msg = str(e)
            print(f"❌ SerpApi 调用失败: {error_msg}")
            
            # 检查是否是 429 错误（请求过多）
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = base_wait_time * (2 ** attempt)
                    print(f"⏳ 遇到请求限制，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
            
            # 其他错误或重试次数用尽，尝试备用方案
            print("🔄 尝试使用备用搜索方案...")
            return google_search_tool_fallback(query)
    
    # 所有尝试都失败
    print("❌ 所有搜索方案都失败了")
    return []

def google_search_tool_fallback(query: str) -> list:
    """
    备用搜索方案，当 SerpApi 不可用时使用
    """
    try:
        print(f"🔍 使用备用方案搜索: {query[:50]}...")
        
        # 方案1: 尝试使用 google_search_tool_2
        results = google_search_tool_2(query)
        if results:
            print(f"✅ 备用方案搜索成功，找到 {len(results)} 条结果")
            return results
        
        # 方案2: 尝试直接解析 Google Scholar HTML
        print("🔄 尝试直接解析 Google Scholar...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 编码查询字符串
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = f"https://scholar.google.com/scholar?hl=en&q={encoded_query}"
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            # 解析搜索结果
            for item in soup.select("div.gs_r.gs_or.gs_scl"):
                title_elem = item.select_one("h3.gs_rt")
                link_elem = item.select_one("h3.gs_rt a")
                snippet_elem = item.select_one("div.gs_rs")
                
                if title_elem and link_elem:
                    info = {
                        "title": title_elem.get_text(" ", strip=True).replace("[HTML]", "").replace("[PDF]", ""),
                        "link": link_elem.get("href"),
                        "snippet": snippet_elem.get_text(" ", strip=True) if snippet_elem else ""
                    }
                    results.append(info)
            
            if results:
                print(f"✅ 直接解析成功，找到 {len(results)} 条结果")
                return results
        
    except Exception as e:
        print(f"❌ 备用搜索方案失败: {e}")
    
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