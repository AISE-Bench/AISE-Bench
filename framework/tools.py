import requests
import asyncio
from typing import List, Optional

CUSTOM_SERVER_BASE = "http://36.103.177.237:8507/"


async def call_custom_server(endpoint: str, params: dict = None):
    """调用自定义服务器API"""
    if params is None:
        params = {}
    
    url = CUSTOM_SERVER_BASE + endpoint
    headers = {"Content-Type": "application/json"}
    
    try:
        response = await asyncio.to_thread(requests.post, url=url, json=params, headers=headers)
        return response.json()
    except Exception as e:
        print(f"API调用失败: {e}")
        raise


async def search_paper_id_tool(**kwargs):
    """搜索论文"""
    return await call_custom_server("search_paper_id", kwargs)


async def search_paper_detail_tool(**kwargs):
    """获取论文详情"""
    return await call_custom_server("search_paper_detail", kwargs)


async def search_author_id_tool(**kwargs):
    """搜索作者"""
    return await call_custom_server("search_author_id", kwargs)


async def search_author_detail_tool(**kwargs):
    """获取作者详情"""
    return await call_custom_server("search_author_detail", kwargs)


async def search_venue_id_tool(**kwargs):
    """搜索期刊/会议"""
    return await call_custom_server("search_venue_id", kwargs)


async def search_venue_detail_tool(**kwargs):
    """获取期刊/会议详情"""
    return await call_custom_server("search_venue_detail", kwargs)


async def search_org_id_tool(**kwargs):
    """搜索机构"""
    return await call_custom_server("search_org_id", kwargs)


async def search_org_detail_tool(**kwargs):
    """获取机构详情"""
    return await call_custom_server("search_org_detail", kwargs)


if __name__ == "__main__":
    print("测试所有API")
    print("=" * 80)
    
    print("【search_paper_id_tool】")
    print(asyncio.run((search_paper_id_tool(
        keywords=["nlp"],
        coauthors="jie tang,zhangfanjin",
    ))))
    print("----------------------------------")
    
    print("【search_paper_detail_tool】")
    print(asyncio.run((search_paper_detail_tool(
        paper_ids=["5390a05a20f70186a0e4a81d"]
    ))))
    print("----------------------------------")
    
    print("【search_author_id_tool】")
    print(asyncio.run((search_author_id_tool(
        interest=["deep learning"],
        org="tsinghua university",
        nation=["China"],
        name=["Yoshua Bengio"],
        size=10
    ))))
    print("----------------------------------")
    
    print("【search_author_detail_tool】")
    print(asyncio.run((search_author_detail_tool(
        author_ids=["560e732e45ce1e59613dbf3b"]
    ))))
    print("----------------------------------")
    
    print("【search_venue_id_tool】")
    print(asyncio.run((search_venue_id_tool(
        keywords=["machine learning"],
        name="NeurIPS",
        name_search_type="fuzzy",
        category="Computer Science",
        category_source="3",
        quartile="A",
        size=10
    ))))
    print("----------------------------------")
    
    print("【search_venue_detail_tool】")
    print(asyncio.run((search_venue_detail_tool(
        ids=["5ea1e340edb6e7d53c011a4c"]
    ))))
    print("----------------------------------")
    
    print("【search_org_id_tool】")
    print(asyncio.run((search_org_id_tool(
        orgs=["Tsinghua University"]
    ))))
    print("----------------------------------")
    
    print("【search_org_detail_tool】")
    print(asyncio.run((search_org_detail_tool(
        ids=["2uqJy1T1i1P0Z2mo7lZtFXmM4e7"]
    ))))
    print("----------------------------------")
    
    print("=" * 80)
    print("测试完成")
