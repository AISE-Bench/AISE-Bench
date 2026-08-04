import asyncio
from typing import List, Optional, Dict, Any

# 导入现有的API调用工具
from tools import (
    search_paper_id_tool,
    search_paper_detail_tool,
    search_author_id_tool,
    search_author_detail_tool,
    search_venue_id_tool,
    search_venue_detail_tool,
    search_org_id_tool,
    search_org_detail_tool
)
from new_tool import search_paper_id_gs_tool
from new_tool_async import search_paper_id_gs

# 工具函数库，供生成的代码调用
async def search_paper_id(
    keywords: Optional[List[str]] = None,
    authors: Optional[List[str]] = None,
    venues: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    size: int = 10
) -> Dict[str, Any]:
    """搜索论文ID"""
    params = {}
    if keywords:
        params["keywords"] = keywords
    if authors:
        params["author"] = authors
    if venues:
        params["venues"] = venues
    if years:
        params["years"] = years
    if size:
        params["size"] = size
    
    try:
        result = await search_paper_id_tool(**params)
        # 确保返回格式正确
        if not isinstance(result, dict):
            return {"data": [], "error": f"Invalid response format: {result}"}
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_paper_detail(paper_ids: List[str]) -> Dict[str, Any]:
    """获取论文详情"""
    try:
        result = await search_paper_detail_tool(paper_ids=paper_ids)
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_author_id(
    name: Optional[str] = None,
    org: Optional[str] = None,
    interest: Optional[List[str]] = None,
    size: int = 10
) -> Dict[str, Any]:
    """搜索作者ID"""
    params = {}
    if name:
        params["name"] = name
    if org:
        params["org"] = org
    if interest:
        params["interest"] = interest
    if size:
        params["size"] = size
    
    try:
        result = await search_author_id_tool(**params)
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_author_detail(author_ids: List[str]) -> Dict[str, Any]:
    """获取作者详情"""
    try:
        result = await search_author_detail_tool(author_ids=author_ids)
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_venue_id(
    name: Optional[str] = None,
    category: Optional[str] = None,
    quartile: Optional[str] = None,
    size: int = 10
) -> Dict[str, Any]:
    """搜索期刊/会议ID"""
    params = {}
    if name:
        params["name"] = name
    if category:
        params["category"] = category
    if quartile:
        params["quartile"] = quartile
    if size:
        params["size"] = size
    
    try:
        result = await search_venue_id_tool(**params)
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_venue_detail(venue_ids: List[str]) -> Dict[str, Any]:
    """获取期刊/会议详情"""
    try:
        result = await search_venue_detail_tool(ids=venue_ids)
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_org_id(orgs: List[str]) -> Dict[str, Any]:
    """搜索机构ID"""
    try:
        result = await search_org_id_tool(orgs=orgs)
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_org_detail(org_ids: List[str]) -> Dict[str, Any]:
    """获取机构详情"""
    try:
        result = await search_org_detail_tool(ids=org_ids)
        return result
    except Exception as e:
        return {"error": str(e)}

async def search_paper_id_gs(query: str) -> Dict[str, Any]:
    """借助谷歌学术搜索得到论文ID"""
    try:
        result = await search_paper_id_gs(query)
        return result
    except Exception as e:
        return {"error": str(e)}

# 工具函数字典，用于注入到执行环境
ACTION_TOOLS = {
    'search_paper_id': search_paper_id,
    'search_paper_detail': search_paper_detail,
    'search_author_id': search_author_id,
    'search_author_detail': search_author_detail,
    'search_venue_id': search_venue_id,
    'search_venue_detail': search_venue_detail,
    'search_org_id': search_org_id,
    'search_org_detail': search_org_detail,
    'search_paper_id_gs': search_paper_id_gs
}