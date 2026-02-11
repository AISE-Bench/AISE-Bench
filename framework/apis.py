
import asyncio
from typing_extensions import Annotated, Doc
from tools import search_paper_id_tool, search_paper_detail_tool, search_author_id_tool, search_author_detail_tool, \
    search_venue_id_tool, search_venue_detail_tool, search_org_id_tool, search_org_detail_tool

async def search_paper_id(
    titles: Annotated[list, Doc("标题列表")] = None,
    keywords: Annotated[list, Doc("关键词列表，用逗号分隔")] = None,
    years: Annotated[list, Doc("发表年份列表")] = None,
    is_sci: Annotated[bool, Doc("是否为SCI论文")] = None,
    language: Annotated[str, Doc("语言，使用ISO 639-1标准")] = None,
    sort: Annotated[str, Doc("排序方式: year, citation")] = "citation",
    author: Annotated[str, Doc("作者姓名")] = None,
    author_id: Annotated[str, Doc("作者ID")] = None,
    coauthors: Annotated[list, Doc("根据多个共同作者搜论文")] = None,
    org: Annotated[str, Doc("机构或学校名称")] = None,
    org_id: Annotated[str, Doc("机构或学校ID")] = None,
    venues: Annotated[list, Doc("期刊或会议列表，使用英文小写缩写，不需要带年份")] = None,
    venue_ids: Annotated[list, Doc("期刊或会议ID列表")] = None,
    size: Annotated[int, Doc("返回结果数量")] = 10,
):
    
    result = await search_paper_id_tool(
        titles=titles,    
        keywords=keywords,
        years=years,
        is_sci=is_sci,
        language=language,
        sort=sort,
        author=author,
        author_id=author_id,
        coauthors=coauthors,
        org=org,
        org_id=org_id,
        venues=venues,
        venue_ids=venue_ids,
        size=size
    )

    return result

async def search_paper_detail(
    paper_ids: Annotated[list, Doc("论文ID列表")] = None
):

    result = await search_paper_detail_tool(
        paper_ids=paper_ids
    )

    return result

async def search_author_id(
    author: Annotated[str, Doc("学者姓名")] = None,
    orgs: Annotated[list, Doc("机构或学校名称")] = None,
    org_ids: Annotated[list, Doc("机构或学校ID列表")] = None,
    interests: Annotated[list, Doc("研究兴趣列表")] = None,
    nations: Annotated[list, Doc("国籍列表，使用英文缩写，如USA、China")] = None,
    venues: Annotated[list, Doc("期刊或会议列表，使用英文小写缩写，不需要带年份")] = None,
    venue_ids: Annotated[list, Doc("期刊或会议ID列表")] = None,
    sort: Annotated[str, Doc("排序方式: n_citation, n_pubs, h_index")] = "n_citation",
    size: Annotated[int, Doc("返回结果数量")] = 10
):

    result = await search_author_id_tool(
        author=author,
        orgs=orgs,
        org_ids=org_ids,
        interests=interests,
        nations=nations,
        venues=venues,
        venue_ids=venue_ids,
        sort=sort,
        size=size
    )

    return result

async def search_author_detail(
    author_ids: Annotated[list, Doc("学者ID列表")] = None
):

    result = await search_author_detail_tool(
        author_ids=author_ids
    )

    return result

async def search_venue_id(
    venue: Annotated[str, Doc("期刊或会议名称")] = None,
    type: Annotated[str, Doc("搜索模式，fuzzy是模糊搜索，exact是精确搜索")] = "fuzzy",
    keywords: Annotated[list, Doc("关键词列表")] = None,
    category: Annotated[str, Doc("学科分类")] = None,
    source: Annotated[str, Doc("来源")] = None,
    source_tier: Annotated[str, Doc("分区")] = None,
    size: Annotated[int, Doc("返回结果数量")] = 10
):

    result = await search_venue_id_tool(
        venue=venue,
        type=type,
        keywords=keywords,
        category=category,
        source=source,
        source_tier=source_tier,
        size=size
    )

    return result

async def search_venue_detail(
    venue_ids: Annotated[list, Doc("期刊或会议ID列表")] = None
):

    result = await search_venue_detail_tool(
        venue_ids=venue_ids
    )

    return result

async def search_org_id(
    orgs: Annotated[list, Doc("机构或学校名称列表")] = None,
):

    result = await search_org_id_tool(
        orgs=orgs
    )

    return result

async def search_org_detail(
    org_ids: Annotated[list, Doc("机构或学校ID列表")] = None
):

    result = await search_org_detail_tool(
        org_ids=org_ids
    )

    return result

if __name__ == "__main__":
     print(asyncio.run((search_paper_id(keywords=["Deep Learning", "Machine Learning"], language="zh"))))
     print("----------------------------------")
     print(asyncio.run((search_paper_detail(paper_ids=["573696026e3b12023e515eec"]))))
     print("----------------------------------")
     print(asyncio.run((search_author_id(author="Yoshua Bengio", orgs=["Université de Montréal"]))))
     print("----------------------------------")
     print(asyncio.run((search_author_detail(author_ids=["53f4ba75dabfaed83977b7db"]))))
     print("----------------------------------")
     print(asyncio.run((search_venue_id(venue="NeurIPS", type="exact"))))
     print("----------------------------------")
     print(asyncio.run((search_venue_detail(venue_ids=["5ea1e340edb6e7d53c011a4c"]))))
     print("----------------------------------")
     print(asyncio.run((search_org_id(orgs=["Tsinghua University"]))))
     print("----------------------------------")
     print(asyncio.run((search_org_detail(org_ids=["5f71b2881c455f439fe3c860"]))))
