
import asyncio
from new_tool import search_paper_id_gs_tool

async def search_paper_id_gs(query):
    loop = asyncio.get_event_loop()
    # 把同步函数放入线程池运行，避免阻塞
    return await loop.run_in_executor(None, search_paper_id_gs_tool, query)