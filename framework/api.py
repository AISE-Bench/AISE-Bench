import asyncio
from typing import Any, Dict
import os
import httpx
import re
import sys
from llm import llm_client

from config import API_TOKEN, API_CONFIG


import logging
import sys
import traceback

import streamlit as st



class AMinerAPI:
    def __init__(self):
        self.base_url = "https://datacenter.aminer.cn"
        self.token = API_TOKEN
        self.api_config = API_CONFIG
        self.headers = {
            "Content-Type": "application/json;charset=utf-8",
            "Authorization": f"Bearer {self.token}" 
        }

    async def call_api(self, api_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        异步调用指定的 API
        
        Args:
            endpoint (str): API 的相对路径
            payload (Dict[str, Any]): 请求的 JSON 数据
        
        Returns:
            Dict[str, Any]: API 返回的 JSON 数据
        """

        api_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', api_name.strip()).group(0)

        if api_name not in self.api_config:
            return {"error": "Invalid API name", "details": f"Unknown API: {api_name}"}
        
           

        # 新的API调用

        api_info = self.api_config[api_name]
        endpoint = api_info["endpoint"]
        method = api_info["method"]

        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}{endpoint}"

                if method == "GET":
                    response = await client.get(
                        url=url,
                        headers=self.headers,
                        params=payload,
                        timeout=10.0  # 设置超时时间
                    )
                    response.raise_for_status()
                    return response.json()  
                
                elif method == "POST":
                    response = await client.post(
                        url=f"{self.base_url}{endpoint}",
                        headers=self.headers,
                        json=payload,
                        timeout=10.0  
                    )
                    response.raise_for_status()
                    return response.json()  
                
                else:
                    return {"error": "Unsupported HTTP method", "details": f"Method {method} is not supported"}

            except httpx.HTTPStatusError as e:
                return {"error": f"HTTP Error: {e.response.status_code}", "details": e.response.text}
            except Exception as e:
                return {"error": "Request failed", "details": str(e)}

# 测试调用 AMiner API
async def main():
    aminer_api = AMinerAPI()

    payload = {
        "use_topic": True,
        'topic_high': "[[\"nlp\"]]",
        # "title":["Attention is all you need"],
        # 'venue_ids': ['5ea1b22bedb6e7d53c00c41b'],
        "size": 10,
        "offset": 0
    }

    payload =  {
    # "org_id": 
    #     ""
    # ,
    # "interest": [
    #     "Ubiquitous Operating System"
    # ],
    "author":"dequan wang",
    "size": 1
}

    # payload = {
    #     "title":["Attention is all you need"]
    # }

    response = await aminer_api.call_api(api_name="search_author_id(1)", payload=payload)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
