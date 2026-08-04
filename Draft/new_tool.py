import concurrent.futures
import json
import re
from difflib import SequenceMatcher

import requests

from google_search import google_search_tool
from language import translate_to_english
from llm import llm_client


MAX_TITLES_FROM_SEARCH = 10
AMINER_SIMILARITY_THRESHOLD = 75


def search_paper_title_via_aminer(title):
    params = {"query": title, "needDetails": True, "page": 0, "size": 20, "filters": []}
    try:
        response = requests.post(
            "https://searchtest.aminer.cn/aminer-search/search/publication",
            json=params,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("hitList", [])
    except Exception as exc:
        print(f"AMiner title search failed for '{title[:80]}': {exc}")
        return []


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _dedupe_titles(titles):
    seen = set()
    clean_titles = []
    for title in titles or []:
        if not isinstance(title, str):
            continue
        title = title.strip().strip(".")
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        clean_titles.append(title)
    return clean_titles


def _titles_from_google_results(google_results):
    return _dedupe_titles(
        [result.get("title", "") for result in google_results if isinstance(result, dict)]
    )[:MAX_TITLES_FROM_SEARCH]


def _parse_titles_from_llm(reply):
    if isinstance(reply, dict):
        print(f"LLM title extraction returned an object/error: {reply}")
        return []

    try:
        params = json.loads(_strip_json_fences(reply))
    except (TypeError, json.JSONDecodeError) as exc:
        print(f"Failed to parse LLM title JSON: {exc}. Raw reply: {str(reply)[:200]}")
        return []

    if isinstance(params, dict):
        return _dedupe_titles(params.get("titles", []))[:MAX_TITLES_FROM_SEARCH]
    if isinstance(params, list):
        return _dedupe_titles(params)[:MAX_TITLES_FROM_SEARCH]
    return []


def parse_user_query_to_structured_params(user_query: str):
    """
    Search scholar pages first, then ask the LLM to extract exact paper titles.
    If the LLM or search provider fails, return a conservative title fallback.
    """
    translated_query = translate_to_english(user_query)
    google_results = google_search_tool(translated_query)
    search_context = "\n".join(
        [
            f"网页标题: {result.get('title', '')}\n网页摘要: {result.get('snippet', '')}"
            for result in google_results
        ][:20]
    )

    fallback_titles = _titles_from_google_results(google_results)
    if not google_results:
        print("Scholar search returned no results; falling back to the translated query.")
        return {"titles": [translated_query]}, search_context

    system_prompt = """
You are an academic search parameter extraction assistant.
Given a user query and Scholar/Web search results, extract exact paper titles
that can be used by a paper title search API.
Return pure JSON only, in this format:
{"titles": ["paper title 1", "paper title 2"]}
Do not invent or complete truncated titles; use titles exactly as shown.
"""
    user_prompt = f"""
User query:
{translated_query}

Search context:
{search_context}

Return JSON only.
"""

    reply = llm_client(system_prompt, user_prompt)
    print("LLM title extraction reply:", reply)
    titles = _parse_titles_from_llm(reply)
    if not titles:
        print("Using search result titles as fallback structured params.")
        titles = fallback_titles

    return {"titles": titles}, search_context


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100


def search_paper_id_gs_tool(query: str):
    """
    Search paper IDs through Scholar-derived titles and AMiner title lookup.
    Returns (paper_id_list, search_context).
    """
    params, search_context = parse_user_query_to_structured_params(query)
    titles_from_context = params.get("titles", []) if isinstance(params, dict) else params
    titles_from_context = _dedupe_titles(titles_from_context)[:MAX_TITLES_FROM_SEARCH]

    if not titles_from_context:
        return [], search_context

    idlist = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(search_paper_title_via_aminer, title)
            for title in titles_from_context
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    for result in results:
        if not isinstance(result, list) or not result:
            continue

        paper = result[0]
        if not isinstance(paper, dict):
            continue

        paper_title = paper.get("title", "").strip()
        paper_id = paper.get("id")
        if not paper_title or not paper_id:
            continue

        for context_title in titles_from_context:
            similarity = string_similarity(paper_title.lower(), context_title.lower())
            if similarity >= AMINER_SIMILARITY_THRESHOLD:
                idlist.append(paper_id)
                break

    return list(dict.fromkeys(idlist)), search_context


if __name__ == "__main__":
    print(search_paper_id_gs_tool("关于智慧博物馆设计方向的文献"))
