import json
import os
import sys
import asyncio
import time
import random
import re
import numpy as np
from typing import List, Dict, Any
sentence_bleu = None
SmoothingFunction = None

# 导入OpenAI客户端
from openai import OpenAI

from llm import llm_client
from config import CHATGLM_API_BASE, CHATGLM_API_KEY
from caller import TaskExecutor

TRANSIENT_ERROR_KEYWORDS = (
    "concurrency",
    "rate limit",
    "429",
    "too many requests",
    "request too many",
    "exceeded",
    "limit",
    "saturated",
    "overloaded",
    "upstream",
    "model_not_found",
    "new_api_error",
    "负载",
    "饱和",
    "并发",
    "请求过多",
    "超出限制",
)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _is_transient_api_error(error_msg: str) -> bool:
    lower_msg = error_msg.lower()
    return any(keyword.lower() in lower_msg for keyword in TRANSIENT_ERROR_KEYWORDS)


def _normalize_plan(plan_response: Any) -> List[Dict[str, Any]]:
    if isinstance(plan_response, dict) and "error" in plan_response:
        raise RuntimeError(f"LLM planning failed: {plan_response['error']}")

    if isinstance(plan_response, str):
        cleaned = _strip_json_fences(plan_response)
        try:
            plan = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            preview = cleaned[:500].replace("\n", "\\n")
            raise ValueError(f"Invalid planning JSON: {exc}. Raw response preview: {preview}") from exc
    else:
        plan = plan_response

    if isinstance(plan, dict) and "error" in plan:
        raise RuntimeError(f"LLM planning failed: {plan['error']}")

    if isinstance(plan, dict):
        for key in ("tasks", "plan"):
            if isinstance(plan.get(key), list):
                plan = plan[key]
                break
        else:
            if isinstance(plan.get("name"), str):
                plan = [plan]

    if not isinstance(plan, list):
        raise ValueError(f"Planning response must be a list of tasks, got {type(plan).__name__}")

    normalized_tasks = []
    for index, task in enumerate(plan):
        if not isinstance(task, dict):
            raise ValueError(f"Task #{index + 1} must be a dict, got {type(task).__name__}")

        if not isinstance(task.get("name"), str) or not task["name"].strip():
            raise ValueError(f"Task #{index + 1} is missing a valid name")

        task = task.copy()
        task.setdefault("rely", [])
        task.setdefault("order", index + 1)
        task.setdefault("params", {})

        if task["rely"] is None:
            task["rely"] = []
        if task["params"] is None:
            task["params"] = {}

        if not isinstance(task["rely"], list):
            raise ValueError(f"Task {task['name']} field 'rely' must be a list")
        if not isinstance(task["params"], dict):
            raise ValueError(f"Task {task['name']} field 'params' must be a dict")

        normalized_tasks.append(task)

    return normalized_tasks

# === DRAFT优化流程配置 ===
# OpenAI API配置
OPENAI_API_KEY = ""
OPENAI_API_BASE = "https://xiaoai.plus/v1"

# 创建OpenAI客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

# 优化参数配置
DRAFT_CONFIG = {
    "episodes": 5,          # 最大优化轮数
    "temperature": 0.2,      # 控制生成的随机性
    "top_p": 1.0,           # 控制生成的多样性
    "max_tokens": 2000,      # 最大生成token数
    "model": "gpt-4o"       # 使用的模型
}

# === 工具文档优化函数 ===
# DRAFT优化流程核心函数

def get_response(messages, temperature, top_p, max_tokens, model):
    """
    获取OpenAI模型响应
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用错误: {str(e)}")
        return {"error": str(e)}

def openai_response(messages, temperature, top_p, max_tokens, model):
    """
    处理OpenAI模型响应，解析JSON格式
    """
    try:
        ans = get_response(messages, temperature, top_p, max_tokens, model)
        if isinstance(ans, dict) and "error" in ans:
            return ans

        # 提取JSON部分
        json_match = re.search(r'```json\n(.*?)\n```', ans, re.DOTALL)
        if json_match:
            cleaned_text = json_match.group(1)
        else:
            # 尝试直接使用整个响应
            cleaned_text = ans

        # 清理JSON字符串
        cleaned_text = cleaned_text.strip()
        # 移除可能的BOM字符
        if cleaned_text.startswith('\ufeff'):
            cleaned_text = cleaned_text[1:]

        ans = json.loads(cleaned_text)
        return ans
    except Exception as e:
        print(f"解析JSON失败: {type(e).__name__}: {e}")
        print(f"原始响应: {ans[:200]}..." if 'ans' in locals() else "无响应内容")
        # 返回默认格式
        return {"User Query": "How to use this tool?"}

def openai_embedding(text):
    """
    获取文本嵌入向量
    """
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Caught an exception of type: {type(e)}")
        print("pausing")
        return []

def cosine_similarity(vec1, vec2):
    """
    计算余弦相似度
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 处理空向量情况
    if vec1.size == 0 or vec2.size == 0:
        return 0.0

    # 处理零向量情况
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm1 * norm2)

def change_name(name):
    """
    处理保留关键字
    """
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name

def standardize(string):
    """
    标准化字符串格式
    """
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def compute_similarity_and_bleu(reference_sentence, candidate_sentence):
    """
    计算相似度和BLEU分数
    """
    global sentence_bleu, SmoothingFunction

    reference_sentence_embedding = openai_embedding(reference_sentence)
    candidate_sentence_embedding = openai_embedding(candidate_sentence)
    similarity = cosine_similarity(reference_sentence_embedding, candidate_sentence_embedding)

    reference = [reference_sentence.lower().split()]
    candidate = candidate_sentence.lower().split()

    if sentence_bleu is None or SmoothingFunction is None:
        try:
            from nltk.translate.bleu_score import sentence_bleu as nltk_sentence_bleu
            from nltk.translate.bleu_score import SmoothingFunction as NltkSmoothingFunction

            sentence_bleu = nltk_sentence_bleu
            SmoothingFunction = NltkSmoothingFunction
        except Exception:
            return similarity

    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    delta = (bleu_score + similarity) / 2
    return delta

def optimize_tool_documentation(tool_info, episodes=None):
    """
    使用DRAFT优化流程优化工具文档

    Args:
        tool_info: 工具信息字典，包含category、name、description、required_parameters、optional_parameters
        episodes: 优化轮数，默认使用DRAFT_CONFIG中的配置

    Returns:
        优化后的工具信息字典
    """
    print(f"\n=== 开始优化工具: {tool_info.get('name', 'unknown')} ===")

    # 使用配置参数
    episodes = episodes or DRAFT_CONFIG["episodes"]
    temperature = DRAFT_CONFIG["temperature"]
    top_p = DRAFT_CONFIG["top_p"]
    max_tokens = DRAFT_CONFIG["max_tokens"]
    model = DRAFT_CONFIG["model"]

    # 加载提示模板
    try:
        with open('prompts/Explorer.txt', 'r', encoding='utf-8') as file:
            example_prompt_template = file.read()
            example_prompt, example_prompt_follow = example_prompt_template.split('=========')

        with open('prompts/Analyzer.txt', 'r', encoding='utf-8') as file:
            suggestion_prompt_template = file.read()
            suggestion_prompt, suggestion_prompt_follow = suggestion_prompt_template.split('=========')

        with open('prompts/Rewriter.txt', 'r', encoding='utf-8') as file:
            rewrite_prompt_template = file.read()
            rewrite_prompt, rewrite_prompt_follow = rewrite_prompt_template.split('=========')
    except Exception as e:
        print(f"⚠️ 加载提示模板失败: {e}")
        print("   将使用默认提示模板继续优化")
        # 使用默认提示模板
        example_prompt = "Generate a user query for the following tool: {Tool Description}"
        example_prompt_follow = "Previous queries: {Explored queries}\nSuggestions: {Suggestions}"
        suggestion_prompt = "Analyze the following tool and usage example: {Tool Description}\nUsage example: {usage_example}"
        suggestion_prompt_follow = "History: {History}"
        rewrite_prompt = "Rewrite the tool description based on suggestions: {Tool Description}\nUsage example: {usage_example}\nSuggestions: {Suggestions}"
        rewrite_prompt_follow = "History: {History}"

    # 提取工具信息
    tool_category = tool_info.get('category', 'general')
    api_name = tool_info.get('name', 'unknown_tool')
    last_tool_description = tool_info.get('description', 'No description provided')
    required_parameters = tool_info.get('required_parameters', {})
    optional_parameters = tool_info.get('optional_parameters', {})

    print(f"原始描述: {last_tool_description}")

    # 初始化优化历史
    explored_queries = []
    explored_queries_embeddings = []
    explored_examples = []
    suggestions = []
    rewrite_description_history = []
    rewrite_description_history.append(last_tool_description)
    rewrite_agent_history = []
    suggestion_from_rewrite_agent = ''

    # 多轮优化
    for episode in range(episodes):
        print(f"\n--- 第 {episode+1}/{episodes} 轮优化 ---")

        tool_info_current = {
            'category': tool_category,
            'name': api_name,
            'description': rewrite_description_history[-1],
            'required_parameters': required_parameters,
            'optional_parameters': optional_parameters
        }
        tool_description = str(tool_info_current)

        # Explorer阶段：生成用户查询
        print("   Explorer阶段: 生成用户查询")
        explore_prompt = example_prompt.replace('{Tool Description}', tool_description)

        if len(explored_queries) > 0:
            explore_prompt_follow = example_prompt_follow.replace('{Explored queries}', str(explored_queries))
            explore_prompt_follow = explore_prompt_follow.replace('{Suggestions}', suggestion_from_rewrite_agent)
            explore_prompt = explore_prompt + "\n" + explore_prompt_follow

            # 尝试生成不同的查询
            for t in range(3):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": explore_prompt}
                ]
                example_ans = openai_response(messages, temperature, top_p, max_tokens, model)

                if isinstance(example_ans, dict) and 'User Query' in example_ans:
                    user_query = example_ans['User Query']
                    cur_embedding = openai_embedding(user_query)

                    # 检查查询是否与之前的相似
                    if cur_embedding:
                        similarity = [cosine_similarity(emb, cur_embedding) for emb in explored_queries_embeddings if emb]
                        if not similarity or all(sim < 0.9 for sim in similarity):
                            break
                        else:
                            print("   ⚠️ 生成的查询与之前相似，重新生成")
                            explore_prompt += f"\nPlease generate a different query from the previous ones."
                    else:
                        break
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": explore_prompt}
            ]
            example_ans = openai_response(messages, temperature, top_p, max_tokens, model)

        # 处理Explorer结果
        if isinstance(example_ans, dict) and 'User Query' in example_ans:
            user_query = example_ans['User Query']
            explored_queries.append(user_query)
            embedding = openai_embedding(user_query)
            if embedding:
                explored_queries_embeddings.append(embedding)
            print(f"   生成的查询: {user_query}")
        else:
            print("   ⚠️ Explorer阶段未生成有效的查询，使用默认查询")
            user_query = f"How to use {api_name} tool?"
            explored_queries.append(user_query)

        # Analyzer阶段：分析使用示例并提供建议
        print("   Analyzer阶段: 分析使用示例并提供建议")
        suggestion_prompt_temp = suggestion_prompt.replace('{Tool Description}', tool_description)
        suggestion_prompt_temp = suggestion_prompt_temp.replace('{usage_example}', str(example_ans))

        if len(rewrite_description_history) > 1:
            suggestion_prompt_follow_temp = suggestion_prompt_follow.replace('{History}', str(rewrite_description_history))
            suggestion_prompt_temp += "\n" + suggestion_prompt_follow_temp

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": suggestion_prompt_temp}
        ]
        suggestion_ans = openai_response(messages, temperature, top_p, max_tokens, model)
        suggestions.append(suggestion_ans)

        if isinstance(suggestion_ans, dict) and 'Suggestions for tool description' in suggestion_ans:
            print("   生成了改进建议")
        else:
            print("   ⚠️ Analyzer阶段未生成有效的建议，使用默认建议")

        # Rewriter阶段：根据建议优化工具文档
        print("   Rewriter阶段: 根据建议优化工具文档")
        rewrite_prompt_temp = rewrite_prompt.replace('{Tool Description}', tool_description)
        rewrite_prompt_temp = rewrite_prompt_temp.replace('{usage_example}', str(example_ans))

        # 添加建议
        if isinstance(suggestion_ans, dict) and 'Suggestions for tool description' in suggestion_ans:
            rewrite_prompt_temp = rewrite_prompt_temp.replace('{Suggestions}', suggestion_ans['Suggestions for tool description'])
        else:
            rewrite_prompt_temp = rewrite_prompt_temp.replace('{Suggestions}', 'Improve clarity and completeness')

        rewrite_prompt_temp = rewrite_prompt_temp.replace('{tool_description}', tool_info_current['description'])

        if len(rewrite_description_history) > 1:
            rewrite_prompt_follow_temp = rewrite_prompt_follow.replace('{History}', str(rewrite_description_history))
            rewrite_prompt_temp += "\n" + rewrite_prompt_follow_temp

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": rewrite_prompt_temp}
        ]
        rewrtite_ans = openai_response(messages, temperature, top_p, max_tokens, model)

        # 处理Rewriter结果
        if isinstance(rewrtite_ans, dict) and 'Rewritten description' in rewrtite_ans:
            rewritten_desc = rewrtite_ans['Rewritten description']
            rewrite_description_history.append(rewritten_desc)
            last_tool_description = rewritten_desc
            print(f"   优化后描述: {rewritten_desc[:100]}...")
        else:
            print("   ⚠️ Rewriter阶段未生成有效的描述，使用原始描述")
            rewrite_description_history.append(last_tool_description)

        # 提取探索建议
        if isinstance(rewrtite_ans, dict) and 'Suggestions for exploring' in rewrtite_ans:
            suggestion_from_rewrite_agent = str(rewrtite_ans['Suggestions for exploring'])

        # 保存优化历史
        rewrite_tool = {'tool_description': rewrtite_ans}
        rewrite_agent_history.append(rewrite_tool)

        # 检查优化是否收敛
        if len(rewrite_description_history) > 1:
            reference_sentence = rewrite_description_history[-2]
            candidate_sentence = rewrite_description_history[-1]
            delta = compute_similarity_and_bleu(reference_sentence, candidate_sentence)
            print(f"   相似度: {delta:.3f}")

            if delta > 0.75:
                print("   ✅ 优化已收敛，提前结束")
                break

    # 更新工具描述
    tool_info['description'] = rewrite_description_history[-1]
    tool_info['optimization_history'] = rewrite_description_history
    tool_info['optimization_rounds'] = len(rewrite_description_history) - 1

    print(f"\n=== 优化完成 ===")
    print(f"最终描述: {tool_info['description']}")
    print(f"优化轮数: {tool_info['optimization_rounds']}")

    return tool_info


def batch_optimize_tools(tools_list, episodes=None):
    """
    批量优化工具文档

    Args:
        tools_list: 工具信息列表
        episodes: 每工具的优化轮数

    Returns:
        优化后的工具信息列表
    """
    print(f"=== 开始批量优化工具文档 ===")
    print(f"共 {len(tools_list)} 个工具待优化")

    optimized_tools = []
    total_optimized = 0

    for i, tool in enumerate(tools_list):
        print(f"\n--- 优化工具 {i+1}/{len(tools_list)} ---")
        try:
            optimized_tool = optimize_tool_documentation(tool, episodes)
            optimized_tools.append(optimized_tool)
            total_optimized += 1
            print(f"✅ 工具 {tool.get('name', 'unknown')} 优化成功")
        except Exception as e:
            print(f"❌ 工具 {tool.get('name', 'unknown')} 优化失败: {e}")
            optimized_tools.append(tool)  # 保留原始工具信息

        # 每优化3个工具后休息一下，避免API rate limit
        if (i + 1) % 3 == 0 and i + 1 < len(tools_list):
            print("\n⏳ 休息2秒，避免API限流...")
            time.sleep(2)

    print(f"\n=== 批量优化完成 ===")
    print(f"成功优化: {total_optimized}/{len(tools_list)} 个工具")

    # 保存优化结果
    output_file = "batch_optimized_tools.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(optimized_tools, f, ensure_ascii=False, indent=2)
    print(f"优化结果已保存至: {output_file}")

    return optimized_tools

RESULT_PROMPT = """
You are a world-class academic expert. You will answer questions from any academic field.

Instructions:

Your answer must be in the same language as the question.

The answer must be concise and scholarly, no longer than 300 words.

Analyze the user's question to determine the underlying intent(s). For each distinct intent, provide a separate, clearly labeled point in your answer.

Use the following knowledge base information (api_output) as your primary source. Only use data from the knowledge base that is directly relevant to the question and its intent(s). Do not use unrelated information, even if present in the knowledge base.

If your answer contains any information drawn from api_output (even partially), you must include a citation [e.g., [1], [2]] for that content. If the information is insufficient, you may supplement with your own knowledge, but always prioritize and cite api_output when used.
In the "reference" field, each citation label (e.g., [1], [2]) should map to a full citation (with the appropriate link, as specified above).
In the "reference" field, only use links or IDs that are present in the knowledge base (api_output); do not invent, infer, or fabricate any links or IDs.
In the "reference" field, citations must be numbered sequentially as [1], [2], [3], [4], etc., starting from [1] and increasing by 1 for each new citation. No duplicate citations are allowed in the "reference" field; each cited item should appear only once in the reference list.

If a citation is an ID, determine its type (paper, author, venue, or org) based on the knowledge base (api_output), and generate the corresponding full URL as follows:
- For papers: https://www.aminer.cn/pub/paper_id
- For scholars: https://www.aminer.cn/profile/author_id
- For journals/conferences: https://www.aminer.cn/open/journal/detail/venue_id
- For institutions: https://www.aminer.cn/institution/org_id
Only generate such URLs if the ID exists in the knowledge base (api_output); do not fabricate or infer IDs.

Output only a JSON object with two fields:
- "answer": your academic answer, written in the same language as the question (max 300 words).
- "reference": a dictionary mapping each citation label to its bibliographic citation (as a webpage or ID from the knowledge base).

Example output:
{{
"answer": "Your detailed answer here, with inline citations [1][2] ...",
"reference": {{"[1]": "https://xxx", "[2]": "ID"}}
}}

Here is the knowledge base output (api_output):
{api_output}

Here is the question:
{question}
"""

PLAN_PROMPT = """你是一个规划专家，你将根据用户的问题，选择一个或多个合适的API，使用API检索到相关信息，从而可以准确回答用户的问题。【thinking过程不要超过十句话】
   你生成的格式示例如下：
    [
        {
            "name": "search_author_id",
            "rely": [],
            "order": 1,
            "params": {"interest": ["Optical communication"]}
        },
        {
            "name": "search_author_detail",
            "rely": ["search_author_id"],
            "order": 2,
            "params": {"ids": []}
        },
    ]
    # 说明：其中，name指的是API的名称，rely是输入参数的来源API名称，只能为api名称，order指的是执行的顺序，如1，2，3；params是API的参数
    # 注意：你不需要生成额外的任何解释，只需要生成上面说明生成json内容即可！（不需要出现```、json等字眼）
    # 注意：如果参数没有具体值，则为""空字符、或者空list，或其他空的值
    # 注意：true和false必须首字母小写！
    # 注意：没有指定排序方式的情况下，都按照citation排序！
    # 注意：如果有多个相同名字的api，则在name中加编号，如search_paper_id(1)、search_paper_id(2)等
    # 注意：如果问题没有指定为中文，关键词一律使用英文单词！

    # 注意：你不要局限于上面的规划顺序，你可以按照你的知识给出任何的api调用顺序！

    # 可用的API如下：
【search_paper_id】
    "search_paper_id": {
        "description": "根据条件搜索论文ID",
        "parameters": {
            "titles": ["论文标题"],
            "keywords": ["关键词"],
            "years": {
                "type": "array",
                "description": "发表年份列表, 年份用整数表示"
            },
            "is_sci": {
                "type": "boolean",
                "description": "是否为SCI论文"
            },
            "language": "论文语言(str), 使用ISO 639-1标准如'en', 'zh'",
            "sort": "排序方式(str)，只能选择: year, citation",
            "author": "作者姓名(str)",
            "author_id": "作者ID(str)",
            "coauthors": {
                "type": "array",
                "description": "共同作者姓名列表，指的是不包括author参数作者的其他作者"
            },
            "org": "机构或学校名称(str)",
            "org_id": "机构或学校ID(str)",
            "venues": {
                "type": "array",
                "description": "期刊或会议列表，使用英文小写缩写，不需要带年份"
            },
            "venue_ids": ["期刊或会议ID"],
            "size": {
                "type": "integer",
                "description": "返回结果数量，必须小于100"
            }
        },
        "response": {
            "description": "返回结果列表, 每个元素为一个字典(代表一篇论文)",
            "data": [
                {
                    "paper_id": "论文ID(str)",
                    "title": "论文标题(str)"
                }
            ]
        }
    },

---
【search_paper_detail】
    "search_paper_detail":{
        "description": "根据论文ID列表，获取论文详细信息",
        "parameters": {
            "paper_ids": ["论文ID"]
        },
        "response":{
            "description": "返回结果列表, 每个元素为一个字典(代表一篇论文)",
            "data": [
                {
                    "paper_id": "论文ID(str)",
                    "title": "论文标题(str)",
                    "abstract": "论文摘要(str)",
                    "year": "发表年份(float)",
                    "citation": "被引用次数(float)",
                    "keywords": ["关键词"],
                    "authors": [
                        {
                            "author": "作者姓名(str)",
                            "author_id": "作者ID(str)",
                            "org": "作者机构(str)",
                            "org_id": "作者机构ID(str)",
                            "email":"作者邮箱(str)"
                        }
                    ],
                    "org": "发表机构(str)",
                    "org_id": "发表机构ID(str)",
                    "venue": "发表的期刊或会议(str)"
                }
            ]
        }
    },

---
【search_author_id】
    {
        "type": "function",
        "function":{
            "name": "search_author_id",
            "description": "根据姓名、机构、兴趣、国家等条件搜索学者",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "学者姓名",
                    },
                    "org": {
                        "type": "string",
                        "description": "学者所在机构",
                    },
                    "size": {
                        "type": "integer",
                        "description": "返回的学者数量，最大为1000",
                    },
                    "interest": {
                        "type": "list",
                        "description": "学者兴趣，格式为[str,str,...]"
                    },
                    "nation": {
                        "type": "list",
                        "description": "学者所在国家，格式为[str,str,...]
                    },
                    "order": {
                        "type": "string",
                        "description": "排序字段名 n_citation, n_pubs, h_index"
                    },
                    "asc": {
                        "type": "boolean",
                        "description": "true 升序 false 降序"
                    }
                },
                "required": []
            }
        }
    },
---
【search_author_detail】根据学者ID列表获取学者详情
输入：ids：[str,str, ...]  #id列表

---
【search_venue_id】根据期刊名称搜索期刊ID、期刊标准名称
输入：
    name：str  #期刊名称
    category：str  #学科分类名称
    category_source:string # 使用数字形式字符串，不同的数字的含义如下： 0: "SJR", 1: "WOS", 2: "GB09", 3: "CCF", 4: "CSCD", 5: "CCJ", 6: "ARXIV", 7: "CJCR", 8: "JCR", 9: "SCI"
    quartile：str  #期刊分区搜索 如"1区", "A", "Q1"
    keywords：list  #期刊关键词列表
    size：number  #返回的期刊数量
输出：期刊id

---
【search_venue_detail】根据期刊ID获取期刊详情
输入：
    ids：[str,str, ...]  #id列表
输出：
alias	array	别名
category_id	string	学科领域id
classes	array	数据源
id	string	id
issn	string	ISSN
lower_alias	array	别名（小写）
name	string	姓名
name_en	string	机构英文名称
name_zh	string	中文名
num	float	优先权号
quartile	string	分区
source_quartiles	array	源分区
total	float	总数
type	string	分类体系
url	string	来源url

---
【search_org_id】根据名称关键词搜索机构ID、名称
输入：
    orgs：[str,str,...]  #机构名称列表
输出：
    机构id列表

---
【search_org_detail】通过机构ID获取机构详情
输入：
    ids：[str,str, ...]  #id列表
输出：
acronyms	array
aliases	array	机构别名
coordinate	array
details	array	机构详情
error	array
established	int	成立时间
external_ids	array
geographic_id	string	地理id
id	string	机构id
image	string	图片
introduction	string	简介
language	string	语言
latitude	float	纬度
longitude	float	经度
name	string	机构名称
name_en	string	机构英文名称
name_zh	string	中文名
relationships	array
src	string	数据源
total	int	返回数据条数
type	string	机构类型

【search_paper_id_gs】借助谷歌学术搜索得到论文id
输入：
    query: "str" # 用户问题
"""

# ======== 主处理流程 ========
# def process_questions(input_file: str, output_file: str):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         questions = json.load(f)

#     results = []

#     for idx, item in enumerate(questions):
#         question = item.get("question", "").strip()
#         if not question:
#             continue

#         print(f"\n===== 处理问题 {idx+1} =====")
#         print("问题内容：", question)



#         # 第一步：调用大模型生成任务规划
#         plan_prompt = PLAN_PROMPT + "用户问题如下：" + question
#         plan = llm_client(prompt=plan_prompt, query=question, model="deepseek-reasoner", api_key="", api_base="https://api.deepseek.com/v1")
#         print("【规划】\n", plan)

#         # 第二步：执行规划
#         executor = TaskExecutor(json.loads(plan))
#         result = asyncio.run(executor.run())

#         print("【执行结果】\n", result)

#         # 第三步：调用大模型进行总结
#         summary_prompt = RESULT_PROMPT.format(api_output=result, question=question)
#         summary = llm_client(prompt=summary_prompt, query=question, model="public-deepseek-v3-0324", api_key="", api_base="https://api.zhipuai-infra.cn/")
#         print("【总结】\n", summary)

#         # 保存结果
#         results.append({
#             "question": question,
#             "plan": plan,
#             "execution_result": result,
#             "summary": summary
#         })

#     # 写入输出文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ 所有问题处理完毕，结果已保存至 {output_file}")

# # ======== 执行入口 ========
# if __name__ == "__main__":
#     process_questions("query.json", "output_9_12.json")

# 单个问题的处理逻辑（带重试）
async def process_single_question(item: Dict, idx: int) -> Dict:
    question = item.get("question", "").strip()
    question_id = item.get("id", idx + 1)
    if not question:
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"\n===== 处理问题 {idx+1}（尝试 {attempt+1}/{max_retries}）=====")
            print("问题内容：", question)

            plan_prompt = PLAN_PROMPT + "用户问题如下：" + question
            plan = llm_client(prompt=plan_prompt, query=question)
            print("【规划】\n", plan)

            tasks = _normalize_plan(plan)
            executor = TaskExecutor(tasks)
            result = await executor.run()
            print("【执行结果】\n", result)

            summary_prompt = RESULT_PROMPT.format(api_output=result, question=question)
            summary = llm_client(prompt=summary_prompt, query=question)

            print("【总结】\n", summary)

            return {
                "id": question_id,
                "question": question,
                "plan": plan,
                "execution_result": result,
                "summary": summary
            }

        except Exception as e:
            error_msg = str(e)
            print(f"❌ 问题 {idx+1} 处理失败（尝试 {attempt+1}/{max_retries}）：{error_msg}")

            is_transient_api_error = _is_transient_api_error(error_msg)

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ 指数退避等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                failed_result = {
                    "id": question_id,
                    "question": question,
                    "error": error_msg
                }
                if is_transient_api_error:
                    failed_result["error_type"] = "concurrency_error"
                return failed_result


# 分组执行逻辑
async def process_question_batch(batch: List[Dict], batch_idx: int, start_idx: int, output_dir: str, output_file: str) -> List[Dict]:
    results = []

    # 先读取已有的结果
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []

    for i, item in enumerate(batch):
        result = await process_single_question(item, start_idx + i)
        if result is not None:
            results.append(result)

            # 每次处理完一个问题后就写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"✅ 问题 {start_idx + i + 1} 处理完成并已保存到 {output_file}")

        if i < len(batch) - 1:
            print(f"⏳ 等待 5 秒后处理下一个问题...")
            time.sleep(5)

    batch_output_file = os.path.join(output_dir, f"output_batch_{batch_idx}.json")
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(results[-(len(batch)):], f, ensure_ascii=False, indent=2)
    print(f"✅ 批次 {batch_idx} 完成，结果保存至 {batch_output_file}")

    return results

# 主函数
def process_questions(input_file: str, output_file: str, batch_size: int = 5):
    output_dir = "0131-draft-batch_outputs"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    all_results = []
    total_batches = (len(questions) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(questions))
        batch = questions[start:end]
        print(f"\n===== 开始处理第 {batch_idx} 批，共 {len(batch)} 个问题 =====")

        batch_results = asyncio.run(process_question_batch(batch, batch_idx, start_idx=start, output_dir=output_dir, output_file=output_file))
        all_results.extend(batch_results)

        if batch_idx < total_batches - 1:
            print(f"⏳ 等待 10 秒后处理下一批次...")
            time.sleep(10)

    print(f"\n🎉 所有批次处理完成，最终结果已保存至 {output_file}")

    # 保存所有失败的问题（包括并发错误）
    incorrect_questions = [{"question": r["question"]} for r in all_results if "error" in r]
    incorrect_file = "0131-draft-incorrect.json"
    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_questions, f, ensure_ascii=False, indent=2)
    print(f"❌ 失败的问题（共 {len(incorrect_questions)} 个）已保存至 {incorrect_file}")

    # 单独保存 API 并发错误的问题
    concurrency_errors = [r for r in all_results if r.get("error_type") == "concurrency_error"]
    if concurrency_errors:
        concurrency_file = "0131-draft-concurrency-errors.json"
        concurrency_data = {
            "total_concurrency_errors": len(concurrency_errors),
            "errors": [{"id": r["id"], "question": r["question"], "error": r["error"]} for r in concurrency_errors]
        }
        with open(concurrency_file, 'w', encoding='utf-8') as f:
            json.dump(concurrency_data, f, ensure_ascii=False, indent=2)
        print(f"⚠️ API 并发错误（共 {len(concurrency_errors)} 个）已保存至 {concurrency_file}")
    else:
        print(f"✅ 未检测到 API 并发错误")

# 使用DRAFT优化流程的示例函数
def optimize_tools_example():
    """
    使用DRAFT优化流程优化工具文档的示例
    """
    print("=== 开始使用DRAFT优化工具文档 ===")

    # 定义示例工具列表（包含所有8个AMiner API）
    tools_to_optimize = [
        {
            "category": "academic",
            "name": "search_paper_id",
            "description": "根据关键词搜索论文ID",
            "required_parameters": {
                "keywords": "论文关键词列表"
            },
            "optional_parameters": {
                "years": "发表年份列表",
                "size": "返回结果数量",
                "coauthors": "共同作者列表",
                "is_sci": "是否为SCI论文",
                "language": "论文语言",
                "sort": "排序方式",
                "author": "作者姓名",
                "author_id": "作者ID",
                "org": "机构名称",
                "org_id": "机构ID",
                "venues": "期刊或会议列表",
                "venue_ids": "期刊或会议ID列表"
            }
        },
        {
            "category": "academic",
            "name": "search_paper_detail",
            "description": "根据论文ID获取论文详情",
            "required_parameters": {
                "paper_ids": "论文ID列表"
            },
            "optional_parameters": {}
        },
        {
            "category": "academic",
            "name": "search_venue_id",
            "description": "根据名称搜索期刊或会议ID",
            "required_parameters": {
                "name": "期刊或会议名称列表"
            },
            "optional_parameters": {
                "size": "返回结果数量"
            }
        },
        {
            "category": "academic",
            "name": "search_venue_detail",
            "description": "根据期刊或会议ID获取详情",
            "required_parameters": {
                "venue_ids": "期刊或会议ID列表"
            },
            "optional_parameters": {}
        },
        {
            "category": "academic",
            "name": "search_author_id",
            "description": "根据姓名搜索作者ID",
            "required_parameters": {
                "name": "作者姓名"
            },
            "optional_parameters": {
                "org": "作者机构",
                "size": "返回结果数量"
            }
        },
        {
            "category": "academic",
            "name": "search_author_detail",
            "description": "根据作者ID获取作者详情",
            "required_parameters": {
                "author_ids": "作者ID列表"
            },
            "optional_parameters": {}
        },
        {
            "category": "academic",
            "name": "search_org_id",
            "description": "根据名称搜索机构ID",
            "required_parameters": {
                "name": "机构名称"
            },
            "optional_parameters": {
                "size": "返回结果数量"
            }
        },
        {
            "category": "academic",
            "name": "search_org_detail",
            "description": "根据机构ID获取机构详情",
            "required_parameters": {
                "org_ids": "机构ID列表"
            },
            "optional_parameters": {}
        },
        {
            "category": "academic",
            "name": "search_paper_id_gs",
            "description": "使用Google Scholar搜索论文ID",
            "required_parameters": {
                "query": "用户查询字符串"
            },
            "optional_parameters": {}
        }
    ]

    # 使用批量优化函数
    optimized_tools = batch_optimize_tools(tools_to_optimize)

    print("\n=== 优化完成 ===")
    print(f"优化结果已保存至 batch_optimized_tools.json")
    print(f"共优化了 {len(optimized_tools)} 个工具")
    return optimized_tools

# 主执行函数
def main():
    """
    主执行函数，可选择运行问题处理或工具优化
    """
    import argparse

    parser = argparse.ArgumentParser(description="选择执行模式")
    parser.add_argument("--mode", choices=["process", "optimize"], default="process",
                        help="执行模式: process (处理问题) 或 optimize (优化工具文档)")
    parser.add_argument("--input", default="question150.json",
                        help="输入文件路径 (默认: query.json)")
    parser.add_argument("--output", default="0608-draft.json",
                        help="输出文件路径 (默认: 0131-draft.json)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="批处理大小 (默认: 5)")
    parser.add_argument("--episodes", type=int,
                        help="优化轮数 (默认: 使用配置中的值)")

    args = parser.parse_args()

    if args.mode == "process":
        # 处理问题
        print(f"=== 开始处理问题 ===")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"批处理大小: {args.batch_size}")
        process_questions(args.input, args.output, args.batch_size)
    else:
        # 优化工具文档
        print(f"=== 开始优化工具文档 ===")
        if args.episodes:
            print(f"优化轮数: {args.episodes}")
        optimize_tools_example()

# ======== 执行入口 ========
if __name__ == "__main__":
    main()
