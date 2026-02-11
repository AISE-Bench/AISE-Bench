import json
import os
import re
import sys
import asyncio
import time
from typing import List, Dict

from llm import llm_client
from config import OPENAI_API_KEY, OPENAI_API_BASE, GPT_MODEL
from caller import TaskExecutor

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
    # 注意：如果参数没有具体值，则用""空字符、或者空list，或其他空的值
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


def _repair_json_trailing_comma(s: str) -> str:
    """修复常见 JSON 语法错误：末尾多余逗号、对象/数组间缺少逗号。"""
    s = re.sub(r",\s*\]", "]", s)
    s = re.sub(r",\s*}", "}", s)
    # 对象或数组之间缺少逗号（如 }  { 或 ]  [），常见于模型输出
    s = re.sub(r"\}\s*\n\s*\{", "},\n{", s)
    s = re.sub(r"\]\s*\n\s*\[", "],\n[", s)
    return s


def _parse_plan_response(plan):
    """将 LLM 返回的规划解析为 list（API 规划数组）。处理空字符串、markdown、Extra data、语法错误等。"""
    if isinstance(plan, list) and plan:
        return plan
    if isinstance(plan, dict):
        return [plan] if plan else []
    s = (plan or "").strip()
    if not s:
        raise ValueError("模型返回的规划为空")
    # 去掉 ```json ... ``` 或 ``` ... ```
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)

    def try_parse(raw: str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # 先尝试直接解析
    obj = try_parse(s)
    if obj is not None:
        return obj if isinstance(obj, list) else [obj]

    # 尝试修复末尾逗号后解析（解决 "Expecting ',' delimiter" 等语法问题，如 id 1960）
    repaired = _repair_json_trailing_comma(s)
    obj = try_parse(repaired)
    if obj is not None:
        return obj if isinstance(obj, list) else [obj]

    # 处理 "Extra data"：前面是合法 JSON 后面有多余内容（如 id 915 返回 "[]" + 真实规划）
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(s):
        s_rest = s[idx:].lstrip()
        if not s_rest:
            break
        try:
            obj, end = decoder.raw_decode(s_rest)
            idx += len(s_rest) - len(s_rest.lstrip()) + end
            # 若解析到空数组 []，可能是模型先输出了 [] 再输出真实规划，继续往后找非空 list
            if isinstance(obj, list) and len(obj) > 0:
                return obj
            if isinstance(obj, dict) and obj:
                return [obj]
        except json.JSONDecodeError:
            break

    # 最后再试一次整段修复后解析
    obj = try_parse(_repair_json_trailing_comma(s))
    if obj is not None:
        return obj if isinstance(obj, list) else [obj]

    preview = (s[:200] + "…") if len(s) > 200 else s
    raise ValueError(f"规划不是合法 JSON，返回内容预览: {preview!r}")


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
            # 使用 GPT（云雾 API）
            plan = llm_client(
                prompt=plan_prompt,
                query=question,
                model=GPT_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE,
            )
            print("【规划】\n", plan)

            plan_obj = _parse_plan_response(plan)
            executor = TaskExecutor(plan_obj)
            result = await executor.run()
            print("【执行结果】\n", result)

            summary_prompt = RESULT_PROMPT.format(api_output=result, question=question)
            summary = llm_client(
                prompt=summary_prompt,
                query=question,
                model=GPT_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE,
            )
            
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
            
            is_concurrency_error = any(keyword in error_msg.lower() for keyword in [
                "并发", "concurrency", "rate limit", "429", "too many requests", 
                "请求过多", "超出限制", "exceeded", "limit"
            ])
            
            if is_concurrency_error:
                print(f"⚠️ 检测到 API 并发错误，跳过重试")
                return {
                    "id": question_id,
                    "question": question,
                    "error": error_msg,
                    "error_type": "concurrency_error"
                }
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ 指数退避等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                return {
                    "id": question_id,
                    "question": question,
                    "error": error_msg
                }


async def process_question_batch(batch: List[Dict], batch_idx: int, start_idx: int, output_dir: str, output_file: str) -> List[Dict]:
    results = []
    
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


def process_questions(input_file: str, output_file: str, batch_size: int = 5):
    output_dir = "glm-batch_outputs"
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

    incorrect_questions = [{"question": r["question"]} for r in all_results if "error" in r]
    incorrect_file = "glm-incorrect.json"
    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_questions, f, ensure_ascii=False, indent=2)
    print(f"❌ 失败的问题（共 {len(incorrect_questions)} 个）已保存至 {incorrect_file}")

    concurrency_errors = [r for r in all_results if r.get("error_type") == "concurrency_error"]
    if concurrency_errors:
        concurrency_file = "glm-concurrency-errors.json"
        concurrency_data = {
            "total_concurrency_errors": len(concurrency_errors),
            "errors": [{"id": r["id"], "question": r["question"], "error": r["error"]} for r in concurrency_errors]
        }
        with open(concurrency_file, 'w', encoding='utf-8') as f:
            json.dump(concurrency_data, f, ensure_ascii=False, indent=2)
        print(f"⚠️ API 并发错误（共 {len(concurrency_errors)} 个）已保存至 {concurrency_file}")
    else:
        print(f"✅ 未检测到 API 并发错误")


if __name__ == "__main__":
    process_questions("reactquestion.json", "gpt-5.2-reactquestion-output.json")
