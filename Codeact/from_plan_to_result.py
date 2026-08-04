import json
import os
import sys
import asyncio
import time
import re
from typing import List, Dict

from llm import llm_client
from config import CHATGLM_API_BASE, CHATGLM_API_KEY
from caller import TaskExecutor


def parse_llm_json(content, expected_type=None, label="LLM response"):
    if isinstance(content, (dict, list)):
        parsed = content
    else:
        cleaned = content.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()

        if expected_type is list:
            start = cleaned.find("[")
            end = cleaned.rfind("]")
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        parsed = json.loads(cleaned)

    if isinstance(parsed, dict) and "error" in parsed:
        raise RuntimeError(f"{label} error: {parsed['error']}")

    if expected_type is not None and not isinstance(parsed, expected_type):
        expected_name = expected_type.__name__
        actual_name = type(parsed).__name__
        raise ValueError(f"{label} must be {expected_name}, got {actual_name}")

    return parsed


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

IMPORTANT CODEACT RULES:
1. STRICTLY BASED ON API OUTPUT: Your answer must be entirely based on the information provided in api_output. Do not use any external knowledge or assumptions.
2. NO FABRICATION: Do not invent, infer, or fabricate any information, IDs, links, or citations that are not explicitly present in api_output.
3. INSUFFICIENT INFORMATION: If api_output does not contain enough information to answer the question, clearly state "暂无相关信息" for the specific points that cannot be addressed.
4. PRECISE CITATIONS: Only cite information that is directly present in api_output, and ensure each citation corresponds to a valid entry in the reference field.
5. ACCURATE JSON FORMAT: Ensure your output is a valid JSON object with no syntax errors. The JSON must contain exactly two fields: "answer" and "reference".

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

# CodeAct 可执行性规则：
# 1. 必须包含的字段：每个任务必须包含 name、rely、order、params 四个字段，无空值
# 2. 参数必填项：
#    - search_paper_id：use_topic 参数必须要传（值为 true 或 false）
# 3. 数据类型规范：
#    - 布尔值：true/false 必须首字母小写
#    - 列表类型：必须使用 [] 格式，如 ["关键词1", "关键词2"]
#    - 字符串类型：必须使用 "" 包裹
# 4. 依赖规则：
#    - rely 字段必须是一个列表，包含当前任务依赖的所有前置任务名称
#    - 依赖传递：如 search_paper_id 需要使用 search_paper_detail 的 coauthors 结果，则必须在 rely 中指定 ["search_paper_detail"]
# 5. 执行顺序：
#    - order 字段必须是整数，按升序执行
#    - 无依赖任务优先执行
# 6. 参数格式：
#    - 参数格式必须与 API 定义完全匹配
#    - 对于 coauthors 参数，当依赖 search_paper_detail 时，会自动从其结果中提取作者信息
# 7. 命名规范：
#    - 如果有多个相同名字的 API，则在 name 中加编号，如 search_paper_id(1)、search_paper_id(2) 等
# 8. 语言规范：
#    - 如果问题没有指定为中文，关键词一律使用英文单词
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

            # 规划后校验逻辑
            plan_json = parse_llm_json(plan, expected_type=list, label="plan")
            if not isinstance(plan_json, list):
                print("❌ 规划格式错误：规划必须是一个列表")
                raise ValueError("规划格式错误：规划必须是一个列表")

            # 检查每个任务是否包含必要字段
            task_names = []
            for task in plan_json:
                if not isinstance(task, dict):
                    print("❌ 规划格式错误：每个任务必须是一个字典")
                    raise ValueError("规划格式错误：每个任务必须是一个字典")

                # 检查必要字段
                required_fields = ["name", "rely", "order", "params"]
                for field in required_fields:
                    if field not in task:
                        print(f"❌ 规划格式错误：任务缺少必要字段 {field}")
                        raise ValueError(f"规划格式错误：任务缺少必要字段 {field}")

                # 检查依赖格式
                if not isinstance(task["rely"], list):
                    print("❌ 规划格式错误：rely 字段必须是一个列表")
                    raise ValueError("规划格式错误：rely 字段必须是一个列表")

                # 检查执行顺序格式
                if not isinstance(task["order"], int):
                    print("❌ 规划格式错误：order 字段必须是一个整数")
                    raise ValueError("规划格式错误：order 字段必须是一个整数")

                # 检查参数格式
                if not isinstance(task["params"], dict):
                    print("❌ 规划格式错误：params 字段必须是一个字典")
                    raise ValueError("规划格式错误：params 字段必须是一个字典")

                # 检查 search_paper_id 的必填参数
                task_name_base = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', task["name"]).group(0)
                if task_name_base == "search_paper_id":
                    # 如果缺少 use_topic 参数，添加默认值 false
                    if "use_topic" not in task["params"]:
                        print("⚠️  规划缺少 use_topic 参数，添加默认值 false")
                        task["params"]["use_topic"] = False

                    # 检查 use_topic 的值类型
                    use_topic = task["params"]["use_topic"]
                    if not isinstance(use_topic, bool):
                        print("❌ 规划格式错误：use_topic 参数必须是布尔值")
                        raise ValueError("规划格式错误：use_topic 参数必须是布尔值")

                task_names.append(task["name"])

            # 检查依赖是否存在
            for task in plan_json:
                for dep in task["rely"]:
                    if dep not in task_names:
                        print(f"❌ 规划格式错误：依赖任务 {dep} 不存在")
                        raise ValueError(f"规划格式错误：依赖任务 {dep} 不存在")

            # 使用更新后的 plan_json 执行任务
            executor = TaskExecutor(plan_json)
            result = await executor.run()
            print("【执行结果】\n", result)

            summary_prompt = RESULT_PROMPT.format(api_output=result, question=question)
            # 优化提示词，确保模型输出JSON格式
            summary_prompt = summary_prompt + "\n\n重要：请确保您的输出是一个有效的JSON对象，只包含answer和reference两个字段，不要添加任何额外的文本或解释。"
            summary = llm_client(prompt=summary_prompt, query=question)

            print("【总结】\n", summary)

            # 总结后校验逻辑
            try:
                # 尝试解析总结为 JSON
                if isinstance(summary, dict):
                    summary_json = summary
                else:
                    # 尝试清理可能的 Markdown 代码块标记
                    cleaned_summary = summary.strip()
                    if cleaned_summary.startswith('```json') and cleaned_summary.endswith('```'):
                        cleaned_summary = cleaned_summary[7:-3].strip()
                    elif cleaned_summary.startswith('```') and cleaned_summary.endswith('```'):
                        cleaned_summary = cleaned_summary[3:-3].strip()
                    # 清理可能的前后缀文本
                    cleaned_summary = re.sub(r'^.*?\{', '{', cleaned_summary)
                    cleaned_summary = re.sub(r'\}.*?$', '}', cleaned_summary)

                    # 尝试解析清理后的内容
                    summary_json = json.loads(cleaned_summary)

                # 检查总结是否包含必要字段
                required_fields = ["answer", "reference"]
                for field in required_fields:
                    if field not in summary_json:
                        print(f"❌ 总结格式错误：缺少必要字段 {field}")
                        raise ValueError(f"总结格式错误：缺少必要字段 {field}")

                # 检查 answer 字段
                if not isinstance(summary_json["answer"], str):
                    print("❌ 总结格式错误：answer 字段必须是字符串")
                    raise ValueError("总结格式错误：answer 字段必须是字符串")

                # 检查 reference 字段
                if not isinstance(summary_json["reference"], dict):
                    print("❌ 总结格式错误：reference 字段必须是字典")
                    raise ValueError("总结格式错误：reference 字段必须是字典")

                # 检查所有引用是否在 reference 中存在
                citations = re.findall(r'\[(\d+)\]', summary_json["answer"])
                if citations:
                    # 提取所有引用编号并转换为整数
                    citation_nums = [int(c) for c in citations]

                    # 检查所有引用是否在 reference 中存在
                    for num in citation_nums:
                        ref_key = f"[{num}]"
                        if ref_key not in summary_json["reference"]:
                            print(f"❌ 总结格式错误：reference 中缺少引用 [{num}]")
                            raise ValueError(f"总结格式错误：reference 中缺少引用 [{num}]")

                # 检查 reference 中的引用是否都在 answer 中使用
                for ref_key in summary_json["reference"]:
                    if ref_key not in summary_json["answer"]:
                        print(f"⚠️  警告：reference 中的引用 {ref_key} 未在 answer 中使用")

                print("✅ 总结格式校验通过")
            except json.JSONDecodeError as e:
                print(f"❌ 总结格式错误：JSON 解析失败 - {e}")
                print(f"生成的总结内容：{summary}")

                # 尝试使用默认值创建有效的总结
                default_summary = {
                    "answer": f"基于执行结果的总结：{str(result)}",
                    "reference": {}
                }
                summary = json.dumps(default_summary, ensure_ascii=False)
                print(f"⚠️  使用默认总结格式：{summary}")
            except Exception as e:
                print(f"❌ 总结校验失败：{e}")
                raise

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

            # 检查是否为 API 并发错误
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
    output_dir = "0609-codeact-batch_outputs"
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
    incorrect_file = "0609-codeact-incorrect.json"
    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_questions, f, ensure_ascii=False, indent=2)
    print(f"❌ 失败的问题（共 {len(incorrect_questions)} 个）已保存至 {incorrect_file}")

    # 单独保存 API 并发错误的问题
    concurrency_errors = [r for r in all_results if r.get("error_type") == "concurrency_error"]
    if concurrency_errors:
        concurrency_file = "0609-codeact-concurrency-errors.json"
        concurrency_data = {
            "total_concurrency_errors": len(concurrency_errors),
            "errors": [{"id": r["id"], "question": r["question"], "error": r["error"]} for r in concurrency_errors]
        }
        with open(concurrency_file, 'w', encoding='utf-8') as f:
            json.dump(concurrency_data, f, ensure_ascii=False, indent=2)
        print(f"⚠️ API 并发错误（共 {len(concurrency_errors)} 个）已保存至 {concurrency_file}")
    else:
        print(f"✅ 未检测到 API 并发错误")

# ======== 执行入口 ========
def load_existing_results(output_file: str) -> List[Dict]:
    if not os.path.exists(output_file):
        return []

    with open(output_file, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            return []

    return results if isinstance(results, list) else []


def result_key(result: Dict) -> str:
    if not isinstance(result, dict):
        return ""
    if result.get("id") is not None:
        return f"id::{result['id']}"
    return f"question::{result.get('question', '')}"


def dedupe_results(results: List[Dict]) -> List[Dict]:
    deduped_results = []
    index_map = {}

    for result in results:
        key = result_key(result)
        if not key:
            continue
        if key in index_map:
            deduped_results[index_map[key]] = result
        else:
            index_map[key] = len(deduped_results)
            deduped_results.append(result)

    return deduped_results


def is_successful_result(result: Dict) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    return bool(result.get("summary"))


def upsert_result(results: List[Dict], new_result: Dict) -> List[Dict]:
    key = result_key(new_result)
    if not key:
        return results

    updated_results = []
    replaced = False
    for result in results:
        if result_key(result) == key:
            if not replaced:
                updated_results.append(new_result)
                replaced = True
            continue
        updated_results.append(result)

    if not replaced:
        updated_results.append(new_result)

    return updated_results


def write_results(output_file: str, results: List[Dict]) -> List[Dict]:
    deduped_results = dedupe_results(results)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduped_results, f, ensure_ascii=False, indent=2)
    return deduped_results


async def process_question_batch_v2(batch: List[Dict], batch_idx: int, start_idx: int, output_dir: str, output_file: str) -> List[Dict]:
    results = dedupe_results(load_existing_results(output_file))
    existing_result_map = {
        result_key(result): result
        for result in results
        if result_key(result)
    }
    batch_results = []

    for i, item in enumerate(batch):
        question_id = item.get("id", start_idx + i + 1)
        current_key = f"id::{question_id}"
        existing_result = existing_result_map.get(current_key)

        if existing_result and is_successful_result(existing_result):
            print(f"Skipping question {question_id}: successful result already exists")
            batch_results.append(existing_result)
            continue

        result = await process_single_question(item, start_idx + i)
        if result is not None:
            results = upsert_result(results, result)
            results = write_results(output_file, results)
            existing_result_map[current_key] = result
            batch_results.append(result)
            print(f"Question {question_id} saved to {output_file}")

        if i < len(batch) - 1:
            print("Waiting 5 seconds before the next question...")
            time.sleep(5)

    batch_output_file = os.path.join(output_dir, f"output_batch_{batch_idx}.json")
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    print(f"Batch {batch_idx} completed and saved to {batch_output_file}")

    return batch_results


def process_questions_v2(input_file: str, output_file: str, batch_size: int = 5):
    output_dir = "0609-codeact-batch_outputs"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    existing_results = dedupe_results(load_existing_results(output_file))
    if existing_results:
        write_results(output_file, existing_results)
        successful_count = sum(1 for result in existing_results if is_successful_result(result))
        failed_count = sum(1 for result in existing_results if result.get("error"))
        print(
            f"Found {len(existing_results)} existing results: "
            f"{successful_count} successful, {failed_count} failed. "
            f"This run will skip successful items and retry failed or missing ones."
        )

    total_batches = (len(questions) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(questions))
        batch = questions[start:end]
        print(f"\n===== Processing batch {batch_idx}, {len(batch)} questions =====")

        asyncio.run(
            process_question_batch_v2(
                batch,
                batch_idx,
                start_idx=start,
                output_dir=output_dir,
                output_file=output_file,
            )
        )

        if batch_idx < total_batches - 1:
            print("Waiting 10 seconds before the next batch...")
            time.sleep(10)

    print(f"\nAll batches finished. Final results saved to {output_file}")

    final_results = dedupe_results(load_existing_results(output_file))

    incorrect_questions = [
        {"id": result.get("id"), "question": result["question"]}
        for result in final_results
        if "error" in result
    ]
    incorrect_file = "0609-codeact-incorrect.json"
    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_questions, f, ensure_ascii=False, indent=2)
    print(f"Failed questions saved to {incorrect_file}: {len(incorrect_questions)} items")

    concurrency_errors = [result for result in final_results if result.get("error_type") == "concurrency_error"]
    if concurrency_errors:
        concurrency_file = "0609-codeact-concurrency-errors.json"
        concurrency_data = {
            "total_concurrency_errors": len(concurrency_errors),
            "errors": [{"id": r["id"], "question": r["question"], "error": r["error"]} for r in concurrency_errors]
        }
        with open(concurrency_file, 'w', encoding='utf-8') as f:
            json.dump(concurrency_data, f, ensure_ascii=False, indent=2)
        print(f"Concurrency errors saved to {concurrency_file}: {len(concurrency_errors)} items")
    else:
        print("No concurrency errors detected")


process_question_batch = process_question_batch_v2
process_questions = process_questions_v2


if __name__ == "__main__":
    process_questions("question150.json", "0609-output.json")
