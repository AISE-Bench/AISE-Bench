import json
import os
import sys
import asyncio
import time
from typing import List, Dict, Optional, Any

from llm import llm_client
from config import CHATGLM_API_BASE, CHATGLM_API_KEY
from caller import TaskExecutor

# 导入SoAy的solution_toolkit
import sys
sys.path.append('SoAy-master')
from SolutionLibrary_toolkit.solution_construction import solution_toolkit

# API组合生成器
class APICombinationGenerator:
    def __init__(self):
        self.toolkit = solution_toolkit(domain='custom')
        self.config_file = 'config/custom_function_config.jsonl'
        self.info_dict_list = self.toolkit.collectInformation(self.config_file)
        self.graph = dict(self.toolkit.buildGraph(info_dict_list=self.info_dict_list))
        self.start_nodes = ['search_paper_id', 'search_author_id', 'search_venue_id', 'search_org_id', 'search_paper_id_gs']
        self.combination_library = self._generate_combinations()
        self._save_combination_library()

    def _generate_combinations(self):
        """生成API组合库"""
        combination_library = []
        for start_node in self.start_nodes:
            if start_node in self.graph:
                sampled_paths = self.toolkit.samplePaths(self.graph, start_node)
                # 为每个路径生成组合
                for path in sampled_paths:
                    try:
                        combination_list = self.toolkit.sampleIO(path=path, info_dict_list=self.info_dict_list)
                        for each in combination_list:
                            combination_library.append({
                                'path': ' -> '.join(path),
                                'io_combinations': each
                            })
                    except Exception as e:
                        print(f"生成组合时出错: {e}")
        return combination_library

    def _save_combination_library(self):
        """保存API组合库到文件"""
        import jsonlines
        import os
        # 确保结果目录存在
        result_dir = 'results/custom'
        os.makedirs(result_dir, exist_ok=True)
        # 保存组合库
        with jsonlines.open(f'{result_dir}/combinations.jsonl', 'w') as f:
            for combo in self.combination_library:
                f.write(combo)
        print(f"解决方案库已保存到 {result_dir}/combinations.jsonl")
        # 显示API关系图
        self.toolkit.showGraph(self.graph)
        print(f"API关系图已保存到 {result_dir}/graph.html")

    def get_relevant_combinations(self, question: str) -> List[Dict[str, Any]]:
        """根据问题获取相关的API组合"""
        # 简单的相关性过滤，实际应用中可以使用更复杂的方法
        relevant_combinations = []
        for combo in self.combination_library:
            # 检查路径中的API是否与问题相关
            path_apis = combo['path'].split(' -> ')
            # 简单的关键词匹配
            if any(api in question.lower() for api in ['paper', 'author', 'venue', 'org']):
                relevant_combinations.append(combo)
        # 返回前10个相关组合
        return relevant_combinations[:10]

# 创建API组合生成器实例
api_comb_generator = APICombinationGenerator()

# API参数标准化处理
def standardize_api_params(api_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """标准化API参数格式，确保参数格式正确"""
    standardized_params = params.copy()

    if api_name == "search_paper_id":
        # 处理coauthors参数
        if "coauthors" in standardized_params:
            if isinstance(standardized_params["coauthors"], list):
                if not standardized_params["coauthors"]:
                    del standardized_params["coauthors"]
                else:
                    standardized_params["coauthors"] = ",".join(standardized_params["coauthors"])
            elif not standardized_params["coauthors"]:
                del standardized_params["coauthors"]
        # 确保排序参数正确
        if "sort" not in standardized_params:
            standardized_params["sort"] = "citation"

    elif api_name == "search_author_detail":
        # 将ids参数转换为author_ids
        if "ids" in standardized_params:
            standardized_params["author_ids"] = standardized_params.pop("ids")

    return standardized_params

# 结果缓存机制
class ApiCache:
    """API调用结果缓存"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, api_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """获取缓存的结果"""
        key = self._generate_key(api_name, params)
        if key in self.cache:
            # 更新访问顺序
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, api_name: str, params: Dict[str, Any], result: Any) -> None:
        """设置缓存的结果"""
        key = self._generate_key(api_name, params)
        # 缓存大小控制
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = result
        self.access_order.append(key)

    def _generate_key(self, api_name: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        sorted_params = sorted(params.items()) if params else []
        return f"{api_name}:{hash(str(sorted_params))}"

# 创建全局缓存实例
api_cache = ApiCache()

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

            # 获取与问题相关的API组合
            relevant_combinations = api_comb_generator.get_relevant_combinations(question)
            print(f"【相关API组合】找到 {len(relevant_combinations)} 个相关组合")

            # 构建包含API组合的提示词
            combinations_prompt = "\n# 推荐的API组合：\n"
            for i, combo in enumerate(relevant_combinations[:3]):  # 只使用前3个组合
                combinations_prompt += f"{i+1}. API路径: {combo['path']}\n"
                combinations_prompt += f"   输入: {combo['io_combinations'].get('input', '')}\n"
                combinations_prompt += f"   输出: {combo['io_combinations'].get('output', '')}\n"

            plan_prompt = PLAN_PROMPT + combinations_prompt + "\n用户问题如下：" + question
            plan = llm_client(prompt=plan_prompt, query=question)
            print("【规划】\n", plan)

            # 解析计划并标准化API参数
            plan_data = plan if isinstance(plan, dict) else json.loads(plan)
            if isinstance(plan_data, list):
                # 标准化每个任务的参数
                for task in plan_data:
                    api_name = task.get("name", "")
                    # 移除编号部分（如search_author_id(1)）
                    base_api_name = api_name.split('(')[0]
                    if "params" in task:
                        task["params"] = standardize_api_params(base_api_name, task["params"])

            executor = TaskExecutor(plan_data)
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

            # 检查是否为参数格式错误
            is_param_error = any(keyword in error_msg.lower() for keyword in [
                "参数", "param", "format", "invalid", "错误"
            ])

            if is_param_error and attempt < max_retries - 1:
                print(f"⚠️ 检测到参数格式错误，重新生成计划...")
                # 直接重试，不等待
                continue

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⏳ 指数退避等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)  # 使用异步等待
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
            await asyncio.sleep(5)  # 使用异步等待

    batch_output_file = os.path.join(output_dir, f"output_batch_{batch_idx}.json")
    # 确保只保存当前批次的结果
    current_batch_results = results[-(len(batch)):] if len(results) >= len(batch) else results
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(current_batch_results, f, ensure_ascii=False, indent=2)
    print(f"✅ 批次 {batch_idx} 完成，结果保存至 {batch_output_file}")

    return results

# 主函数
def process_questions(input_file: str, output_file: str, batch_size: int = 5):
    output_dir = "0604-soay-batch_outputs"
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
            wait_time = min(10, 5 + batch_idx * 2)  # 动态调整等待时间
            print(f"⏳ 等待 {wait_time} 秒后处理下一批次...")
            time.sleep(wait_time)

    print(f"\n🎉 所有批次处理完成，最终结果已保存至 {output_file}")

    # 保存所有失败的问题（包括并发错误）
    incorrect_questions = [{"question": r["question"]} for r in all_results if "error" in r]
    incorrect_file = "gemini-3-pro-preview-11-2025-incorrect.json"
    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_questions, f, ensure_ascii=False, indent=2)
    print(f"❌ 失败的问题（共 {len(incorrect_questions)} 个）已保存至 {incorrect_file}")

    # 单独保存 API 并发错误的问题
    concurrency_errors = [r for r in all_results if r.get("error_type") == "concurrency_error"]
    if concurrency_errors:
        concurrency_file = "0604-soay-concurrency-errors.json"
        concurrency_data = {
            "total_concurrency_errors": len(concurrency_errors),
            "errors": [{"id": r["id"], "question": r["question"], "error": r["error"]} for r in concurrency_errors]
        }
        with open(concurrency_file, 'w', encoding='utf-8') as f:
            json.dump(concurrency_data, f, ensure_ascii=False, indent=2)
        print(f"⚠️ API 并发错误（共 {len(concurrency_errors)} 个）已保存至 {concurrency_file}")
    else:
        print(f"✅ 未检测到 API 并发错误")

    # 保存处理统计信息
    success_count = len([r for r in all_results if "error" not in r])
    total_count = len(all_results)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0

    stats_file = "0604-soay-stats.json"
    stats_data = {
        "total_questions": total_count,
        "success_count": success_count,
        "error_count": len(incorrect_questions),
        "concurrency_error_count": len(concurrency_errors),
        "success_rate": f"{success_rate:.2f}%"
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"📊 处理统计信息已保存至 {stats_file}")

# ======== 执行入口 ========
if __name__ == "__main__":
    process_questions("question150.json", "0604-soay.json")
