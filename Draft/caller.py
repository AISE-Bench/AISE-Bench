import json, os, re
import asyncio
from api import AMinerAPI
from llm import llm_client
import logging
import sys
import streamlit as st

from new_tool_async import search_paper_id_gs

from tools import search_paper_id_tool, search_paper_detail_tool, search_author_detail_tool

# 设置OpenAI API配置（已在from_plan_to_result.py中配置）

# 延迟导入DRAFT优化流程，避免循环依赖
optimize_tool_documentation = None
batch_optimize_tools = None

def _lazy_import_draft():
    """
    延迟导入DRAFT优化流程
    """
    global optimize_tool_documentation, batch_optimize_tools
    if optimize_tool_documentation is None:
        from from_plan_to_result import optimize_tool_documentation as opt_doc
        from from_plan_to_result import batch_optimize_tools as batch_opt
        optimize_tool_documentation = opt_doc
        batch_optimize_tools = batch_opt

# 彻底清除旧的 `logging` 配置，确保 `logger` 正确生效
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 配置 Logger
logger = logging.getLogger(__name__)  # 获取 Logger 实例
logger.setLevel(logging.DEBUG)  # 确保 DEBUG 级别日志会被记录

# 创建文件处理器（写入日志文件，使用追加模式）
try:
    file_handler = logging.FileHandler("debug.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
except OSError:
    file_handler = None

# 创建控制台处理器（输出到终端）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 设置日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
if file_handler is not None:
    file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 确保 `logger` 之前没有 `Handler`，防止日志丢失
logger.handlers.clear()
if file_handler is not None:
    logger.addHandler(file_handler)
logger.addHandler(console_handler)


API_PARAM = {}

PARA = {
    "key1": "vaule1(str)",
    "key2": ['value2','value22']
}
PROMPT = """
【请注意：布尔类型的参数的值：True、False需要首字母一定要大写！不允许出现true、false！】

可用的API如下：
【search_paper_id】
use_topic参数必须要传！！
---
    {
        "type": "function",
        "function":{
            "name": "search_paper_id",
            "description": "基于多种条件搜论文",
            "parameters": {
                "type": "object",
                "properties": {
                    "use_topic": {
                        "type": "boolean",
                        "description": "是否使用联合搜索。如果为true，则使用topic字段进行搜索",
                    },
                    "topic_high":{
                        "type": "string",
                        "description": "含义为论文关键词，use_topic必须为true时可使用，是一个列表，示例值：['大模型','推理加速']",
                    },
                    "title":{
                        "type": "list",
                        "description": "当use_topic为false时，可以使用title查询，格式为[str,str,...]"
                    },
                    "year":{
                        "type": "list",
                        "description": "筛选年份，格式为[int,int,...]"
                    },
                    "n_citation_flag":{
                        "type": "boolean",
                        "description": "如果开启，会给引用量大的进行加分"
                    },
                    "size":{
                        "type": "integer",
                        "description": "返回的论文数量，最大为10"
                    },
                    "force_citation_sort":{
                        "type": "boolean",
                        "description": "完全按照citation排序"
                    },
                    "force_year_sort":{
                        "type": "boolean",
                        "description": "完全按照year排序"
                    },
                    "author_id":{
                        "type": "string",
                        "description": "作者id"
                    },
                    "author_terms":{
                        "type": "list",
                        "description": "作者名，作者名字查询，array里为or，尽可能多写变体，如张三、zhang s、zhangsan、z s等；格式为[str,str,...]"
                    },
                    "org_terms":{
                        "type": "list",
                        "description": "机构名字查询，array里为or；格式为[str,str,...]"
                    },
                    "org_id":{
                        "type": "string",
                        "description": "机构id"
                    },
                    "venue_id":{
                        "type": "string",
                        "description": "期刊会议id"
                    },
                    "venue_ids":{
                        "type": "list",
                        "description": "期刊会议id，格式为[str,str,...]"
                    },
                    "keywords":{
                        "type": "list",
                        "description": "论文关键词，格式为[str,str,...]"
                    },
                    "coauthors":{
                        "type": "list",
                        "description": "共同作者姓名列表，格式为[str,str,...]"
                    }
                },
                "required": ["use_topic"]
            }
        }
    },
输出：论文id，论文标题

---
【search_paper_detail】
输入：
    ids：[str,str, ...]  #论文id列表
输出：
_id	string	论文id
abstract	string	摘要
authors	array	作者
doi	string	DOI
eissn	string	EISSN
email	string	电子邮件
i	string	数组中i表示 被合并论文的id
issn	string	ISSN
issue	string	Issue
keywords	array	作者关键词
l	string	数组中l 代表被合并论文的源
lang	string	语言
n_citation	float	引用值
name	string	姓名
online_issn	string	online_issn
org	string	机构
orgid	string	机构id
page_end	string	结束页
page_start	string	开始页
publisher	string	来源关键词
raw	string	引用串
s	string	数据源中ID
sid	string	数据源中ID
title	string	标题
total	float	总数
ts	string	更新时间
url	array	来源url
venue	object	Venue
venue_hhb_id	string	venue_id
venue_id_list	array	venue_ids
versions	array	历史版本
vname	string	出版社名称
volume	string	卷号
vsid	string	数据源中出版社ID
year	float	年份

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
    name_search_type：str  #可选值: "fuzzy" 或 "exact"，默认为 "fuzzy" fuzzy为模糊搜索，exact为精确搜索
    category：str  #学科分类名称
    category_source:str # 不同的数字的含义如下： 0: "SJR", 1: "WOS", 2: "GB09", 3: "CCF", 4: "CSCD", 5: "CCJ", 6: "ARXIV", 7: "CJCR", 8: "JCR", 9: "SCI"
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
    name：str  #机构名称
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

class TaskExecutor:
    def __init__(self, tasks):
        self.tasks = {task["name"]: task for task in tasks}
        self.results = {}
        self.status = {}
        self.inputs = {}
        self.dependencies = {task["name"]: set(task["rely"]) for task in tasks}
        self.pending_tasks = {task["name"]: task for task in tasks if not task["rely"]}
        self.optimized_tools = {}  # 存储优化后的工具信息 {tool_name: optimized_tool_info}

    async def execute_api_call(self, api_name, params):
        aminer_api = AMinerAPI()
        logger.debug(f"调用{api_name}，传入参数: {params}")
        API_PARAM[api_name] = params
        try:
            # todo 更加优雅, 删除（1）等
            api_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', api_name).group(0)

            # 新的API调用
            if api_name == "search_paper_id":
                # 处理 coauthors 参数格式
                if "coauthors" in params:
                    if isinstance(params["coauthors"], list):
                        # 如果是空列表，删除该参数
                        if not params["coauthors"]:
                            del params["coauthors"]
                        else:
                            params["coauthors"] = ",".join(params["coauthors"])
                    # 如果是空字符串，删除该参数
                    elif not params["coauthors"]:
                        del params["coauthors"]
                # 移除可能存在的 use_topic 参数，因为 search_paper_id_tool 不接受这个参数
                if "use_topic" in params:
                    del params["use_topic"]
                response = await search_paper_id_tool(**params)

            elif api_name == "search_paper_detail":
                response = await search_paper_detail_tool(**params)

            elif api_name == "search_author_detail":
                # 将ids参数转换为author_ids
                if "ids" in params:
                    params["author_ids"] = params.pop("ids")
                response = await search_author_detail_tool(**params)

            elif api_name == "search_paper_id_gs":
                response = await search_paper_id_gs(**params)
                # 处理search_paper_id_gs的返回值（返回(idlist, search_context)元组）
                if isinstance(response, tuple) and len(response) == 2:
                    # 提取论文ID列表（response[0]是ID列表，response[1]是搜索上下文）
                    paper_ids = response[0]
                    response = {"data": paper_ids}
                else:
                    # 如果返回格式不正确，使用空列表
                    response = {"data": []}
                
            else:
                response = await aminer_api.call_api(api_name=api_name, payload=params)
                logger.debug(f"调用{api_name}，返回: {response}")

            return params, response

        except Exception as e:
            print(f"Error calling {api_name}: {e}")
            return params,{{api_name}: {str(e)}}

    async def execute_task(self, task_name):
        task = self.tasks[task_name]
        api_name = task["name"]
        params = task["params"]

        if params.get("topic_high"):
            params["topic_high"] = str(params["topic_high"])

        # 处理依赖项，如果 order > 1 且有依赖，收集所有 rely 任务的结果并调用 LLM 生成 params
        if task["order"] > 1 and task["rely"]:
            
            api_name_temp = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', api_name).group(0)

            file_path = os.path.join("prompts/apis", f"{api_name_temp}.txt")  # 构造文件路径
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    params_temp = f.read().strip()
            except FileNotFoundError:
                print(f"警告：未找到 API 提示文件 {file_path}")
                params_temp = ""

            # 只收集已完成的依赖任务结果
            rely_results = {}
            print(f"=== 依赖任务检查 ===")
            print(f"当前任务: {api_name}")
            print(f"当前任务 order: {task['order']}")
            print(f"当前任务依赖: {task['rely']}")
            print(f"当前可用的结果: {list(self.results.keys())}")
            print(f"依赖列表是否为空: {not task['rely']}")
            
            for dep in task["rely"]:
                if not dep:  # 跳过空依赖
                    continue
                print(f"检查依赖: {dep}")
                print(f"  依赖类型: {type(dep)}")
                print(f"  依赖值: '{dep}'")
                
                # 检查是否有完全匹配的依赖任务
                if dep in self.results:
                    print(f"✓ 找到完全匹配的依赖任务: {dep}")
                    rely_results[dep] = self.results[dep]
                else:
                    # 检查是否有带编号的依赖任务（如 search_author_id(1)）
                    found = False
                    print(f"  未找到完全匹配，开始检查基础名称匹配...")
                    for completed_task in self.results:
                        base_task_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', completed_task).group(0)
                        print(f"  检查已完成任务: '{completed_task}', 基础名称: '{base_task_name}'")
                        if base_task_name == dep:
                            print(f"✓ 找到基础名称匹配的依赖任务: {completed_task}")
                            rely_results[completed_task] = self.results[completed_task]
                            found = True
                            break
                    if not found:
                        print(f"✗ 未找到匹配的依赖任务: {dep}")
                        print(f"  可用的已完成任务: {list(self.results.keys())}")
            
            # 检查是否有依赖任务结果
            if not rely_results:
                print(f"✗ 依赖任务尚未完成，跳过 {api_name} 任务，等待依赖任务完成")
                print(f"  原因: rely_results 为空")
                # 不要标记为 failed，直接返回，让任务保持 pending 状态
                return
            print(f"✓ 成功收集依赖任务结果: {list(rely_results.keys())}")

            prompt = f"""请根据用户给出的数据，重新生成 {api_name} 所需的参数。
                参数的格式示例如下：{PARA}
                api可用的参数如下：{params_temp}
                参考的参数如下：{params}
                你需要从中找出 {api_name} 所需的参数，并按照格式示例生成参数。
            【请注意，你只需生成上述的参数格式即可,不需要任何额外的解释或者说明！】
            【你需要生成完整的参数，不要中间截断！】
            """
            query = f"用户给出的数据：{rely_results}"

            print("~~~提示词~~~")
            print("Prompt:", prompt)
            print("~~~输入~~~")
            print("Query:", query)

            
            result = llm_client(prompt=prompt, query=query)
            
            if isinstance(result, dict):
                result = json.dumps(result, ensure_ascii=False)  # 转换为 JSON 字符串

            # cleaned_string = result.replace("```json", "").replace("```", "").strip()
            try:
                result = eval(result)
            except:
                result = eval(result, {"null": None, "true": True, "false": False})
                
            params = result

            if params.get("topic_high"):
                params["topic_high"] = json.dumps([params["topic_high"]], ensure_ascii=False)  # 转换为 JSON 字符串
                # parsed_data = json.loads(params["topic_high"])  # 解析为 Python 列表
                # params["topic_high"] = json.dumps(parsed_data)  # 转换为 JSON 格式（带转义）

            print("~~~成功输出~~~")
            print("Params:", params)

        # 特殊处理：如果是 search_paper_id 任务，且依赖 search_paper_detail 任务
        # 从 search_paper_detail 结果中提取合作者信息并添加到 coauthors 参数
        api_name_temp = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', api_name).group(0)
        print(f"=== 调试信息 ===")
        print(f"当前任务: {api_name}")
        print(f"任务名称: {api_name_temp}")
        print(f"任务依赖: {task['rely']}")
        print(f"当前可用的结果: {list(self.results.keys())}")
        print(f"当前参数: {params}")
        
        if api_name_temp == "search_paper_id":
            print(f"=== 开始处理 search_paper_id 任务 ===")
            for dep in task["rely"]:
                print(f"检查依赖任务: {dep}")
                # 检查依赖任务是否已完成
                dep_completed = False
                dep_result = {}
                dep_task_info = {}
                
                # 首先检查是否有完全匹配的依赖任务
                if dep in self.results:
                    dep_completed = True
                    dep_result = self.results.get(dep, {})
                    dep_task_info = self.tasks.get(dep, {})
                else:
                    # 检查是否有带编号的依赖任务（如 search_paper_detail(1)）
                    for completed_task in self.results:
                        base_task_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', completed_task).group(0)
                        if base_task_name == dep:
                            dep_completed = True
                            dep_result = self.results.get(completed_task, {})
                            dep_task_info = self.tasks.get(completed_task, {})
                            break
                
                if dep_completed:
                    print(f"依赖任务 {dep} 已完成，开始提取合作者信息")
                    print(f"依赖任务的结果类型: {type(dep_result)}")
                    print(f"依赖任务的结果内容: {dep_result}")
                    
                    # 检查结果是否是字典
                    if isinstance(dep_result, dict) and dep_result.get("data"):
                        print(f"依赖任务的结果包含 data 字段")
                        data = dep_result["data"]
                        print(f"data 字段的类型: {type(data)}")
                        
                        # 如果 data 是列表
                        if isinstance(data, list):
                            print(f"data 字段是列表，开始处理列表中的每个元素")
                            # 检查依赖任务是否是 search_paper_detail
                            dep_api_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', dep_task_info.get("name", "")).group(0)
                            print(f"依赖任务的 API 名称: {dep_api_name}")
                            
                            if dep_api_name == "search_paper_detail":
                                print(f"开始从 search_paper_detail 提取合作者信息")
                                # 从 search_paper_detail 结果中提取合作者信息
                                coauthors = []
                                for paper_info in data:
                                    if isinstance(paper_info, dict):
                                        print(f"处理论文信息: {paper_info.get('title', '无标题')}")
                                        if paper_info.get("authors"):
                                            for author in paper_info["authors"]:
                                                if isinstance(author, dict):
                                                    # 提取作者姓名
                                                    author_name = author.get("author") or author.get("name")
                                                    if author_name:
                                                        coauthors.append(author_name)
                                                        print(f"提取到作者: {author_name}")
                                # 如果提取到合作者信息，添加到 params 中
                                if coauthors:
                                    # 去重
                                    coauthors = list(set(coauthors))
                                    # 直接转换为逗号分隔的字符串
                                    params["coauthors"] = ",".join(coauthors)
                                    print(f"从 search_paper_detail 提取的合作者信息: {params['coauthors']}")
                            else:
                                print(f"依赖任务不是 search_paper_detail，而是 {dep_api_name}")
                        else:
                            print(f"data 字段不是列表，类型是: {type(data)}")
                    else:
                        print(f"依赖任务的结果不包含 data 字段或 data 字段为空")
                else:
                    print(f"依赖任务 {dep} 尚未完成，无法提取合作者信息")
        
        print(f"=== API 调用前的参数 ===")
        print(f"调用 API: {api_name}")
        print(f"调用参数: {params}")

        self.status[task_name] = "running"
        try:
            inputs, result = await self.execute_api_call(api_name, params)

            print(f"=== API 调用后的结果 ===")
            print(f"API 结果: {result}")
                
            necessary_fields = ["paper_id", "id", "interests","n_citation", "citation", "name","gender","org","org_name", "org_id","title","_id","authors","abstract","keywords","orgid","venue","year","ai_interests","bio","bio_zh","emails","nation","position","position_zh","work", "h_index"] 
                
            # 处理结果数据
            if isinstance(result, dict) and result.get("data"):
                if all(isinstance(item, str) for item in result["data"]):  # 检查所有元素是否都是字符串
                    pass
                else:
                    result["data"] = [
                        {key: item[key] for key in necessary_fields if key in item} 
                        for item in result["data"]
                    ]

            # 保存结果
            self.results[task_name] = result
            self.status[task_name] = "completed"
            self.inputs[task_name] = inputs
            print(f"✓ 任务 {task_name} 完成，结果已保存")
        except Exception as e:
            print(f"API 调用失败: {str(e)}")
            self.status[task_name] = "failed"
            self.results[task_name] = {"error": str(e)}

    def get_tool_info(self, task_name):
        """
        获取工具信息，优先使用优化后的工具信息
        
        Args:
            task_name: 任务名称
            
        Returns:
            工具信息字典
        """
        # 提取基础任务名称（去除编号）
        base_task_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', task_name).group(0)
        
        # 优先使用优化后的工具信息
        if base_task_name in self.optimized_tools:
            print(f"✓ 使用优化后的工具信息: {base_task_name}")
            return self.optimized_tools[base_task_name]
        
        # 如果没有优化信息，使用默认的工具信息映射
        print(f"⚠ 使用默认工具信息: {base_task_name}")
        
        # 定义工具信息映射
        tool_info_map = {
            "search_paper_id": {
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
            "search_paper_detail": {
                "category": "academic",
                "name": "search_paper_detail",
                "description": "根据论文ID获取论文详情",
                "required_parameters": {
                    "paper_ids": "论文ID列表"
                },
                "optional_parameters": {}
            },
            "search_venue_id": {
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
            "search_venue_detail": {
                "category": "academic",
                "name": "search_venue_detail",
                "description": "根据期刊或会议ID获取详情",
                "required_parameters": {
                    "venue_ids": "期刊或会议ID列表"
                },
                "optional_parameters": {}
            },
            "search_author_id": {
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
            "search_author_detail": {
                "category": "academic",
                "name": "search_author_detail",
                "description": "根据作者ID获取作者详情",
                "required_parameters": {
                    "author_ids": "作者ID列表"
                },
                "optional_parameters": {}
            },
            "search_org_id": {
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
            "search_org_detail": {
                "category": "academic",
                "name": "search_org_detail",
                "description": "根据机构ID获取机构详情",
                "required_parameters": {
                    "org_ids": "机构ID列表"
                },
                "optional_parameters": {}
            },
            "search_paper_id_gs": {
                "category": "academic",
                "name": "search_paper_id_gs",
                "description": "使用Google Scholar搜索论文ID",
                "required_parameters": {
                    "keywords": "论文关键词列表"
                },
                "optional_parameters": {
                    "years": "发表年份列表",
                    "size": "返回结果数量"
                }
            }
        }
        
        return tool_info_map.get(base_task_name, {
            "category": "academic",
            "name": base_task_name,
            "description": f"执行{base_task_name}操作",
            "required_parameters": {},
            "optional_parameters": {}
        })
    
    def update_tool_info(self, optimized_tool):
        """
        更新工具信息到内存中
        
        Args:
            optimized_tool: 优化后的工具信息
        """
        tool_name = optimized_tool.get('name', 'unknown')
        
        # 将优化后的工具信息保存到内存中
        self.optimized_tools[tool_name] = optimized_tool
        
        print(f"\n=== 工具信息已更新 ===")
        print(f"工具名称: {tool_name}")
        print(f"优化后描述: {optimized_tool.get('description', '无描述')}")
    
    def optimize_tools(self):
        """
        优化所有工具的文档（如果优化文件不存在才执行）
        """
        # 检查batch_optimized_tools.json是否存在
        if os.path.exists("batch_optimized_tools.json"):
            print(f"\n=== 发现已存在的优化文件，跳过工具优化 ===")
            print(f"使用 batch_optimized_tools.json 中的优化结果")
            
            # 加载优化后的工具信息到内存
            try:
                with open("batch_optimized_tools.json", "r", encoding="utf-8") as f:
                    batch_optimized_tools = json.load(f)
                
                # 将优化后的工具信息存储到内存中
                for tool in batch_optimized_tools:
                    tool_name = tool.get('name', 'unknown')
                    self.optimized_tools[tool_name] = tool
                
                print(f"✅ 已加载 {len(batch_optimized_tools)} 个优化后的工具到内存")
                return batch_optimized_tools
            except Exception as e:
                print(f"❌ 加载 batch_optimized_tools.json 失败: {e}")
                return []
        
        # 检查optimized_tools.json是否存在
        if os.path.exists("optimized_tools.json"):
            print(f"\n=== 发现已存在的优化文件，跳过工具优化 ===")
            print(f"使用 optimized_tools.json 中的优化结果")
            
            # 加载优化后的工具信息到内存
            try:
                with open("optimized_tools.json", "r", encoding="utf-8") as f:
                    optimized_tools = json.load(f)
                
                # 将优化后的工具信息存储到内存中
                for tool in optimized_tools:
                    tool_name = tool.get('name', 'unknown')
                    self.optimized_tools[tool_name] = tool
                
                print(f"✅ 已加载 {len(optimized_tools)} 个优化后的工具到内存")
                return optimized_tools
            except Exception as e:
                print(f"❌ 加载 optimized_tools.json 失败: {e}")
                return []
        
        print(f"\n=== 开始优化工具文档 ===")
        
        # 延迟导入DRAFT优化流程
        try:
            _lazy_import_draft()
            if optimize_tool_documentation is None:
                print("⚠️ 无法导入DRAFT优化流程，跳过工具优化")
                return []
        except Exception as e:
            print(f"⚠️ 导入DRAFT优化流程失败: {e}")
            return []
        
        # 收集所有唯一的工具名称
        unique_tools = set()
        for task_name in self.tasks:
            base_task_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', task_name).group(0)
            unique_tools.add(base_task_name)
        
        print(f"发现 {len(unique_tools)} 个唯一工具:")
        for tool_name in unique_tools:
            print(f"- {tool_name}")
        
        # 优化每个工具
        optimized_tools = []
        for tool_name in unique_tools:
            try:
                # 获取工具信息
                tool_info = self.get_tool_info(tool_name)
                
                # 调用DRAFT优化流程
                optimized_tool = optimize_tool_documentation(tool_info)
                
                # 更新工具信息
                self.update_tool_info(optimized_tool)
                optimized_tools.append(optimized_tool)
                
            except Exception as e:
                print(f"❌ 工具 {tool_name} 优化失败: {e}")
        
        # 保存优化结果
        if optimized_tools:
            with open("optimized_tools.json", "w", encoding="utf-8") as f:
                json.dump(optimized_tools, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 工具文档优化完成，结果已保存至 optimized_tools.json")
        
        return optimized_tools
    
    async def run(self, optimize_tools_first=True):
        """
        运行任务执行器
        
        Args:
            optimize_tools_first: 是否先优化工具文档
            
        Returns:
            输入参数和执行结果
        """
        # 先优化工具文档
        if optimize_tools_first:
            self.optimize_tools()
        
        iteration_count = 0
        max_iterations = 100
        
        while self.pending_tasks:
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"警告：达到最大迭代次数 {max_iterations}，可能存在循环依赖")
                print(f"未完成的任务: {list(self.pending_tasks.keys())}")
                print(f"已完成的结果: {list(self.results.keys())}")
                print(f"任务状态: {self.status}")
                break
            
            # 只执行状态为 pending 的任务
            tasks_to_execute = [task for task in list(self.pending_tasks.keys()) 
                               if self.status.get(task) != "running"]
            
            if not tasks_to_execute:
                # 没有可执行的任务，说明存在循环依赖或所有任务都在运行中
                print(f"警告：没有可执行的任务，可能存在循环依赖")
                print(f"未完成的任务: {list(self.pending_tasks.keys())}")
                print(f"任务状态: {self.status}")
                break
            
            tasks = [self.execute_task(task) for task in tasks_to_execute]
            await asyncio.gather(*tasks)

            # 移除已完成或失败的任务
            for task in list(self.pending_tasks.keys()):
                if self.status.get(task) == "completed":
                    del self.pending_tasks[task]
                elif self.status.get(task) == "failed":
                    del self.pending_tasks[task]

            # 添加依赖已完成的任务
            for task_name, task in self.tasks.items():
                if task_name not in self.results and task_name not in self.pending_tasks:
                    # 检查所有依赖任务是否已完成
                    all_completed = True
                    for dep in self.dependencies[task_name]:
                        # 检查是否有完全匹配的依赖任务
                        dep_completed = dep in self.results
                        if not dep_completed:
                            # 检查是否有带编号的依赖任务（如 search_author_id(1)）
                            for completed_task in self.results:
                                base_task_name = re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*', completed_task).group(0)
                                if base_task_name == dep:
                                    dep_completed = True
                                    break
                        if not dep_completed:
                            all_completed = False
                            break
                    if all_completed:
                        self.pending_tasks[task_name] = task
                        print(f"添加任务到待执行列表: {task_name}")

        return self.inputs, self.results


TASKS = [{"name": "search_paper_id", "rely": [], "order": 1, "params": {"keywords": ["llm"]}}]

async def main():
    executor = TaskExecutor(TASKS)
    inputs, results = await executor.run()
    print("Execution Results:", results)
    print("Execution Inputs:", inputs)

if __name__ == "__main__":
    asyncio.run(main())

# async def execute_api_call(**args):
#     aminer_api = AMinerAPI()

#     try:
#         api_name = args.pop("api_name")
#         payload = args.pop("params")
#         if api_name == "search_paper_id":
#             response = await search_paper_id_tool(**payload)
#             response = {"data": response}
#         else:
#             response = await aminer_api.call_api(api_name=api_name, payload=payload)
#         print(response)
#         return response

#     except Exception as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":

#     kk = {"name": "search_venue_id", "rely": [], "order": 1, "params": {"name": ["kdd"]}}
#     asyncio.run(execute_api_call(api_name=kk["name"], params=kk["params"]))






