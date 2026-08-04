import json
from llm import llm_client
from caller import TaskExecutor
import asyncio

class AcademicMethod:
    def __init__(self, question):
        self.question = question
    
    def get_examples(self, tool):
        # 可以返回一些示例，这里暂时返回None
        return None
    
    async def step(self, tool, examples, it=0, prev_outputs=None):
        # 生成任务规划
        plan_prompt = f"""你是一个规划专家，你将根据用户的问题，选择一个或多个合适的API，使用API检索到相关信息，从而可以准确回答用户的问题。【thinking过程不要超过十句话】
   你生成的格式示例如下：
    [
        {{
            "name": "search_author_id",
            "rely": [],
            "order": 1,
            "params": {{"interest": ["Optical communication"]}}
        }},
        {{
            "name": "search_author_detail",
            "rely": ["search_author_id"],
            "order": 2,
            "params": {{"ids": []}}
        }},
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
    "search_paper_id": {{
        "description": "根据条件搜索论文ID",
        "parameters": {{
            "titles": ["论文标题"], 
            "keywords": ["关键词"], 
            "years": {{
                "type": "array",
                "description": "发表年份列表, 年份用整数表示"
            }},
            "is_sci": {{
                "type": "boolean",
                "description": "是否为SCI论文"
            }},
            "language": "论文语言(str), 使用ISO 639-1标准如'en', 'zh'",
            "sort": "排序方式(str)，只能选择: year, citation",
            "author": "作者姓名(str)",
            "author_id": "作者ID(str)",
            "coauthors": {{
                "type": "array",
                "description": "共同作者姓名列表，指的是不包括author参数作者的其他作者"
            }},
            "org": "机构或学校名称(str)",
            "org_id": "机构或学校ID(str)",
            "venues": {{
                "type": "array",
                "description": "期刊或会议列表，使用英文小写缩写，不需要带年份"            
            }},
            "venue_ids": ["期刊或会议ID"], 
            "size": {{
                "type": "integer",
                "description": "返回结果数量，必须小于100"                   
            }}
        }},
        "response": {{
            "description": "返回结果列表, 每个元素为一个字典(代表一篇论文)",
            "data": [
                {{
                    "paper_id": "论文ID(str)",
                    "title": "论文标题(str)"
                }}
            ]
        }}                     
    }},

---
【search_paper_detail】
    "search_paper_detail":{{
        "description": "根据论文ID列表，获取论文详细信息",
        "parameters": {{
            "paper_ids": ["论文ID"]
        }},
        "response":{{
            "description": "返回结果列表, 每个元素为一个字典(代表一篇论文)",
            "data": [
                {{
                    "paper_id": "论文ID(str)",
                    "title": "论文标题(str)",
                    "abstract": "论文摘要(str)",
                    "year": "发表年份(float)",
                    "citation": "被引用次数(float)",
                    "keywords": ["关键词"],
                    "authors": [
                        {{
                            "author": "作者姓名(str)",
                            "author_id": "作者ID(str)",
                            "org": "作者机构(str)",
                            "org_id": "作者机构ID(str)",
                            "email":"作者邮箱(str)"
                        }}
                    ],
                    "org": "发表机构(str)",
                    "org_id": "发表机构ID(str)",    
                    "venue": "发表的期刊或会议(str)"
                }}
            ]
        }}
    }},

---
【search_author_id】
    {{
        "type": "function",
        "function":{{
            "name": "search_author_id",
            "description": "根据姓名、机构、兴趣、国家等条件搜索学者",
            "parameters": {{
                "type": "object",
                "properties": {{
                    "name": {{
                        "type": "string",
                        "description": "学者姓名",
                    }},
                    "org": {{
                        "type": "string",
                        "description": "学者所在机构",
                    }},
                    "size": {{
                        "type": "integer",
                        "description": "返回的学者数量，最大为1000",
                    }},
                    "interest": {{
                        "type": "list",
                        "description": "学者兴趣，格式为[str,str,...]"
                    }},
                    "nation": {{
                        "type": "list",
                        "description": "学者所在国家，格式为[str,str,...]"
                    }},
                    "order": {{
                        "type": "string",
                        "description": "排序字段名 n_citation, n_pubs, h_index"
                    }},
                    "asc": {{
                        "type": "boolean",
                        "description": "true 升序 false 降序"
                    }}
                }},
                "required": []
            }}
        }}
    }},
---
【search_author_detail】根据学者ID列表获取学者详情
输入：ids：[str,str, ...]  #id列表

---
【search_venue_id】根据期刊名称搜索期刊ID、期刊标准名称
输入： 
    name：str  #期刊名称
    name_search_type：str  #可选值: "fuzzy" 或 "exact"，默认为 "fuzzy" fuzzy为模糊搜索，exact为精确搜索
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

用户问题如下：{self.question}"""
        
        plan = llm_client(prompt=plan_prompt, query=self.question)
        print(f"【生成的规划】\n{plan}")
        
        # 执行规划
        try:
            executor = TaskExecutor(plan if isinstance(plan, dict) else json.loads(plan))
            result = await executor.run()
            print(f"【执行结果】\n{result}")
            
            # 评估结果质量
            score_prompt = f"""请评估以下API执行结果对回答用户问题的质量，返回一个0-100的分数，分数越高表示结果越有用。
用户问题：{self.question}
API执行结果：{result}
请只返回一个数字分数，不要有其他任何文字。"""
            
            score = llm_client(prompt=score_prompt, query=self.question)
            print(f"【评估分数】\n{score}")
            
            # 解析分数
            try:
                score = float(score)
            except:
                # 如果无法解析，返回一个默认分数
                score = 50.0
                
            return result, plan, score
        except Exception as e:
            print(f"执行失败：{str(e)}")
            return None, plan, 0.0
