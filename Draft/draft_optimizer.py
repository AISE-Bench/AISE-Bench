import json
import re
import sys
import os
import time
import numpy as np
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Dict, List, Any

class DraftOptimizer:
    def __init__(self, model="glm-4.7", temperature=0.2, top_p=1, max_tokens=2000, episodes=5, api_key=""):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.episodes = episodes
        self.api_key = api_key
        self.prompts = self._load_prompts()
    
    def _load_prompts(self):
        prompts = {}
        try:
            with open('prompts/Explorer.txt', 'r', encoding='utf-8') as file:
                content = file.read()
                if '=========' in content:
                    prompts['explorer'], prompts['explorer_follow'] = content.split('=========')
                else:
                    prompts['explorer'] = content
                    prompts['explorer_follow'] = ""
            
            with open('prompts/Analyzer.txt', 'r', encoding='utf-8') as file:
                content = file.read()
                if '=========' in content:
                    prompts['analyzer'], prompts['analyzer_follow'] = content.split('=========')
                else:
                    prompts['analyzer'] = content
                    prompts['analyzer_follow'] = ""
            
            with open('prompts/Rewriter.txt', 'r', encoding='utf-8') as file:
                content = file.read()
                if '=========' in content:
                    prompts['rewriter'], prompts['rewriter_follow'] = content.split('=========')
                else:
                    prompts['rewriter'] = content
                    prompts['rewriter_follow'] = ""
            
            with open('prompts/rewrite_tool_doc.txt', 'r', encoding='utf-8') as file:
                prompts['rewrite_tool_doc'] = file.read()
        except FileNotFoundError:
            print("警告：未找到提示文件，使用默认提示")
            # 设置默认提示
            prompts['explorer'] = "你是一个工具探索专家，根据工具描述生成用户查询和参数。\n工具描述：{Tool Description}\n\n请直接返回用户查询，不需要JSON格式。"
            prompts['explorer_follow'] = "已探索查询：{Explored queries}\n建议：{Suggestions}\n\n请直接返回新的用户查询，不需要JSON格式。"
            prompts['analyzer'] = "你是一个工具分析专家，分析工具使用示例并提供改进建议。\n工具描述：{Tool Description}\n使用示例：{usage_example}\n\n请直接返回改进建议，不需要JSON格式。"
            prompts['analyzer_follow'] = "历史改进：{History}\n\n请直接返回新的改进建议，不需要JSON格式。"
            prompts['rewriter'] = "你是一个工具文档重写专家，根据分析结果重写工具文档。\n工具描述：{Tool Description}\n使用示例：{usage_example}\n建议：{Suggestions}\n当前文档：{tool_description}\n\n请直接返回重写后的工具文档，不需要JSON格式。"
            prompts['rewriter_follow'] = "历史改进：{History}\n\n请直接返回新的重写文档，不需要JSON格式。"
            prompts['rewrite_tool_doc'] = "你是一个工具文档优化专家，优化整个工具文档。\n工具描述：{Tool Description}\n\n请直接返回优化后的工具文档，不需要JSON格式。"
        return prompts
    
    def glm_response(self, messages):
        try:
            # 使用Zhipu AI OpenAI兼容API
            url = "https://open.bigmodel.cn/api/mss/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            }
            print(f"  发送API请求: {url}")
            print(f"  请求模型: {self.model}")
            print(f"  请求消息数量: {len(messages)}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            print(f"  API响应状态码: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            print(f"  API响应内容长度: {len(str(result))}")
            
            # 处理OpenAI格式的响应
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    print(f"  LLM生成内容: {content[:200]}...")
                    
                    # 尝试解析JSON格式响应
                    try:
                        # 清理响应内容，移除可能的代码块包装
                        cleaned_content = content.strip()
                        # 移除代码块标记
                        if cleaned_content.startswith('```json'):
                            cleaned_content = cleaned_content[7:]
                        elif cleaned_content.startswith('```'):
                            cleaned_content = cleaned_content[3:]
                        if cleaned_content.endswith('```'):
                            cleaned_content = cleaned_content[:-3]
                        cleaned_content = cleaned_content.strip()
                        
                        # 尝试解析JSON
                        import json
                        json_response = json.loads(cleaned_content)
                        return json_response
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {str(e)}")
                        # 如果不是JSON格式，返回原始内容
                        return content
            # 如果没有有效的响应，返回空字符串
            return ""
        except Exception as e:
            print(f"Zhipu AI API错误: {str(e)}")
            # 即使出错，也返回空字符串
            return ""
    
    def glm_embedding(self, text):
        try:
            # 使用Zhipu AI嵌入API
            url = "https://open.bigmodel.cn/api/mss/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "text-embedding-3-small",  # OpenAI兼容的嵌入模型
                "input": text
            }
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            return []
        except Exception as e:
            print(f"Zhipu AI Embedding API错误: {str(e)}")
            return []
    
    def cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def compute_similarity_and_bleu(self, reference_sentence, candidate_sentence):
        if not reference_sentence or not candidate_sentence:
            return 0.0
        
        # 计算相似度
        reference_embedding = self.glm_embedding(reference_sentence)
        candidate_embedding = self.glm_embedding(candidate_sentence)
        similarity = self.cosine_similarity(reference_embedding, candidate_embedding)
        
        # 计算BLEU分数
        reference = [reference_sentence.lower().split()]
        candidate = candidate_sentence.lower().split()
        smoothie = SmoothingFunction().method4
        try:
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        except:
            bleu_score = 0.0
        
        # 计算内容质量评分（通过LLM评估）
        quality_prompt = f"请评估以下两个工具文档的质量，返回0-1之间的分数，1表示质量最高：\n\n原始文档：{reference_sentence[:300]}...\n\n优化后文档：{candidate_sentence[:300]}...\n\n请只返回分数，不要有其他内容。"
        
        quality_messages = [
            {"role": "system", "content": "你是一个工具文档评估专家，能够客观评估文档质量。"},
            {"role": "user", "content": quality_prompt}
        ]
        
        quality_response = self.glm_response(quality_messages)
        try:
            quality_score = float(quality_response.strip())
            quality_score = max(0.0, min(1.0, quality_score))
        except:
            quality_score = 0.5
        
        # 综合评分
        delta = (bleu_score * 0.2 + similarity * 0.3 + quality_score * 0.5)
        return delta
    
    def change_name(self, name):
        change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
        if name in change_list:
            name = "is_" + name.lower()
        return name
    
    def standardize(self, string):
        if not string:
            return ""
        res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_")
        string = res.sub("_", string)
        string = re.sub(r"(_)\1+", "_", string).lower()
        while string and string[0] == "_":
            string = string[1:]
        while string and string[-1] == "_":
            string = string[:-1]
        if string and string[0].isdigit():
            string = "get_" + string
        return string
    
    def optimize_tool_documentation(self, tool_info):
        """
        优化工具文档
        :param tool_info: 工具信息字典
        :return: 优化后的工具信息字典
        """
        if not tool_info:
            return tool_info
        
        # 检查是否为字符串类型（可能是工具名称）
        if isinstance(tool_info, str):
            return tool_info
        
        print(f"\n=== DRAFT优化详细过程 ===")
        print(f"优化前工具信息: {tool_info}")
        
        # 处理单个工具的信息
        if "name" in tool_info and "description" in tool_info:
            api_name = tool_info["name"]
            last_description = tool_info["description"]
            required_params = tool_info.get("required_parameters", {})
            optional_params = tool_info.get("optional_parameters", {})
            
            print(f"\n--- 开始优化 API: {api_name} ---")
            print(f"初始文档: {last_description}")
            
            # 初始化探索历史
            explored_queries = []
            rewrite_history = []
            rewrite_history.append(last_description)
            
            # 执行多轮优化
            for episode in range(self.episodes):
                print(f"\n=== 第 {episode+1} 轮优化 ===")
                # 准备工具信息
                current_tool_info = {
                    'name': api_name,
                    'description': rewrite_history[-1],
                    'required_parameters': required_params,
                    'optional_parameters': optional_params
                }
                
                # 1. 探索器 - 生成真实查询
                print(f"1. 探索器 - 生成查询示例")
                explorer_prompt = self.prompts['explorer'].format(**{"Tool Description": last_description})
                if explored_queries:
                    explorer_prompt += "\n" + self.prompts['explorer_follow'].format(
                        **{"Explored queries": "\n".join(explored_queries)},
                        Suggestions=""
                    )
                
                explorer_messages = [
                    {"role": "system", "content": "你是一个工具探索专家，根据工具描述生成用户查询和参数。"},
                    {"role": "user", "content": explorer_prompt}
                ]
                
                explorer_response = self.glm_response(explorer_messages)
                if explorer_response:
                    try:
                        # 处理JSON格式响应
                        if isinstance(explorer_response, dict) and "User Query" in explorer_response:
                            user_query = explorer_response["User Query"]
                            print(f"  生成的查询: {user_query[:200]}...")
                            explored_queries.append(user_query)
                        else:
                            # 尝试从字符串中提取User Query
                            import json
                            if isinstance(explorer_response, str):
                                # 尝试解析字符串为JSON
                                json_response = json.loads(explorer_response)
                                if isinstance(json_response, dict) and "User Query" in json_response:
                                    user_query = json_response["User Query"]
                                    print(f"  生成的查询: {user_query[:200]}...")
                                    explored_queries.append(user_query)
                                else:
                                    # 处理非JSON格式响应
                                    print(f"  生成的查询: {explorer_response[:200]}...")
                                    explored_queries.append(explorer_response)
                            else:
                                # 处理其他类型的响应
                                print(f"  生成的查询: {str(explorer_response)[:200]}...")
                                explored_queries.append(str(explorer_response))
                    except Exception as e:
                        print(f"处理探索器响应失败: {str(e)}")
                        # 即使出错，也添加响应到历史
                        explored_queries.append(str(explorer_response))
                
                # 2. 分析器 - 生成真实分析建议
                print(f"2. 分析器 - 分析查询结果")
                analyzer_prompt = self.prompts['analyzer'].format(
                    **{"Tool Description": last_description},
                    usage_example="\n".join(explored_queries[-3:]) if explored_queries else "暂无使用示例"
                )
                
                analyzer_messages = [
                    {"role": "system", "content": "你是一个工具分析专家，分析工具使用示例并提供改进建议。"},
                    {"role": "user", "content": analyzer_prompt}
                ]
                
                analyzer_response = self.glm_response(analyzer_messages)
                # 处理JSON格式响应
                if isinstance(analyzer_response, dict) and "Suggestion" in analyzer_response:
                    suggestions = analyzer_response["Suggestion"]
                else:
                    suggestions = analyzer_response if analyzer_response else "建议：1. 明确说明API的用途和功能；2. 详细说明参数的含义和使用方法；3. 指出返回结果的格式和内容。"
                print(f"  分析建议: {suggestions[:200]}...")
                
                # 3. 重写器 - 生成真实优化描述
                print(f"3. 重写器 - 优化工具文档")
                rewriter_prompt = self.prompts['rewriter'].format(
                    **{"Tool Description": last_description},
                    usage_example="\n".join(explored_queries[-3:]) if explored_queries else "暂无使用示例",
                    Suggestions=suggestions,
                    tool_description=rewrite_history[-1]
                )
                
                rewriter_messages = [
                    {"role": "system", "content": "你是一个工具文档重写专家，根据分析结果重写工具文档。"},
                    {"role": "user", "content": rewriter_prompt}
                ]
                
                rewritten_desc = self.glm_response(rewriter_messages)
                # 处理JSON格式响应
                if isinstance(rewritten_desc, dict) and "Rewritten Description" in rewritten_desc:
                    rewritten_desc = rewritten_desc["Rewritten Description"]
                elif not rewritten_desc:
                    rewritten_desc = rewrite_history[-1]
                
                print(f"  优化前: {rewrite_history[-1][:100]}...")
                print(f"  优化后: {rewritten_desc[:100]}...")
                rewrite_history.append(rewritten_desc)
                last_description = rewritten_desc
                
                # 检查是否收敛
                if len(rewrite_history) > 1:
                    delta = self.compute_similarity_and_bleu(rewrite_history[-2], rewrite_history[-1])
                    print(f"4. 评估 - 相似度和BLEU分数: {delta:.4f}")
                    if delta > 0.85:
                        print(f"  优化收敛，停止迭代")
                        break
            
            # 更新API文档
            tool_info['description'] = rewrite_history[-1]
            print(f"\n--- API优化完成 ---")
            print(f"最终文档: {tool_info['description']}")
        # 处理包含tool_guidelines的工具信息
        elif "tool_guidelines" in tool_info:
            for api_name, api_info in tool_info["tool_guidelines"].items():
                if "description" in api_info:
                    last_description = api_info["description"]
                    required_params = api_info.get("required_parameters", {})
                    optional_params = api_info.get("optional_parameters", {})
                    
                    print(f"\n--- 开始优化 API: {api_name} ---")
                    print(f"初始文档: {last_description}")
                    
                    # 初始化探索历史
                    explored_queries = []
                    rewrite_history = []
                    rewrite_history.append(last_description)
                    
                    # 执行多轮优化
                    for episode in range(self.episodes):
                        print(f"\n=== 第 {episode+1} 轮优化 ===")
                        # 准备工具信息
                        current_tool_info = {
                            'category': tool_info.get('category', ''),
                            'name': api_name,
                            'description': rewrite_history[-1],
                            'required_parameters': required_params,
                            'optional_parameters': optional_params
                        }
                        
                        # 1. 探索器 - 生成真实查询
                        print(f"1. 探索器 - 生成查询示例")
                        explorer_prompt = self.prompts['explorer'].format(**{"Tool Description": last_description})
                        if explored_queries:
                            explorer_prompt += "\n" + self.prompts['explorer_follow'].format(
                                **{"Explored queries": "\n".join(explored_queries)},
                                Suggestions=""
                            )
                        
                        explorer_messages = [
                            {"role": "system", "content": "你是一个工具探索专家，根据工具描述生成用户查询和参数。"},
                            {"role": "user", "content": explorer_prompt}
                        ]
                        
                        explorer_response = self.glm_response(explorer_messages)
                        if explorer_response:
                            try:
                                # 处理JSON格式响应
                                if isinstance(explorer_response, dict) and "User Query" in explorer_response:
                                    user_query = explorer_response["User Query"]
                                    print(f"  生成的查询: {user_query[:200]}...")
                                    explored_queries.append(user_query)
                                else:
                                    # 尝试从字符串中提取User Query
                                    import json
                                    if isinstance(explorer_response, str):
                                        # 尝试解析字符串为JSON
                                        json_response = json.loads(explorer_response)
                                        if isinstance(json_response, dict) and "User Query" in json_response:
                                            user_query = json_response["User Query"]
                                            print(f"  生成的查询: {user_query[:200]}...")
                                            explored_queries.append(user_query)
                                        else:
                                            # 处理非JSON格式响应
                                            print(f"  生成的查询: {explorer_response[:200]}...")
                                            explored_queries.append(explorer_response)
                                    else:
                                        # 处理其他类型的响应
                                        print(f"  生成的查询: {str(explorer_response)[:200]}...")
                                        explored_queries.append(str(explorer_response))
                            except Exception as e:
                                print(f"处理探索器响应失败: {str(e)}")
                                # 即使出错，也添加响应到历史
                                explored_queries.append(str(explorer_response))
                        
                        # 2. 分析器 - 生成真实分析建议
                        print(f"2. 分析器 - 分析查询结果")
                        analyzer_prompt = self.prompts['analyzer'].format(
                            **{"Tool Description": last_description},
                            usage_example="\n".join(explored_queries[-3:]) if explored_queries else "暂无使用示例"
                        )
                        
                        analyzer_messages = [
                            {"role": "system", "content": "你是一个工具分析专家，分析工具使用示例并提供改进建议。"},
                            {"role": "user", "content": analyzer_prompt}
                        ]
                        
                        analyzer_response = self.glm_response(analyzer_messages)
                        # 处理JSON格式响应
                        if isinstance(analyzer_response, dict) and "Suggestion" in analyzer_response:
                            suggestions = analyzer_response["Suggestion"]
                        else:
                            suggestions = analyzer_response if analyzer_response else "建议：1. 明确说明API的用途和功能；2. 详细说明参数的含义和使用方法；3. 指出返回结果的格式和内容。"
                        print(f"  分析建议: {suggestions[:200]}...")
                        
                        # 3. 重写器 - 生成真实优化描述
                        print(f"3. 重写器 - 优化工具文档")
                        rewriter_prompt = self.prompts['rewriter'].format(
                            **{"Tool Description": last_description},
                            usage_example="\n".join(explored_queries[-3:]) if explored_queries else "暂无使用示例",
                            Suggestions=suggestions,
                            tool_description=rewrite_history[-1]
                        )
                        
                        rewriter_messages = [
                            {"role": "system", "content": "你是一个工具文档重写专家，根据分析结果重写工具文档。"},
                            {"role": "user", "content": rewriter_prompt}
                        ]
                        
                        rewritten_desc = self.glm_response(rewriter_messages)
                        # 处理JSON格式响应
                        if isinstance(rewritten_desc, dict) and "Rewritten Description" in rewritten_desc:
                            rewritten_desc = rewritten_desc["Rewritten Description"]
                        elif not rewritten_desc:
                            rewritten_desc = rewrite_history[-1]
                        
                        print(f"  优化前: {rewrite_history[-1][:100]}...")
                        print(f"  优化后: {rewritten_desc[:100]}...")
                        rewrite_history.append(rewritten_desc)
                        last_description = rewritten_desc
                        
                        # 检查是否收敛
                        if len(rewrite_history) > 1:
                            delta = self.compute_similarity_and_bleu(rewrite_history[-2], rewrite_history[-1])
                            print(f"4. 评估 - 相似度和BLEU分数: {delta:.4f}")
                            if delta > 0.85:
                                print(f"  优化收敛，停止迭代")
                                break
                    
                    # 更新API文档
                    api_info['description'] = rewrite_history[-1]
                    print(f"\n--- API优化完成 ---")
                    print(f"最终文档: {api_info['description']}")
        
        # 优化整个工具文档
        print(f"\n--- 优化整个工具文档 ---")
        # 生成全局工具描述
        tool_name = tool_info.get('tool_name', 'search_api')
        tool_description = f"{tool_name} 是一个用于搜索学术论文信息的工具。"
        tool_description += "\n该工具具有以下功能："
        
        # 检查是否包含tool_guidelines
        if "tool_guidelines" in tool_info:
            for i, (api_name, api_info) in enumerate(tool_info["tool_guidelines"].items(), 1):
                api_desc = api_info.get('description', api_name)
                tool_description += f"\n{i}. {api_name}：{api_desc}"
        else:
            # 处理单个工具的情况
            if "name" in tool_info:
                api_name = tool_info["name"]
                api_desc = tool_info.get('description', api_name)
                tool_description += f"\n1. {api_name}：{api_desc}"
        
        print(f"全局优化结果: {tool_description[:200]}...")
        tool_info['tool_description'] = tool_description
        
        print(f"\n=== DRAFT优化完成 ===")
        print(f"优化后工具信息: {tool_info}")
        
        return tool_info
    
    def optimize_tool_documents(self, tools_info):
        """
        优化多个工具文档
        :param tools_info: 工具信息列表或字典
        :return: 优化后的工具信息
        """
        if isinstance(tools_info, list):
            return [self.optimize_tool_documentation(tool) for tool in tools_info]
        elif isinstance(tools_info, dict):
            # 检查是否为包含 tool_guidelines 的工具信息结构
            if "tool_guidelines" in tools_info:
                # 直接调用 optimize_tool_documentation 处理整个结构
                return self.optimize_tool_documentation(tools_info)
            # 检查是否为包含多个工具信息的字典
            elif all(isinstance(v, dict) and "name" in v and "description" in v for v in tools_info.values()):
                # 处理包含多个工具信息的字典
                optimized_tools = {}
                for tool_name, tool_info in tools_info.items():
                    optimized_tools[tool_name] = self.optimize_tool_documentation(tool_info)
                return optimized_tools
            else:
                # 处理其他类型的字典
                return {k: self.optimize_tool_documentation(v) for k, v in tools_info.items()}
        return tools_info
