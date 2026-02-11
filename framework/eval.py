import json
import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from statistics import mean
from openai import OpenAI

# ========== 配置区域 ==========
# 若使用代理或本地部署模型可修改
MODEL_NAME = "deepseek-chat"
API_KEY = "" 
API_BASE = "https://api.deepseek.com/v1"

GOLD_PATH = "gold_answer.json"
PRED_PATH = "gemini-3-pro-preview-11-2025-output.json"
OUTPUT_JSON = "0108-gemini-3-pro-preview-11-2025-eval_results.json"
OUTPUT_METRICS = "0108-gemini-3-pro-preview-11-2025-eval_average.json"


# ========== 通用函数 ==========

def _extract_rating_json(text: str):
    """从模型返回中提取包含 rating 字段的合法 JSON
    支持：
    - ```json ... ``` 或 ``` ... ``` 包裹
    - 行内/末尾的 {"rating": 0.85}
    - 整段就是 {...} 的情况（即使后面有额外文本）
    """
    if not text:
        return None
    text = text.strip()
    
    # 1. 先尝试提取 ```json ... ``` 或 ``` ... ``` 包裹的内容
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if md_match:
        candidate = md_match.group(1).strip()
        if "rating" in candidate.lower():
            try:
                json.loads(candidate)  # 验证是否为有效 JSON
                return candidate
            except:
                pass
    
    # 2. 查找包含 "rating" 的 JSON 对象（支持多行、嵌套）
    # 从第一个 { 开始，找到匹配的 }，使用括号匹配确保提取完整
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                candidate = text[start_idx:i+1]
                if "rating" in candidate.lower():
                    try:
                        json.loads(candidate)  # 验证是否为有效 JSON
                        return candidate
                    except:
                        pass
                start_idx = -1
    
    return None


def llm_score(prompt: str) -> float:
    """通用LLM打分函数，返回 0~1 之间的浮点数"""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    content = None
    raw = None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            print("⚠️ 评分出错：模型返回为空")
            return 0.0
        
        raw = _extract_rating_json(content)
        if not raw:
            print("⚠️ 评分出错：无法从返回中提取 JSON")
            print(f"   原始返回前 500 字: {repr(content[:500])}")
            return 0.0
        
        data = json.loads(raw)
        rating = float(data.get("rating", 0))
        return max(0.0, min(rating, 1.0))
    except json.JSONDecodeError as e:
        print(f"⚠️ 评分出错（JSON 解析失败）: {e}")
        print(f"   提取的片段: {repr(raw[:200]) if raw else 'None'}")
        print(f"   原始返回前 300 字: {repr(content[:300]) if content else 'None'}")
        return 0.0
    except KeyError as e:
        print(f"⚠️ 评分出错（缺少 rating 字段）: {e}")
        print(f"   解析后的数据: {data if 'data' in locals() else '未解析'}")
        return 0.0
    except Exception as e:
        print(f"⚠️ 评分出错（其他异常）: {type(e).__name__}: {e}")
        if content:
            print(f"   原始返回前 300 字: {repr(content[:300])}")
        return 0.0


def parse_link(answer):
    """提取文本中的 URL"""
    try:
        urls = re.findall(r'https?://[^\s"\'\]}]+', str(answer))
        return set(urls)
    except Exception:
        return set()


def judge_precision(gold_answer, pred_answer):
    gold_links, pred_links = parse_link(gold_answer), parse_link(pred_answer)
    if not gold_links or not pred_links:
        return 0.0
    return len(pred_links & gold_links) / len(pred_links)


def judge_recall(gold_answer, pred_answer):
    gold_links, pred_links = parse_link(gold_answer), parse_link(pred_answer)
    if not gold_links or not pred_links:
        return 0.0
    return len(pred_links & gold_links) / len(gold_links)

import re
import json

def judge_clarity(pred_answer):
    """
    判断结构化回答是否清晰（Clarity）：
    支持 markdown 包裹（```json ... ```）
    结构中只能包含 answer 和 reference 两个字段
    reference 的 key 从 [1] 开始连续编号，value 必须是合法 aminer 链接
    answer 中必须包含且仅包含所有引用编号
    除 JSON / markdown 之外不得有额外内容
    返回：1 （清晰） 或 0 （不清晰）
    """

    def extract_json_str(result_str: str):
        """
        提取 result 中的纯 JSON 字符串，允许 markdown 包裹
        """
        result_str = result_str.strip()

        # ```json ... ``` 或 ``` ... ``` 包裹
        md_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", result_str, flags=re.IGNORECASE | re.DOTALL)
        if md_match:
            return md_match.group(1).strip()

        # 直接 JSON
        if result_str.startswith("{") and result_str.endswith("}"):
            return result_str

        return None

    def is_reference_valid(reference_dict):
        """
        校验 reference：
        - 是 dict
        - key 从 [1] 开始连续
        - value 为含 aminer 链接的字符串
        """
        if not isinstance(reference_dict, dict) or not reference_dict:
            return False

        keys = list(reference_dict.keys())
        # 检查 key 格式正确：[1], [2], ...
        if any(not re.fullmatch(r"\[\d+\]", k) for k in keys):
            return False

        # 检查 key 为连续编号
        nums = sorted(int(k.strip("[]")) for k in keys)
        if nums != list(range(1, len(nums) + 1)):
            return False

        # 检查 value 合法
        return all(isinstance(v, str) and "https://www.aminer.cn" in v for v in reference_dict.values())

    def extract_citation_keys(text: str):
        """提取形如 [1], [2] 的引用编号"""
        return set(re.findall(r"\[\d+\]", text))

    # Step 1: 获取 result 字段
    summary_field = pred_answer.get("summary", "")
    result_str = str(summary_field).strip()
    if not result_str:
        return 0

    # Step 2: 提取 JSON 主体
    json_str = extract_json_str(result_str)
    if not json_str:
        return 0

    # Step 3: 解析 JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return 0

    # Step 4: 必须只含 answer 和 reference
    if set(parsed.keys()) != {"answer", "reference"}:
        return 0

    answer = parsed["answer"]
    reference = parsed["reference"]

    # Step 5: reference 合法性
    if not is_reference_valid(reference):
        return 0

    # Step 6: answer 中引用必须完全匹配 reference
    cited, defined = extract_citation_keys(answer), set(reference.keys())
    if cited != defined:
        return 0

    # 全部通过
    return 1


def planning_edit_distance(pred_plan, gold_plan):
    """计算规划步骤的编辑距离"""
    pred_seq = [p["name"] for p in sorted(pred_plan, key=lambda x: x.get("order", 0))]
    gold_seq = [g["name"] for g in sorted(gold_plan, key=lambda x: x.get("order", 0))]

    n, m = len(pred_seq), len(gold_seq)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred_seq[i - 1] == gold_seq[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1,
                           dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


def check_api_success(execution_result):
    """检查API执行是否成功，检查 execution_result 中的错误"""
    if not execution_result:
        return 0
    last_result = execution_result[-1] if isinstance(execution_result, list) else execution_result
    for api_name, result in last_result.items():
        if isinstance(result, dict):
            if result.get("error") or result.get("success") == False:
                return 0
    return 1


# ========== 各维度 Prompt 定义 ==========

def make_prompt_correctness(question, gold, pred):
    return f"""
请你给待评估答案的“准确性”打分，评分范围 [0,1]:

问题：{question}
参考答案：{gold}
待评估答案：{pred}

准确性定义：待评估答案与参考答案相比，是否准确地回答问题，可以参考（待评估答案中出现的相关的信息 / 待评估答案中出现的全部信息）
非常差 —— 待评估答案所有内容均与参考答案及问题无关。
较差 —— 待评估答案存在大量与参考答案及问题无关的内容。
一般 —— 待评估答案存在一些与参考答案及问题无关的信息。
良好 —— 待评估答案与参考答案及问题基本相关，仅有个别偏差。
非常好 —— 待评估答案与参考答案及问题全部相关。

注意：如果待评估答案只给出找信息的步骤或方式，没给出找的结果，评分为0

**必须遵守的评分规则：**
1. **名单/人物/论文类问题**（如“合作者中有谁”“有哪些研究者”“相关论文”）：先计算**匹配百分比** = 待评估答案中与参考答案**一致的项目数**（同一人物、同一论文或同一链接视为一致）/ **参考答案中的项目总数**。rating 按该百分比给分（如 5 个中答对 4 个则 4/5=0.8；100%→1.0，0%→0）。若待评估答案与参考答案**完全不一致**（无重叠），则不得超过 **0.5**。
2. **论文/文献类问题**（如“相关论文”“有哪些文献”）：若待评估答案**未提供任何论文链接**（reference 为空或声称检索失败、未返回数据），则准确性不得超过 **0.8**；若有链接则仍按上条匹配百分比给分。
3. 非名单类问题（如事实判断、单一结论）仍按与参考答案的相关性、准确性综合给分。

请输出JSON：
{{"rating": 数字}}
"""


def make_prompt_integrality(question, gold, pred):
    return f"""
请你给待评估答案的“完整性”打分，评分范围 [0,1]:
1 表示完全覆盖，0 表示完全不覆盖。

问题：{question}
参考答案：{gold}
待评估答案答案：{pred}

完整性定义：与参考答案相比，待评估答案表述是否完整全面，有无遗漏信息点，可以参考（待评估答案中出现的相关的信息 / 参考答案中出现的全部信息）
非常差 —— 所有信息点均遗漏。
较差 —— 大部分信息点遗漏。
一般 —— 遗漏一些信息。
良好 —— 大部分信息均覆盖，仅有个别偏差。
非常好 —— 待评估答案覆盖了参考答案的全部信息。

注意：如果待评估答案只给出找信息的步骤或方式，没给出找的结果，评分为0

请输出JSON：
{{"rating": 数字}}
"""


def make_prompt_completeness(question, pred, gold):
    return f"""
请你评估该答案是否“完成了问题的核心目标且内容正确”，范围 [0,1]:
1=既完整覆盖又内容正确，0=未完成或内容错误。
**完成度同时考虑两点：是否覆盖、回应了各子查询/核对项，以及针对每项的回答是否与参考答案一致、正确。**

问题：{question}
参考答案：{gold}
待评估答案：{pred}

你应该如此评价：
（1）从用户查询中提取所有**子查询（sub-queries）
（2）为每个子查询确定其最终的核心意图（central intent）
（3）基于黄金答案，为每个子查询提取一个用于评估的核对清单（evaluation checklist），该清单需对应其核心意图。"
    请严格遵循以下指南：
    1. **子查询识别（Sub-query identification）
       - 如果用户查询只有一个核心意图，请提取完整查询作为唯一的子查询。
       - 如果用户查询中明显包含多个核心意图，请将每个核心意图分别提取为独立的子查询。
       - **不要（DO NOT）**：臆造不存在的子查询，或提取那些偏离主问题、无关紧要的子查询。
    2. **子查询改写（Sub-query Rewriting）**
       - 如果查询只有一个核心意图，请原样保留完整查询，作为唯一子查询。
       - 如果存在多个核心意图，请将每个子查询改写为一个能够独立理解的问题，避免依赖原始查询上下文。
    3. **核心意图（Central intent）**
       - 为每个子查询识别出其最终期望的输出形式（例如：论文列表、学者姓名、事实结论、研究观点等）。
       - 忽略中间推理步骤，只保留用户**最终想要的结果目标**。
    4. **评估核对清单提取（Evaluation checklist extraction）**
       - 从黄金答案中提取出与子查询核心意图直接对应的关键评估点。
       - 每一个事实实体（人物、论文、机构、属性、结论等）都应作为**独立的一项**列出。
       - 该清单应当足以让评估者仅凭清单内容即可判断外部答案的正确性，而无需阅读完整的黄金答案。
       - **忽略参考链接（reference URLs）**，仅考虑黄金答案中的事实性文本内容。

你需要根据上述指南，在思考过程中为问题生成子查询、改写后的子查询、核心意图以及评估核对清单，并按照清单的**完成且正确**的程度打分（仅覆盖但内容错误应扣分）。

输出：
{{"rating": 数字}}
"""


def make_prompt_faithfulness(pred, api_output):
    return f"""
请你评估以下回答的“忠实度”，即是否严格基于API返回内容，无编造成分，评分范围[0,1]:
忠实度定义：回答是否严格基于 API 返回的内容，是否存在编造、虚构、幻觉等非真实信息。
1=完全忠实，0=完全虚构。

答案：{pred}
API输出：{api_output}

输出：
{{"rating": 数字}}
"""


# ========== 主逻辑 ==========

def main():
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    gold_dict = {item["qid"]: item for item in gold_data}
    pred_dict = {item["id"]: item for item in pred_data}
    results, metrics = [], defaultdict(list)

    for pred_item in tqdm(pred_data, desc="评估中", ncols=100):
        qid = pred_item["id"]
        gold_item = gold_dict.get(qid)
        if not gold_item:
            print(f"⚠️  未找到问题 {qid} 的标准答案，跳过评估")
            continue
        question = gold_item.get("question", "")
        gold_ans = gold_item.get("result_edit", "")
        pred_ans = pred_item.get("summary", "")

        precision = judge_precision(gold_ans, pred_ans)
        recall = judge_recall(gold_ans, pred_ans)
        clarity = judge_clarity(pred_item)
        
        success = check_api_success(pred_item.get("execution_result", []))
        
        pred_plan = json.loads(pred_item.get("plan", "[]"))
        edit_distance = planning_edit_distance(
            pred_plan,
            gold_item.get("planning_text", [])
        )

        correctness = llm_score(make_prompt_correctness(question, gold_ans, pred_ans))
        integrality = llm_score(make_prompt_integrality(question, gold_ans, pred_ans))
        completeness = llm_score(make_prompt_completeness(question, pred_ans, gold_ans))
        
        execution_result = pred_item.get("execution_result", [])
        api_output = execution_result[-1] if execution_result else {}
        faithfulness = llm_score(make_prompt_faithfulness(pred_ans, api_output))

        eval_result = {
            "qid": qid,
            "precision": precision, "recall": recall, "clarity": clarity,
            "correctness": correctness, "integrality": integrality,
            "completeness": completeness, "faithfulness": faithfulness,
            "success": success, "edit_distance": edit_distance
        }

        for k, v in eval_result.items():
            if k != "qid":
                metrics[k].append(v)
        results.append(eval_result)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    average = {k: round(mean(v), 4) for k, v in metrics.items()}
    print("\n=== 平均指标 ===")
    for k, v in average.items():
        print(f"{k}: {v}")

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(average, f, indent=2)


if __name__ == "__main__":
    main()