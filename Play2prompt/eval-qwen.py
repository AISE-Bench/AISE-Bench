import json
import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from statistics import mean
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Set, Tuple
import unicodedata
from openai import OpenAI

# ========== 閰嶇疆鍖哄煙 ==========
# 鑻ヤ娇鐢ㄤ唬鐞嗘垨鏈湴閮ㄧ讲妯″瀷鍙慨鏀?
MODEL_NAME = "deepseek-chat"
API_KEY = "" 
API_BASE = "https://api.deepseek.com/v1"

GOLD_PATH = "gold_answer.json"
PRED_PATH = "0131-play2prompt-output.json"
OUTPUT_JSON = "0409-deepseek-eval_results.json"
OUTPUT_METRICS = "0409-deepseek-eval_average.json"


# ========== 閫氱敤鍑芥暟 ==========

# Hand-written bilingual aliases. This map can be extended over time.
MANUAL_BILINGUAL_ALIASES: Dict[str, Set[str]] = {
    "tsinghua university": {"\u6e05\u534e\u5927\u5b66"},
    "peking university": {"\u5317\u4eac\u5927\u5b66"},
    "jie tang": {"\u5510\u6770"},
}


CRITICAL_PARAMS: Dict[str, Set[str]] = {
    "search_paper_id": {"titles", "keywords", "years", "author", "author_id", "org", "org_id", "venues", "venue_ids", "coauthors"},
    "search_paper_detail": {"paper_ids"},
    "search_author_id": {"author", "name", "org", "orgs", "org_ids", "interests", "interest"},
    "search_author_detail": {"author_ids", "ids"},
    "search_venue_id": {"venue", "name"},
    "search_venue_detail": {"venue_ids", "ids"},
    "search_org_id": {"orgs", "name"},
    "search_org_detail": {"org_ids", "ids"},
    "search_paper_id_gs": {"query"},
}

GLOBAL_ALIAS_LOOKUP: Dict[str, Set[str]] = {}


def normalize_text(value: Any) -> str:
    """Normalize multilingual text for robust matching."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_person_name(value: Any) -> str:
    """
    Normalize names and make two-token order variants comparable.
    Example: "Jie Tang" and "Tang Jie".
    """
    text = normalize_text(value)
    if not text:
        return ""
    parts = text.split()
    if len(parts) == 2:
        direct = f"{parts[0]} {parts[1]}"
        swapped = f"{parts[1]} {parts[0]}"
        return min(direct, swapped)
    return text


def iter_dict_nodes(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield every dict node from a nested dict/list object."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dict_nodes(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_dict_nodes(item)


def build_bilingual_alias_lookup(gold_data: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Build alias pairs from:
    1) manual aliases
    2) gold fields like (name, name_zh), (org, org_zh), (orgs[i], org_zhs[i]).
    """
    alias_lookup: Dict[str, Set[str]] = defaultdict(set)

    def add_alias_pair(a: Any, b: Any) -> None:
        a_norm = normalize_text(a)
        b_norm = normalize_text(b)
        if not a_norm or not b_norm or a_norm == b_norm:
            return
        alias_lookup[a_norm].add(b_norm)
        alias_lookup[b_norm].add(a_norm)

    for en_name, zh_names in MANUAL_BILINGUAL_ALIASES.items():
        for zh_name in zh_names:
            add_alias_pair(en_name, zh_name)

    for item in gold_data:
        for node in iter_dict_nodes(item):
            add_alias_pair(node.get("name"), node.get("name_zh"))
            add_alias_pair(node.get("org"), node.get("org_zh"))

            orgs = node.get("orgs")
            org_zhs = node.get("org_zhs")
            if isinstance(orgs, list) and isinstance(org_zhs, list):
                for en_org, zh_org in zip(orgs, org_zhs):
                    add_alias_pair(en_org, zh_org)

    return alias_lookup


def is_bilingual_name_match(a: Any, b: Any, alias_lookup: Dict[str, Set[str]], fuzzy_threshold: float = 0.92) -> bool:
    """
    Match order:
    1) exact normalized match
    2) person-name order invariant match
    3) alias lookup match
    4) high-threshold fuzzy fallback
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    if normalize_person_name(a) == normalize_person_name(b):
        return True
    if b_norm in alias_lookup.get(a_norm, set()) or a_norm in alias_lookup.get(b_norm, set()):
        return True

    return SequenceMatcher(None, a_norm, b_norm).ratio() >= fuzzy_threshold


def simple_lemma(token: str) -> str:
    """A lightweight lemmatizer to reduce trivial morphology mismatch."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def normalize_lexical_form(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    tokens = [simple_lemma(t) for t in text.split()]
    return " ".join(tokens)


def base_api_name(name: Any) -> str:
    text = str(name or "").strip()
    m = re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*", text)
    return m.group(0) if m else text


def unordered_list_semantic_equal(gold_list: List[Any], pred_list: List[Any]) -> bool:
    """Order-insensitive list equality using semantic matching."""
    if len(gold_list) != len(pred_list):
        return False
    used = [False] * len(pred_list)
    for g in gold_list:
        found = False
        for idx, p in enumerate(pred_list):
            if used[idx]:
                continue
            if semantic_equal(g, p):
                used[idx] = True
                found = True
                break
        if not found:
            return False
    return True


def semantic_equal(a: Any, b: Any) -> bool:
    """Semantic-ish equality for parameter values."""
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-8

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(semantic_equal(a[k], b[k]) for k in a.keys())

    if isinstance(a, (list, tuple, set)) and isinstance(b, (list, tuple, set)):
        return unordered_list_semantic_equal(list(a), list(b))

    if is_bilingual_name_match(a, b, GLOBAL_ALIAS_LOOKUP):
        return True

    a_lex = normalize_lexical_form(a)
    b_lex = normalize_lexical_form(b)
    if not a_lex or not b_lex:
        return False
    if a_lex == b_lex:
        return True
    return SequenceMatcher(None, a_lex, b_lex).ratio() >= 0.92


def safe_parse_plan(plan_value: Any) -> List[Dict[str, Any]]:
    if isinstance(plan_value, list):
        return [x for x in plan_value if isinstance(x, dict)]
    if isinstance(plan_value, str):
        try:
            parsed = json.loads(plan_value)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            return []
    return []


def calls_from_api_input_dict(api_input: Any) -> List[Tuple[str, Dict[str, Any]]]:
    calls = []
    if not isinstance(api_input, dict):
        return calls
    for api_name, params in api_input.items():
        if not isinstance(params, dict):
            params = {}
        calls.append((str(api_name), params))
    return calls


def calls_from_plan(plan_list: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    if not plan_list:
        return []

    indexed_plan = list(enumerate(plan_list))
    sorted_plan = sorted(indexed_plan, key=lambda x: (x[1].get("order", 10**9), x[0]))
    calls = []
    for _, step in sorted_plan:
        name = str(step.get("name", ""))
        params = step.get("params", {})
        if not isinstance(params, dict):
            params = {}
        calls.append((name, params))
    return calls


def split_calls_by_base(calls: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for name, params in calls:
        grouped[base_api_name(name)].append(params if isinstance(params, dict) else {})
    return grouped


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_param_metrics(gold_item: Dict[str, Any], pred_item: Dict[str, Any]) -> Dict[str, float]:
    """Compute parameter-level precision/recall/F1, split by critical vs optional keys."""
    gold_calls = calls_from_api_input_dict(gold_item.get("api_input", {}))

    execution_result = pred_item.get("execution_result", [])
    pred_calls = []
    if isinstance(execution_result, list) and execution_result and isinstance(execution_result[0], dict):
        pred_calls = calls_from_api_input_dict(execution_result[0])
    if not pred_calls:
        pred_calls = calls_from_plan(safe_parse_plan(pred_item.get("plan", [])))

    gold_by_api = split_calls_by_base(gold_calls)
    pred_by_api = split_calls_by_base(pred_calls)
    all_apis = set(gold_by_api.keys()) | set(pred_by_api.keys())

    tp = fp = fn = 0
    ctp = cfp = cfn = 0
    otp = ofp = ofn = 0

    def update_counts(kind: str, api: str, key: str) -> None:
        nonlocal tp, fp, fn, ctp, cfp, cfn, otp, ofp, ofn
        is_critical = key in CRITICAL_PARAMS.get(api, set())
        if kind == "tp":
            tp += 1
            if is_critical:
                ctp += 1
            else:
                otp += 1
        elif kind == "fp":
            fp += 1
            if is_critical:
                cfp += 1
            else:
                ofp += 1
        elif kind == "fn":
            fn += 1
            if is_critical:
                cfn += 1
            else:
                ofn += 1

    for api in all_apis:
        g_list = gold_by_api.get(api, [])
        p_list = pred_by_api.get(api, [])
        n = max(len(g_list), len(p_list))

        for i in range(n):
            g_params = g_list[i] if i < len(g_list) else {}
            p_params = p_list[i] if i < len(p_list) else {}

            g_keys = set(g_params.keys())
            p_keys = set(p_params.keys())
            all_keys = g_keys | p_keys

            for key in all_keys:
                in_g = key in g_params
                in_p = key in p_params
                if in_g and in_p:
                    if semantic_equal(g_params[key], p_params[key]):
                        update_counts("tp", api, key)
                    else:
                        update_counts("fp", api, key)
                        update_counts("fn", api, key)
                elif in_p and not in_g:
                    update_counts("fp", api, key)
                elif in_g and not in_p:
                    update_counts("fn", api, key)

    precision, recall, f1 = prf(tp, fp, fn)
    c_precision, c_recall, c_f1 = prf(ctp, cfp, cfn)
    o_precision, o_recall, o_f1 = prf(otp, ofp, ofn)

    return {
        "param_precision": precision,
        "param_recall": recall,
        "param_f1": f1,
        "critical_param_precision": c_precision,
        "critical_param_recall": c_recall,
        "critical_param_f1": c_f1,
        "optional_param_precision": o_precision,
        "optional_param_recall": o_recall,
        "optional_param_f1": o_f1,
    }


def score_step_result(step_result: Any) -> float:
    """
    Per-step score for PCS:
    - 1.0: success with informative non-empty payload
    - 0.5: success but empty payload
    - 0.0: failed or not executed
    """
    if step_result is None:
        return 0.0

    if isinstance(step_result, dict):
        if step_result.get("error") or step_result.get("success") is False:
            return 0.0

        if "data" in step_result:
            data = step_result.get("data")
            if data in (None, "", [], {}):
                return 0.5
            return 1.0

        informative_keys = [
            k for k, v in step_result.items()
            if k not in {"success", "msg", "log_id", "total"} and v not in (None, "", [], {})
        ]
        if informative_keys:
            return 1.0
        return 0.5

    if isinstance(step_result, list):
        return 1.0 if len(step_result) > 0 else 0.5

    if isinstance(step_result, str):
        return 1.0 if step_result.strip() else 0.5

    return 1.0


def compute_partial_completion_score(pred_item: Dict[str, Any]) -> float:
    plan_list = safe_parse_plan(pred_item.get("plan", []))
    if not plan_list:
        return 0.0

    indexed_plan = list(enumerate(plan_list))
    ordered_steps = [step for _, step in sorted(indexed_plan, key=lambda x: (x[1].get("order", 10**9), x[0]))]

    execution_result = pred_item.get("execution_result", [])
    outputs = execution_result[-1] if isinstance(execution_result, list) and execution_result else {}
    if not isinstance(outputs, dict):
        outputs = {}

    used_exact = set()
    outputs_by_base: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    for k, v in outputs.items():
        outputs_by_base[base_api_name(k)].append((k, v))
    used_base_idx: Dict[str, int] = defaultdict(int)

    step_scores: List[float] = []
    for step in ordered_steps:
        step_name = str(step.get("name", ""))
        step_base = base_api_name(step_name)
        result_obj = None

        if step_name in outputs and step_name not in used_exact:
            result_obj = outputs[step_name]
            used_exact.add(step_name)
        else:
            idx = used_base_idx[step_base]
            base_items = outputs_by_base.get(step_base, [])
            if idx < len(base_items):
                result_obj = base_items[idx][1]
                used_base_idx[step_base] += 1

        step_scores.append(score_step_result(result_obj))

    return float(sum(step_scores) / len(step_scores)) if step_scores else 0.0


def llm_score(prompt: str) -> float:
    """閫氱敤LLM鎵撳垎鍑芥暟锛岃繑鍥?0~1 涔嬮棿鐨勬诞鐐规暟"""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        rating = float(data["rating"])
        return max(0.0, min(rating, 1.0))
    except Exception as e:
        print(f"Scoring error: {e}")
        return 0.0


def parse_link(answer):
    """鎻愬彇鏂囨湰涓殑 URL"""
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

def judge_clarity(pred_answer):
    """
    鍒ゆ柇缁撴瀯鍖栧洖绛旀槸鍚︽竻鏅帮紙Clarity锛夛細
    鏀寔 markdown 鍖呰９锛坄``json ... ```锛?
    缁撴瀯涓彧鑳藉寘鍚?answer 鍜?reference 涓や釜瀛楁
    reference 鐨?key 浠?[1] 寮€濮嬭繛缁紪鍙凤紝value 蹇呴』鏄悎娉?aminer 閾炬帴
    answer 涓繀椤诲寘鍚笖浠呭寘鍚墍鏈夊紩鐢ㄧ紪鍙?
    闄?JSON / markdown 涔嬪涓嶅緱鏈夐澶栧唴瀹?
    杩斿洖锛? 锛堟竻鏅帮級 鎴?0 锛堜笉娓呮櫚锛?
    """

    def extract_json_str(result_str: str):
        """
        鎻愬彇 result 涓殑绾?JSON 瀛楃涓诧紝鍏佽 markdown 鍖呰９
        """
        result_str = result_str.strip()

        # ```json ... ``` 鎴?``` ... ``` 鍖呰９
        md_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", result_str, flags=re.IGNORECASE | re.DOTALL)
        if md_match:
            return md_match.group(1).strip()

        # 鐩存帴 JSON
        if result_str.startswith("{") and result_str.endswith("}"):
            return result_str

        return None

    def is_reference_valid(reference_dict):
        """
        鏍￠獙 reference锛?
        - 鏄?dict
        - key 浠?[1] 寮€濮嬭繛缁?
        - value 涓哄惈 aminer 閾炬帴鐨勫瓧绗︿覆
        """
        if not isinstance(reference_dict, dict) or not reference_dict:
            return False

        keys = list(reference_dict.keys())
        # 妫€鏌?key 鏍煎紡姝ｇ‘锛歔1], [2], ...
        if any(not re.fullmatch(r"\[\d+\]", k) for k in keys):
            return False

        # 妫€鏌?key 涓鸿繛缁紪鍙?
        nums = sorted(int(k.strip("[]")) for k in keys)
        if nums != list(range(1, len(nums) + 1)):
            return False

        # 妫€鏌?value 鍚堟硶
        return all(isinstance(v, str) and "https://www.aminer.cn" in v for v in reference_dict.values())

    def extract_citation_keys(text: str):
        return set(re.findall(r"\[\d+\]", text))

    # Step 1: 鑾峰彇 result 瀛楁
    summary_field = pred_answer.get("summary", "")
    result_str = str(summary_field).strip()
    if not result_str:
        return 0

    # Step 2: 鎻愬彇 JSON 涓讳綋
    json_str = extract_json_str(result_str)
    if not json_str:
        return 0

    # Step 3: 瑙ｆ瀽 JSON
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return 0

    # Step 4: 蹇呴』鍙惈 answer 鍜?reference
    if set(parsed.keys()) != {"answer", "reference"}:
        return 0

    answer = parsed["answer"]
    reference = parsed["reference"]

    # Step 5: reference 鍚堟硶鎬?
    if not is_reference_valid(reference):
        return 0

    # Step 6: answer 涓紩鐢ㄥ繀椤诲畬鍏ㄥ尮閰?reference
    cited, defined = extract_citation_keys(answer), set(reference.keys())
    if cited != defined:
        return 0

    # 鍏ㄩ儴閫氳繃
    return 1


def planning_edit_distance(pred_plan, gold_plan):
    # 确保 pred_plan 和 gold_plan 都是列表
    if not isinstance(pred_plan, list):
        pred_plan = []
    if not isinstance(gold_plan, list):
        gold_plan = []
    
    # 过滤掉非字典元素
    pred_plan = [p for p in pred_plan if isinstance(p, dict)]
    gold_plan = [g for g in gold_plan if isinstance(g, dict)]
    
    # 确保所有元素都有 'name' 字段
    pred_plan = [p for p in pred_plan if 'name' in p]
    gold_plan = [g for g in gold_plan if 'name' in g]
    
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
    """妫€鏌PI鎵ц鏄惁鎴愬姛锛屾鏌?execution_result 涓殑閿欒"""
    if not execution_result:
        return 0
    last_result = execution_result[-1] if isinstance(execution_result, list) else execution_result
    for api_name, result in last_result.items():
        if isinstance(result, dict):
            if result.get("error") or result.get("success") == False:
                return 0
    return 1


# ========== 鍚勭淮搴?Prompt 瀹氫箟 ==========

def make_prompt_correctness(question, gold, pred):
    return f"""
Rate the correctness of the predicted answer on a scale of [0, 1].
Return JSON only: {{"rating": <number>}}

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
"""


def make_prompt_integrality(question, gold, pred):
    return f"""
Rate the integrality (coverage of key points) of the predicted answer on [0, 1].
Return JSON only: {{"rating": <number>}}

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
"""


def make_prompt_completeness(question, pred, gold):
    return f"""
Rate whether the predicted answer fully completes the core objective of the question on [0, 1].
Return JSON only: {{"rating": <number>}}

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
"""


def make_prompt_faithfulness(pred, api_output):
    return f"""
Rate faithfulness of the predicted answer to the provided API output on [0, 1].
Return JSON only: {{"rating": <number>}}

Predicted answer: {pred}
API output: {api_output}
"""


# ========== 涓婚€昏緫 ==========

def main():
    global GLOBAL_ALIAS_LOOKUP

    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # Build bilingual alias lookup once and reuse in parameter-level matching logic.
    GLOBAL_ALIAS_LOOKUP = build_bilingual_alias_lookup(gold_data)
    print(f"Loaded bilingual alias entries: {len(GLOBAL_ALIAS_LOOKUP)}")

    gold_dict = {item["qid"]: item for item in gold_data}
    pred_dict = {item["id"]: item for item in pred_data}
    results, metrics = [], defaultdict(list)

    for pred_item in tqdm(pred_data, desc="Evaluating", ncols=100):
        qid = pred_item["id"]
        gold_item = gold_dict.get(qid)
        if not gold_item:
            print(f"鈿狅笍  鏈壘鍒伴棶棰?{qid} 鐨勬爣鍑嗙瓟妗堬紝璺宠繃璇勪及")
            continue
        question = gold_item.get("question", "")
        gold_ans = gold_item.get("result_edit", "")
        pred_ans = pred_item.get("summary", "")

        precision = judge_precision(gold_ans, pred_ans)
        recall = judge_recall(gold_ans, pred_ans)
        clarity = judge_clarity(pred_item)
        
        success = check_api_success(pred_item.get("execution_result", []))
        
        # 处理 plan 字段，可能是字符串或字典/列表
        plan_data = pred_item.get("plan", [])
        if isinstance(plan_data, str):
            try:
                pred_plan = json.loads(plan_data)
            except json.JSONDecodeError:
                pred_plan = []
        else:
            pred_plan = plan_data
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
        param_metrics = compute_param_metrics(gold_item, pred_item)
        partial_completion_score = compute_partial_completion_score(pred_item)

        eval_result = {
            "qid": qid,
            "precision": precision, "recall": recall, "clarity": clarity,
            "correctness": correctness, "integrality": integrality,
            "completeness": completeness, "faithfulness": faithfulness,
            "success": success, "edit_distance": edit_distance,
            "partial_completion_score": partial_completion_score,
            **param_metrics
        }

        for k, v in eval_result.items():
            if k != "qid":
                metrics[k].append(v)
        results.append(eval_result)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    average = {k: round(mean(v), 4) for k, v in metrics.items()}
    print("\n=== 骞冲潎鎸囨爣 ===")
    for k, v in average.items():
        print(f"{k}: {v}")

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(average, f, indent=2)


if __name__ == "__main__":
    main()

