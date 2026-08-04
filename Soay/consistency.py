import json, re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import accuracy_score, roc_auc_score
import choix
import glob
import os

def escape_latex(json_str):
    """转义 JSON 中的裸反斜杠"""
    json_str = re.sub(r'(?<!\\)\\(?![bfnrtu\\"\'\\])', r'\\\\', json_str)
    return json_str

MODEL_MAP = {
    "glm4": "glm",
    "qwen3": "qwen",
    "gemini": "gemini-3-pro-preview",
    "deepseek": "firedeepseek-v3.2",
    "qwen3-235b-a22b": "qwen3-235b-a22b"
}

def map_model_name(model_name: str) -> str:
    # 直接映射常见模型名
    model_mapping = {
        'gemini': 'gemini-3-pro-preview',
        'qwen3-235b-a22b': 'qwen3-235b-a22b',
        'deepseek': 'firedeepseek-v3.2'
    }
    if model_name in model_mapping:
        return model_mapping[model_name]
    # 对于其他模型名，使用原映射
    for prefix, target in MODEL_MAP.items():
        if model_name.lower().startswith(prefix):
            return target
    return model_name

# ========= 数据读取函数 =========

def load_score_file(file_path):
    """读取自动打分文件，返回 {模型: {qa_id: f1_score}}"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 使用os.path处理路径，兼容Windows和Linux
    base_name = os.path.basename(file_path)
    # 移除文件扩展名
    if base_name.endswith('.jsonl'):
        base_name = base_name[:-6]  # 移除.jsonl扩展名
    # 提取模型名，处理包含多个下划线的情况
    if 'soay_eval' in base_name:
        # 对于soay_eval_results_filled.jsonl格式的文件
        model_name = base_name.split('-soay_eval')[0]
    else:
        # 对于其他格式的文件
        model_name = base_name.split('_')[0]
    model_name = map_model_name(model_name)
    print(f"Extracted model name: {model_name}")

    scores = defaultdict(dict)
    for item in data:
        uid = f"{item['id']}_{item['part_idx']}"
        content = item["score"]
        if item["predicted_answer"] == "":
            scores[model_name][uid] = 0.0
            continue
        if isinstance(content, list):
            # 如果是多段组成的 list
            correctness, completeness = 0.0, 0.0
            for c in content:
                print(c)
                print(file_path)
                if c == "": continue
                if isinstance(c, str):
                    if "```json" in c:
                        pattern = r'```json(.*)```'
                        c = re.search(pattern, c, re.DOTALL).group(1)
                        c = escape_latex(c)
                    score_info = json.loads(c, strict=False)
                else:
                    # 如果c已经是字典
                    score_info = c
                if "Correctness" in score_info:
                    correctness = float(score_info["Correctness"]["rating"])
                if "Completeness" in score_info:
                    completeness = float(score_info["Completeness"]["rating"])
        else:
            print(content)
            print(file_path)
            if isinstance(content, str):
                if "```json" in content:
                    pattern = r'```json(.*)```'
                    content = re.search(pattern, content, re.DOTALL).group(1)
                    content = escape_latex(content)
                score_info = json.loads(content, strict=False)
            else:
                # 如果content已经是字典
                score_info = content
            correctness = float(score_info["Correctness"]["rating"])
            completeness = float(score_info["Completeness"]["rating"])

        if correctness + completeness > 0:
            f1_score = 2 * correctness * completeness / (correctness + completeness)
        else:
            f1_score = 0.0

        scores[model_name][uid] = f1_score

    return scores


def load_pairwise_file(file_path):
    """
    返回：
    - pairs_bt: list[(winner, loser)]（方法1用BT拟合）
    - win_counts: {model: 胜场数}（方法2用）
    - pairwise_items: list[dict]（方法3逐题计算）
    - models: list
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    win_counts = defaultdict(float)
    pairs_bt = []
    pairwise_items = []
    models = set()

    for item in data:
        # 处理不同格式的pairwise文件
        if "file_a" in item:
            # 旧格式：有file_a和file_b字段
            a = item["file_a"]
            b = item["file_b"]
        elif "pair" in item:
            # 新格式：有pair字段，是一个包含两个模型名的列表
            a = item["pair"][0]
            b = item["pair"][1]
        else:
            continue  # 跳过不支持的格式
        
        # 获取better字段
        if "better" in item:
            better = item["better"]
        elif "better_model" in item:
            better = item["better_model"]
        else:
            continue  # 跳过不支持的格式
        
        # 获取id和part_idx
        if "part_idx" in item:
            qa_id = f"{item['id']}_{item['part_idx']}"
        elif "idx" in item:
            qa_id = f"{item['id']}_{item['idx']}"
        else:
            qa_id = f"{item['id']}_0"

        models.update([a, b])

        if better.lower() == "tie":
            win_counts[a] += 0.5
            win_counts[b] += 0.5
            # tie 不加入BT的赢-输对
        else:
            win_counts[better] += 1
            loser = b if better == a else a
            pairs_bt.append((better, loser))

        pairwise_items.append({
            "model_a": a,
            "model_b": b,
            "better": better,
            "qa_id": qa_id
        })

    return pairs_bt, win_counts, pairwise_items, list(models)


# ========= 方法实现 =========

def fit_bt_scores(pairs_bt, models):
    """Bradley–Terry 拟合全局分"""
    model_to_idx = {m: i for i, m in enumerate(models)}
    bt_data = [(model_to_idx[w], model_to_idx[l]) for w, l in pairs_bt]
    scores = choix.ilsr_pairwise(len(models), bt_data, alpha=0.01)
    return pd.Series(scores, index=models)


def method3_pairwise_metrics(pairwise_items, model_item_scores_for_setting):
    """
    方法三：题目级别 pairwise 一致性指标
      - Accuracy
      - ROC AUC
      - Kendall's tau-like
    """
    y_true, y_score = [], []
    concordant, discordant = 0, 0

    for item in pairwise_items:
        if item["better"].lower() == "tie":
            continue
        m_a = item["model_a"]
        m_b = item["model_b"]
        qa_id = item["qa_id"]

        if qa_id not in model_item_scores_for_setting.get(m_a, {}) or \
           qa_id not in model_item_scores_for_setting.get(m_b, {}):
            continue

        score_a = model_item_scores_for_setting[m_a][qa_id]
        score_b = model_item_scores_for_setting[m_b][qa_id]
        diff = score_a - score_b

        # 原精度/ROC
        y_score.append(diff)
        y_true.append(1 if item["better"] == m_a else 0)

        # tau-like
        human_pref = 1 if item["better"] == m_a else -1
        llm_pref = 1 if diff > 0 else -1 if diff < 0 else 0
        if llm_pref == 0:
            continue  # LLM 平局不计入 tau-like
        if llm_pref == human_pref:
            concordant += 1
        else:
            discordant += 1

    if not y_true:
        return None, None, None

    acc = accuracy_score(y_true, [1 if s > 0 else 0 for s in y_score])
    auc = roc_auc_score(y_true, y_score)
    tau_like = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else None

    return acc, auc, tau_like


def compare_versions(score_files_list, pairwise_file):
    # 读入人类数据
    pairs_bt, win_counts, pairwise_items, models = load_pairwise_file(pairwise_file)
    human_bt_scores = fit_bt_scores(pairs_bt, models)

    results = []
    
    # 遍历每个评估设定
    for setting_name, files in score_files_list.items():
        model_item_scores = {}
        for file in files:
            scores = load_score_file(file)
            model_item_scores.update(scores)

        # 方法1/2: 模型级分数
        avg_scores = {m: np.mean(list(scores.values()))
                      for m, scores in model_item_scores.items()}

        # 只保留在两个数据集中都存在的模型
        common_models = [m for m in models if m in avg_scores]
        print(f"Common models: {common_models}")

        # 方法一：Pearson/Spearman/Kendall vs BT 分数
        if len(common_models) >= 2:
            pearson_bt, _ = pearsonr([avg_scores[m] for m in common_models],
                                     [human_bt_scores[m] for m in common_models])
            spearman_bt, _ = spearmanr([avg_scores[m] for m in common_models],
                                       [human_bt_scores[m] for m in common_models])
            kendall_bt, _ = kendalltau([avg_scores[m] for m in common_models],
                                       [human_bt_scores[m] for m in common_models])
        else:
            pearson_bt, spearman_bt, kendall_bt = None, None, None

        # 方法二：Spearman/Kendall vs 胜场数
        if len(common_models) >= 2:
            spear_win, _ = spearmanr([avg_scores[m] for m in common_models],
                                     [win_counts[m] for m in common_models])
            kend_win, _ = kendalltau([avg_scores[m] for m in common_models],
                                     [win_counts[m] for m in common_models])
        else:
            spear_win, kend_win = None, None

        # 方法三：题目级 pairwise 一致性
        acc, auc, tau_like = method3_pairwise_metrics(pairwise_items, model_item_scores)

        results.append({
            'setting': setting_name,
            'Pearson_BT': pearson_bt,
            'Spearman_BT': spearman_bt,
            'Kendall_BT': kendall_bt,
            'Spearman_win': spear_win,
            'Kendall_win': kend_win,
            'Pairwise_acc': acc,
            'Pairwise_auc': auc,
            'Pairwise_tau_like': tau_like
        })

    results_df = pd.DataFrame(results)
    # 遍历数值列四舍五入到四位
    num_cols = results_df.select_dtypes(include=['float', 'float64']).columns
    results_df[num_cols] = results_df[num_cols].round(4)

    # ====== 计算综合分并选优 ======
    core_metrics = [
        'Spearman_BT', 'Kendall_BT',
        'Spearman_win', 'Kendall_win',
        'Pairwise_acc', 'Pairwise_auc', 'Pairwise_tau_like'
    ]

    # 将 [-1,1] 范围的指标映射到 [0,1]
    range_01_metrics = ['Spearman_BT', 'Kendall_BT', 'Spearman_win', 'Kendall_win', 'Pairwise_tau_like']

    for col in range_01_metrics:
        results_df[col] = (results_df[col] + 1) / 2  # 假设理论范围 [-1, 1]

    # Acc / AUC 已经是 0-1，无需处理
    # 计算综合分
    results_df['consistency_score'] = results_df[core_metrics].mean(axis=1)

    best_idx = results_df['consistency_score'].idxmax()
    best_setting_row = results_df.loc[best_idx]

    # # 保存结果
    # with open("result_consist.json", 'w', encoding='utf-8') as f:
    #     json.dump(results_df.to_dict(orient='records'), f, indent=2, ensure_ascii=False)
    
    results_df.to_csv("result_consist_old.csv", index=False, encoding='utf-8-sig')

    return results_df, best_setting_row


def build_score_files_list(base_dir):
    """
    base_dir: 设置所在的根目录
        ├─ setting1/
        │    ├─ deepseek_score.json
        │    ├─ glm4_score.json
        │    └─ qwen3_score.json
        ├─ setting2/
        │    ├─ ...
        └─ setting3/...
    返回: { setting_name: [文件路径, 文件路径, ...] }
    """
    score_files_list = {}

    # 遍历 base_dir 下的所有子目录
    for setting_dir in sorted(os.listdir(base_dir)):
        setting_path = os.path.join(base_dir, setting_dir)
        # if setting_path in ["/workspace/yelin/openreview/output/abs+single", "/workspace/yelin/openreview/output/abs+together", "/workspace/yelin/openreview/output/noabs+single"]: continue
        if os.path.isdir(setting_path):
            # 找到该 setting 下的所有 json 和 jsonl 文件
            json_files = sorted(glob.glob(os.path.join(setting_path, "*.json")))
            jsonl_files = sorted(glob.glob(os.path.join(setting_path, "*.jsonl")))
            all_files = json_files + jsonl_files
            if all_files:  # 至少有一个文件
                score_files_list[setting_dir] = all_files

    return score_files_list

if __name__ == "__main__":

    # ============ 调用实例 ============
    #score_files_list = build_score_files_list("/workspace/yelin/openreview/output")

    # 使用提供的输入文件
    score_files_list = {
          "soay_eval": [
              "e:\\ZHIPU AI\\Code reproduction\\1.17-soay\\gemini-soay_eval_results_filled.jsonl",
              "e:\\ZHIPU AI\\Code reproduction\\1.17-soay\\qwen3-235b-a22b-soay_eval_results_filled.jsonl",
              "e:\\ZHIPU AI\\Code reproduction\\1.17-soay\\deepseek-soay_eval_results_filled.jsonl"
          ]
    }

    # 使用提供的pairwise文件
    pairwise_file = "e:\\ZHIPU AI\\Code reproduction\\1.17-soay\\closest_three_models_pair_annotations.jsonl"

    df_results, best_setting_row = compare_versions(score_files_list, pairwise_file)
    print(df_results)

    print("\n===== 最优一致性设置 =====")
    print(f"设置名: {best_setting_row['setting']}")
    print(f"综合一致性分数: {best_setting_row['consistency_score']:.4f}")
    print("各指标:")
    print(best_setting_row.to_string())