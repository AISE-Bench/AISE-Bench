import json
import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import accuracy_score, roc_auc_score
import choix
import pandas as pd

# 读取并解析模型得分文件
def load_score_file(file_path):
    """读取自动打分文件，返回 {模型: {qa_id: f1_score}}"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 从文件名中提取模型名称
    # 例如：gemini-soay_eval_results.jsonl -> gemini
    # 例如：qwen3-235b-a22b-soay_eval_results.jsonl -> qwen3-235b-a22b
    file_name = os.path.basename(file_path)
    model_name = file_name.split('-')[0]
    
    scores = {}
    model_scores = {}
    
    for item in data:
        uid = f"{item['id']}_{item['part_idx']}"
        content = item["score"]
        
        # 计算F1分数
        if isinstance(content, dict) and "Correctness" in content and "Completeness" in content:
            correctness = float(content["Correctness"]["rating"])
            completeness = float(content["Completeness"]["rating"])
            if correctness + completeness > 0:
                f1_score = 2 * correctness * completeness / (correctness + completeness)
            else:
                f1_score = 0.0
        else:
            f1_score = 0.0
        
        model_scores[uid] = f1_score
    
    scores[model_name] = model_scores
    return scores

# 读取并解析pairwise文件
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

    win_counts = {}
    pairs_bt = []
    pairwise_items = []
    models = set()

    for item in data:
        a = item["file_a"]
        b = item["file_b"]
        better = item["better"]
        qa_id = f"{item['id']}_{item['part_idx']}"

        models.update([a, b])

        if better.lower() == "tie":
            win_counts[a] = win_counts.get(a, 0) + 0.5
            win_counts[b] = win_counts.get(b, 0) + 0.5
            # tie 不加入BT的赢-输对
        else:
            win_counts[better] = win_counts.get(better, 0) + 1
            loser = b if better == a else a
            pairs_bt.append((better, loser))

        pairwise_items.append({
            "model_a": a,
            "model_b": b,
            "better": better,
            "qa_id": qa_id
        })

    return pairs_bt, win_counts, pairwise_items, list(models)

# Bradley–Terry 拟合全局分
def fit_bt_scores(pairs_bt, models):
    """Bradley–Terry 拟合全局分"""
    model_to_idx = {m: i for i, m in enumerate(models)}
    bt_data = [(model_to_idx[w], model_to_idx[l]) for w, l in pairs_bt]
    scores = choix.ilsr_pairwise(len(models), bt_data, alpha=0.01)
    return pd.Series(scores, index=models)

# 方法三：题目级别 pairwise 一致性指标
def method3_pairwise_metrics(pairwise_items, model_item_scores):
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

        # 查找模型得分
        score_a = 0.0
        score_b = 0.0
        
        # 尝试匹配模型名称（处理前缀匹配）
        for model_name, scores in model_item_scores.items():
            if model_name in m_a and qa_id in scores:
                score_a = scores[qa_id]
            if model_name in m_b and qa_id in scores:
                score_b = scores[qa_id]

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

# 比较不同评估设定
def compare_versions(score_files_list, pairwise_file):
    # 读入人类数据
    pairs_bt, win_counts, pairwise_items, all_models = load_pairwise_file(pairwise_file)
    
    results = []
    
    # 遍历每个评估设定
    for setting_name, files in score_files_list.items():
        model_item_scores = {}
        for file in files:
            scores = load_score_file(file)
            model_item_scores.update(scores)
        
        # 获取当前设定中可用的模型
        available_models = list(model_item_scores.keys())
        print(f"当前设定 '{setting_name}' 可用模型: {available_models}")
        
        if not available_models:
            print(f"警告: 设定 '{setting_name}' 没有可用模型，跳过")
            continue
        
        # 过滤人类数据，只保留可用模型的比较
        filtered_pairs_bt = []
        filtered_win_counts = {}
        filtered_pairwise_items = []
        
        for item in pairwise_items:
            m_a = item["model_a"]
            m_b = item["model_b"]
            
            # 只保留两个模型都可用的比较
            a_available = any(model in m_a for model in available_models)
            b_available = any(model in m_b for model in available_models)
            
            if a_available and b_available:
                filtered_pairwise_items.append(item)
                
                # 更新胜场数
                better = item["better"]
                better_available = any(model in better for model in available_models)
                if better_available:
                    if better not in filtered_win_counts:
                        filtered_win_counts[better] = 0
                    filtered_win_counts[better] += 1
                    
                    # 更新BT对
                    loser = m_b if better == m_a else m_a
                    filtered_pairs_bt.append((better, loser))
        
        # 如果没有足够的比较数据，跳过
        if not filtered_pairs_bt:
            print(f"警告: 设定 '{setting_name}' 没有足够的比较数据，跳过")
            continue
        
        # 拟合BT分数
        # 提取所有在filtered_pairs_bt中出现的模型
        bt_models = set()
        for w, l in filtered_pairs_bt:
            bt_models.add(w)
            bt_models.add(l)
        bt_models = list(bt_models)
        
        human_bt_scores = fit_bt_scores(filtered_pairs_bt, bt_models)
        
        # 方法1/2: 模型级分数
        avg_scores = {}
        for m, scores in model_item_scores.items():
            if scores:
                avg_scores[m] = np.mean(list(scores.values()))
            else:
                avg_scores[m] = 0.0

        # 准备模型级比较数据
        model_level_data = []
        for model in available_models:
            # 查找对应的BT分数
            bt_score = 0.0
            for bt_model in bt_models:
                if model in bt_model:
                    bt_score = human_bt_scores[bt_model]
                    break
            
            # 查找对应的胜场数
            win_count = 0
            for wc_model, count in filtered_win_counts.items():
                if model in wc_model:
                    win_count = count
                    break
            
            model_level_data.append({
                'model': model,
                'avg_score': avg_scores[model],
                'bt_score': bt_score,
                'win_count': win_count
            })
        
        # 方法一：Pearson/Spearman/Kendall vs BT 分数
        if len(model_level_data) >= 2:
            pearson_bt, _ = pearsonr(
                [d['avg_score'] for d in model_level_data],
                [d['bt_score'] for d in model_level_data]
            )
            spearman_bt, _ = spearmanr(
                [d['avg_score'] for d in model_level_data],
                [d['bt_score'] for d in model_level_data]
            )
            kendall_bt, _ = kendalltau(
                [d['avg_score'] for d in model_level_data],
                [d['bt_score'] for d in model_level_data]
            )
        else:
            pearson_bt = spearman_bt = kendall_bt = None

        # 方法二：Spearman/Kendall vs 胜场数
        if len(model_level_data) >= 2:
            spear_win, _ = spearmanr(
                [d['avg_score'] for d in model_level_data],
                [d['win_count'] for d in model_level_data]
            )
            kend_win, _ = kendalltau(
                [d['avg_score'] for d in model_level_data],
                [d['win_count'] for d in model_level_data]
            )
        else:
            spear_win = kend_win = None

        # 方法三：题目级别 pairwise 一致性指标
        acc, auc, tau_like = method3_pairwise_metrics(filtered_pairwise_items, model_item_scores)

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

    if not results:
        print("错误: 没有有效的评估结果")
        return None, None

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
        if col in results_df.columns:
            results_df[col] = (results_df[col] + 1) / 2  # 假设理论范围 [-1, 1]

    # Acc / AUC 已经是 0-1，无需处理
    # 计算综合分
    results_df['consistency_score'] = results_df[core_metrics].mean(axis=1)

    best_idx = results_df['consistency_score'].idxmax()
    best_setting_row = results_df.loc[best_idx]

    results_df.to_csv("result_consist_old.csv", index=False, encoding='utf-8-sig')

    return results_df, best_setting_row

# 主函数
if __name__ == "__main__":
    # 过滤pairwise文件，只保留用户有得分文件的模型
    input_pairwise_file = 'closest_three_models_pair_annotations_converted.jsonl'
    output_pairwise_file = 'closest_three_models_pair_annotations_filtered.jsonl'
    
    # 用户拥有的模型得分文件（根据用户提供的信息）
    available_models = {'deepseek', 'qwen3', 'gemini'}
    
    # 读取并过滤pairwise文件
    with open(input_pairwise_file, 'r', encoding='utf-8') as f_in, open(output_pairwise_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            file_a = data['file_a']
            file_b = data['file_b']
            
            # 检查两个模型是否都在可用模型列表中
            # 注意：我们需要匹配模型名称的前缀
            a_available = any(model in file_a.lower() for model in available_models)
            b_available = any(model in file_b.lower() for model in available_models)
            
            if a_available and b_available:
                f_out.write(line + '\n')
    
    print(f"过滤完成，输出文件: {output_pairwise_file}")
    
    # 现在运行一致性评估
    print("\n开始运行一致性评估...")
    print("注意：只评估用户提供的模型")
    
    # 设置得分文件列表（使用填充后的文件）
    score_files_list = {
        "soay_eval": [
            "./gemini-soay_eval_results_filled.jsonl",
            "./qwen3-235b-a22b-soay_eval_results_filled.jsonl",
            "./deepseek-soay_eval_results_filled.jsonl"
        ]
    }
    
    # 运行评估
    df_results, best_setting_row = compare_versions(score_files_list, output_pairwise_file)
    
    if df_results is not None:
        print("\n评估结果:")
        print(df_results)
        
        if best_setting_row is not None:
            print("\n===== 最优一致性设置 =====")
            print(f"设置名: {best_setting_row['setting']}")
            print(f"综合一致性分数: {best_setting_row['consistency_score']:.4f}")
    else:
        print("评估失败，可能是因为没有足够的有效数据")
