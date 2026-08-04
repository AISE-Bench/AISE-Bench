"""Compute human-judge consistency metrics for AISE-Bench.

This script implements the metrics used by the paper section
"Human Judge Consistency":

- P-BT: Pearson correlation between judge scores and Bradley-Terry
  scores fitted from human pairwise preferences.
- PW-AUC: Pairwise AUC between judge score differences and human
  pairwise preferences.
- Avg: mean of P-BT and PW-AUC.

Input score files should contain one JSON object per line with at least:
  {"id": ..., "score": {"Correctness": {"rating": ...},
                        "Completeness": {"rating": ...}}}

The per-item judge score is the harmonic mean of Correctness and
Completeness, matching the existing project scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class PairwiseItem:
    qa_key: str
    model_a: str
    model_b: str
    better: str


def extract_json_object(text: str) -> Any:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
    return json.loads(text)


def get_rating(score_info: dict[str, Any], name: str) -> float:
    value = score_info.get(name, {})
    if isinstance(value, dict):
        value = value.get("rating", 0.0)
    return float(value or 0.0)


def harmonic_score(correctness: float, completeness: float) -> float:
    if correctness + completeness <= 0:
        return 0.0
    return 2 * correctness * completeness / (correctness + completeness)


def make_key(item: dict[str, Any], match_key: str) -> str:
    if match_key == "id":
        return str(item["id"])
    part_idx = item.get("part_idx", item.get("idx", 0))
    return f"{item['id']}_{part_idx}"


def load_score_file(path: str, match_key: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            score_info = item.get("score", {})
            if isinstance(score_info, str):
                score_info = extract_json_object(score_info)
            elif isinstance(score_info, list):
                merged: dict[str, Any] = {}
                for part in score_info:
                    if not part:
                        continue
                    if isinstance(part, str):
                        part = extract_json_object(part)
                    merged.update(part)
                score_info = merged

            if "Correctness" in score_info or "Completeness" in score_info:
                correctness = get_rating(score_info, "Correctness")
                completeness = get_rating(score_info, "Completeness")
            else:
                correctness = float(score_info.get("correctness", 0.0))
                completeness = float(score_info.get("completeness", 0.0))

            if item.get("predicted_answer", "non-empty") == "":
                scores[make_key(item, match_key)] = 0.0
            else:
                scores[make_key(item, match_key)] = harmonic_score(correctness, completeness)
    return scores


def load_pairwise_file(path: str, match_key: str) -> list[PairwiseItem]:
    items: list[PairwiseItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "pair" in item:
                model_a, model_b = item["pair"]
            else:
                model_a, model_b = item["file_a"], item["file_b"]
            better = item.get("better_model", item.get("better"))
            if not better:
                continue
            items.append(PairwiseItem(make_key(item, match_key), model_a, model_b, better))
    return items


def fit_bradley_terry(items: list[PairwiseItem], models: list[str]) -> dict[str, float]:
    model_to_idx = {model: i for i, model in enumerate(models)}
    n_models = len(models)
    wins = [0.01 for _ in range(n_models)]
    comparisons: list[tuple[int, int]] = []
    for item in items:
        if item.better.lower() == "tie":
            continue
        if item.model_a not in model_to_idx or item.model_b not in model_to_idx:
            continue
        loser = item.model_b if item.better == item.model_a else item.model_a
        if item.better not in model_to_idx or loser not in model_to_idx:
            continue
        winner_idx = model_to_idx[item.better]
        loser_idx = model_to_idx[loser]
        wins[winner_idx] += 1.0
        comparisons.append((winner_idx, loser_idx))

    if not comparisons:
        return {}

    abilities = [1.0 for _ in range(n_models)]
    for _ in range(1000):
        denom = [0.0 for _ in range(n_models)]
        for winner_idx, loser_idx in comparisons:
            total = abilities[winner_idx] + abilities[loser_idx]
            denom[winner_idx] += 1.0 / total
            denom[loser_idx] += 1.0 / total

        updated = [
            wins[i] / denom[i] if denom[i] > 0 else abilities[i]
            for i in range(n_models)
        ]
        avg_ability = mean(updated)
        updated = [value / avg_ability for value in updated]
        if max(abs(updated[i] - abilities[i]) for i in range(n_models)) < 1e-10:
            abilities = updated
            break
        abilities = updated

    scores = [math.log(value) for value in abilities]
    return dict(zip(models, scores))


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def binary_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(1 for label in y_true if label == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    indexed_scores = sorted(enumerate(y_score), key=lambda pair: pair[1])
    ranks = [0.0 for _ in y_score]
    i = 0
    while i < len(indexed_scores):
        j = i
        while j + 1 < len(indexed_scores) and indexed_scores[j + 1][1] == indexed_scores[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed_scores[k][0]] = avg_rank
        i = j + 1

    positive_rank_sum = sum(rank for rank, label in zip(ranks, y_true) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_pairwise_auc(
    items: list[PairwiseItem],
    model_scores: dict[str, dict[str, float]],
) -> float | None:
    y_true: list[int] = []
    y_score: list[float] = []

    for item in items:
        if item.better.lower() == "tie":
            continue
        if item.model_a not in model_scores or item.model_b not in model_scores:
            continue
        scores_a = model_scores[item.model_a]
        scores_b = model_scores[item.model_b]
        if item.qa_key not in scores_a or item.qa_key not in scores_b:
            continue

        y_true.append(1 if item.better == item.model_a else 0)
        y_score.append(scores_a[item.qa_key] - scores_b[item.qa_key])

    return binary_auc(y_true, y_score)


def compute_metrics(
    pairwise_items: list[PairwiseItem],
    model_scores: dict[str, dict[str, float]],
) -> dict[str, float | int | None]:
    models = sorted(model_scores)
    bt_scores = fit_bradley_terry(pairwise_items, models)
    avg_scores = {
        model: float(mean(list(scores.values())))
        for model, scores in model_scores.items()
        if scores
    }

    common = [model for model in models if model in bt_scores and model in avg_scores]
    if len(common) >= 2 and len(set(avg_scores[m] for m in common)) > 1:
        p_bt = pearson_corr([avg_scores[m] for m in common], [bt_scores[m] for m in common])
    else:
        p_bt = None

    pw_auc = compute_pairwise_auc(pairwise_items, model_scores)
    values = [value for value in [p_bt, pw_auc] if value is not None and not math.isnan(value)]
    avg = float(mean(values)) if values else None

    matched_pairwise = 0
    for item in pairwise_items:
        if item.model_a in model_scores and item.model_b in model_scores:
            if item.qa_key in model_scores[item.model_a] and item.qa_key in model_scores[item.model_b]:
                matched_pairwise += 1

    return {
        "P-BT": p_bt,
        "PW-AUC": pw_auc,
        "Avg": avg,
        "models": len(models),
        "pairwise_rows": len(pairwise_items),
        "matched_pairwise_rows": matched_pairwise,
    }


def parse_score_files(items: list[str], match_key: str) -> dict[str, dict[str, float]]:
    model_scores: dict[str, dict[str, float]] = {}
    for item in items:
        if "=" in item:
            model, path = item.split("=", 1)
        else:
            path = item
            model = os.path.basename(path).split("-soay_eval")[0].removesuffix(".jsonl")
        model_scores[model] = load_score_file(path, match_key)
    return model_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise", required=True, help="Human pairwise annotation JSONL.")
    parser.add_argument(
        "--score-files",
        nargs="+",
        required=True,
        help="Score files as model=path. Model names must match pairwise names.",
    )
    parser.add_argument(
        "--match-key",
        choices=["id", "id_part"],
        default="id_part",
        help="Use id_part for exact part matching, or id when pairwise idx is not part_idx.",
    )
    parser.add_argument("--output", help="Optional CSV output path.")
    args = parser.parse_args()

    pairwise_items = load_pairwise_file(args.pairwise, args.match_key)
    model_scores = parse_score_files(args.score_files, args.match_key)
    metrics = compute_metrics(pairwise_items, model_scores)

    row = {
        key: (round(value, 4) if isinstance(value, float) else value)
        for key, value in metrics.items()
    }
    fieldnames = list(row)
    print(" ".join(f"{name:>22}" for name in fieldnames))
    print(" ".join(f"{str(row[name]):>22}" for name in fieldnames))
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
