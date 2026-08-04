# Human Judge Consistency

This folder contains the GitHub-ready code for the paper section
`4.4 Human Judge Consistency`.

It computes:

- `P-BT`: fit a Bradley-Terry model from human pairwise preferences, then
  compute Pearson correlation between those BT scores and judge scores.
- `PW-AUC`: pairwise AUC using judge score differences against human
  pairwise preferences.
- `Avg`: mean of `P-BT` and `PW-AUC`.

## Run

Install dependencies:

```bash
No external dependency is required.
```

Example with the local files in this workspace:

```bash
python scoring_consistency/score_consistency.py \
  --pairwise closest_three_models_pair_annotations.jsonl \
  --score-files \
    gemini-3-pro-preview=gemini-soay_eval_results_filled.jsonl \
    qwen3-235b-a22b=qwen3-235b-a22b-soay_eval_results_filled.jsonl \
    firedeepseek-v3.2=deepseek-soay_eval_results_filled.jsonl \
  --match-key id \
  --output scoring_consistency/result_consistency.csv
```

Use `--match-key id_part` when the human pairwise file and score files share
the same `id` plus `part_idx` keys. Use `--match-key id` when the pairwise
file's `idx` field is an annotation index rather than the answer part index.
