PLAY2PROMPT - Interactive Tool-use Example Generation & Tool Documentation Optimization
===
Implementation for the paper [PLAY2PROMPT](https://aclanthology.org/2025.findings-acl.1347/).

## Installation
1. Install requirements in `requirements.txt`
2. Update submodules `git submodule update --init`

## API/Dataset Setup
1. Setup and install dataset and evaluation code (see `bfcl` for example). Also
   need to sign-up and set required API keys (eg `.env` in
   `bfcl/berkeley-function-call-leaderboard`).
2. Create script for loading dataset and API wrapper (for interacting with APIs). See `bfcl_api.py` for example.
3. Create script for evaluating generated examples/descriptions. Write newly
   generated examples/descriptions to files, run evaluation script, then read
   scores and results/error. See `bfcl_eval.py` for example.

## Run Optimization
This project was developed with IBM-hosted LLMs (mainly LLaMA models); to use, set/export the env variables `LLM_API_KEY` and `RITS_ENDPOINT`. To use vLLM or other API services, modify `rits.py`.

For BFCL, we also need to set:
```
SCRIPT_PATH=$(realpath "$0")
CUR_DIR=$(dirname "${SCRIPT_PATH}")
DATA_DIR=${CUR_DIR}/bfcl/berkeley-function-call-leaderboard
```

For example generation:
```
python main.py \
  --method example \
  --data_dir ${DATA_DIR} \
  --tmp_dir ${DATA_DIR}/data_tmp \
  --max_eval_threads 2 \
  --search_num_workers 5 \
  --gen_model_id meta-llama/Llama-3.1-8B-Instruct \
  --tool_model_id meta-llama/Llama-3.1-8B-Instruct \
  --save_dir outputs/generated_examples \
  --batch_size 10 \
  --expand_num 3 \
  --top_k 10 \
  --max_iterations 3 \
  --num_init_loop 50 \
  --num_feedback_steps 2 \
  --num_refine_steps 3 \
  --score_eval_weight 0.0 \
  --max_score 3.0 \
  --check_valid \
  --early_stop \
  $@
```

For documentation optimization:
```
python main.py \
  --method description \
  --data_dir ${DATA_DIR} \
  --tmp_dir ${DATA_DIR}/data_tmp \
  --gen_model_id meta-llama/Llama-3.1-8B-Instruct \
  --tool_model_id meta-llama/Llama-3.1-8B-Instruct \
  --examples_dir outputs/generated_examples \
  --save_dir outputs/generated_descriptions \
  --num_examples_for_desc 10 \
  --batch_size 5 \
  --expand_num 5 \
  --max_iterations 3 \
  --top_k 3 \
  --max_score 100 \
  --early_stop \
  $@
```

## Code Structure
- `main.py`: main optimization script
- `beam_search`: beam search framework
- `example_method.py`: defines a single search step for tool-use example
  optimization
- `description_method`: defines a single search step for description
  optimization
- `*_api.py`: dataset loading and dataset API wrapper
- `*_eval.py`: defines pipeline for evaluating performance on generated data for a dataset

## Citation
If you find our work helpful, please cite us as:
```
@inproceedings{fang-etal-2025-play2prompt,
    title = "{PLAY}2{PROMPT}: Zero-shot Tool Instruction Optimization for {LLM} Agents via Tool Play",
    author = "Fang, Wei  and
      Zhang, Yang  and
      Qian, Kaizhi  and
      Glass, James R.  and
      Zhu, Yada",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1347/",
    pages = "26274--26290",
    ISBN = "979-8-89176-256-5",
}
```
