<div align="center">

# AISE-Bench: A Full-Cycle Curated Benchmark for Information Seeking on Academic Knowledge Graphs

</div>

<p align="center">
    🌐 <a href="https://aise-bench.github.io/" target="_blank">Project Page</a> •
    📖 <a href="https://arxiv.org/abs/2607.20498" target="_blank">KDD 2026 Paper</a> •
    🤗 <a href="https://huggingface.co/datasets/zhengyang6666/AISE-Bench" target="_blank">Hugging Face</a> •
</p>

<div align="center">
    <img src=assets/bench.png width=100% />
</div>

*Official code and dataset of the paper AISE-Bench: A Full-Cycle Curated Benchmark for Information Seeking on Academic Knowledge Graphs (KDD 2026).*

***

AISE-Bench is a real-world, full-cycle annotated benchmark built on authentic AMiner user search queries, containing 1,133 human-verified academic QA pairs with executable multi-step API trajectories, standardized query taxonomies, parameter-validated API calls and source-grounded answers with canonical reference links. It supports end-to-end evaluation of LLM tool agents across API planning, parameter filling, multi-step execution and reference grounding.

## 🚀 Quick Start

### Dependencies

First, create a conda environment and install all pip package requirements.
```bash
conda create -n aise python==3.11.13
conda activate aise

pip install -r requirements.txt
```


### Download from Hugging Face
```bash
pip install -U huggingface_hub
hf download AISE-Bench/AISE-Bench data/ --repo-type dataset --local-dir ./
```

### Processed Data Download
You may also download full preprocessed benchmark data directly from Hugging Face.
The processed data includes:

- `id/`: Unique serial number for each sample
- `quetsion/`: 	Original real academic search query submitted by AMiner users
- `planning_text/`:  Gold-standard multi-step API planning sequence, records the required tool calling order
- `api_input/`: Standardized input parameters for each API (author name, search keywords, etc.)
- `api_output/`: 	Return results from AMiner academic KG API, containing entity IDs, execution status and prompt message
- `result_edit/`: Human-written final answer grounded by reference citations [1]

### CAW Annotation Pipeline
The framework/ directory implements the Customized Agent Workflow (CAW) annotation system for generating gold API trajectories and grounded answers from raw user queries.
All benchmark data is stored in the data/ folder, covering raw queries, multi-dimensional taxonomy labels, gold API execution paths and official test split.
CAW Auto Annotation: Run planner-executor-synthesizer pipeline to generate standard API chains and answers
```bash
python CAW/from_plan_to_result.py
```


### 🧩 Consistency Evaluation
The consistency/ directory provides standalone scripts to compute alignment metrics between LLM judge scores and human pairwise preferences (P-BT, PW-AUC).
See consistency/README.md for details.
## ✈️ Inference
We reproduce 14 mainstream LLM agent frameworks (AvaTaR, CodeAct, DRAFT, PLAY2PROMPT, SoAy, etc.) as baseline implementations. Gemini-3-Pro is used as the base example, you can replace it with any supported LLM.
## Evaluation
After generating model prediction files, run comprehensive multi-dimensional evaluation script to calculate all process & answer metrics:
```bash
python eval.py
```
