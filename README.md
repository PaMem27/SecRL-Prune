# SecRL-Prune

Official implementation of **SecRL-Prune: Structured Reinforcement Learning-Based Pruning of CodeLLMs for Preserving Adversarial Code Mutation**.

SecRL-Prune learns structured feed-forward-network (FFN) channel masks for CodeLLMs using a reinforcement-learning pruning policy. The policy is trained with a teacher-preservation reward based on forward KL divergence between a frozen dense teacher and a pruned student over cached teacher Top-K next-token distributions.

> **Status:** Research code. The current implementation focuses on LLaMA/CodeLLaMA-style Hugging Face causal language models with MLP modules containing `gate_proj`, `up_proj`, and `down_proj`.

---

## Repository structure

```text
SecRL-Prune/
├── secrl_prune/
│   ├── __init__.py
│   ├── kd_dataset.py          # Load and validate cached teacher Top-K KD records
│   ├── kd_penalty.py          # Forward KL over teacher Top-K + OTHER bucket
│   ├── pruned_view.py         # Runtime FFN channel-mask view for LLaMA-like MLPs
│   └── pruner.py              # PruneNet-style RL policy over FFN channel groups
├── scripts/
│   ├── teacher_topk.py        # Collect teacher Top-K targets from prompts
│   ├── train_pruner_min.py    # Train the pruning policy with REINFORCE + KD reward
│   └── export_pruned_model.py # Export a structurally pruned Hugging Face model
├── data/
│   └── dataset.example.jsonl  # Small example prompt dataset
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

---

## Method overview

SecRL-Prune runs in three stages:

1. **Teacher target collection**
   - Run a dense teacher CodeLLM on a calibration prompt dataset.
   - Cache the teacher's Top-K next-token probabilities at each generation step.
   - Store raw, unnormalized Top-K probabilities so the missing probability mass can be treated as an **OTHER** bucket.

2. **Policy learning**
   - Attach a runtime FFN masking view to the frozen student model.
   - Sample structured channel masks from the pruning policy.
   - Measure teacher-student divergence using forward KL on `Top-K ∪ OTHER`.
   - Update the policy with REINFORCE so masks with lower KL and the desired keep ratio become more likely.

3. **Structural export**
   - Convert the learned mask into real FFN weight slicing.
   - Save the resulting pruned model in Hugging Face format.

---

## Installation

Create a fresh environment:

```bash
git clone https://github.com/<your-username>/SecRL-Prune.git
cd SecRL-Prune

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For gated Hugging Face models, log in first:

```bash
huggingface-cli login
```

---

## Input dataset format

The calibration dataset is a JSONL file with one prompt per line:

```json
{"task_id": "601", "prompt": "def max_chain_length(arr, n):\n    # Write a function to find the longest chain which can be formed from the given set of pairs.\n    # Write your code below\n"}
```

Required fields:

| Field | Description |
|---|---|
| `task_id` | Unique problem or prompt identifier |
| `prompt` | Code-generation prompt passed to the teacher model |

A small example file is provided at:

```text
data/dataset.example.jsonl
```

---

## Step 1: Collect teacher Top-K targets

Run the dense teacher once and cache its Top-K output distribution:

```bash
python scripts/teacher_topk.py \
  --dataset data/dataset.example.jsonl \
  --out runs/kd_targets.jsonl \
  --solutions_dir runs/teacher_solutions \
  --model codellama/CodeLlama-7b-Python-hf \
  --top_k 124 \
  --max_new_tokens 192 \
  --temperature 1.0 \
  --device cuda
```

This creates:

```text
runs/kd_targets.jsonl
runs/teacher_solutions/
```

Each KD target row contains:

| Field | Description |
|---|---|
| `prompt` | Input prompt |
| `continuation` | Greedy teacher continuation |
| `topk_ids` | Teacher Top-K token IDs per generation step |
| `topk_probs` | Raw teacher Top-K probabilities per generation step |
| `topk_logprobs` | Log-probabilities for the same tokens |
| `input_len` | Prompt length in tokens |
| `gen_len` | Number of generated KD steps |

---

## Step 2: Sanity-check the KD dataset

Before training, verify that the cached teacher records are readable and that Top-K mass is valid:

```bash
python -m secrl_prune.kd_dataset \
  --kd_targets_path runs/kd_targets.jsonl \
  --model_name_or_path codellama/CodeLlama-7b-Python-hf \
  --topk 124
```

Expected behavior:

- `sum(topK) <= 1.0`
- `OTHER = 1 - sum(topK)` may be positive
- warnings may appear if tokenizer settings differ from collection time

---

## Step 3: Train the pruning policy

Train a policy that samples FFN channel masks and receives reward from teacher preservation:

```bash
python scripts/train_pruner_min.py \
  --kd_targets_path runs/kd_targets.jsonl \
  --model_name_or_path codellama/CodeLlama-7b-Python-hf \
  --out_dir runs/secrl_prune_20p \
  --steps 100 \
  --batch_size 8 \
  --lr 3e-4 \
  --temperature 1.0 \
  --topk 124 \
  --max_gen_steps 64 \
  --keep_target 0.8 \
  --sparsity_coef 50.0 \
  --group_size 16 \
  --seed 13
```

Here, `keep_target=0.8` means the policy keeps about 80% of FFN channels and prunes about 20%.

Training writes:

```text
runs/secrl_prune_20p/pruner_policy.pt
runs/secrl_prune_20p/train_log.jsonl
```

The log file contains per-step values such as average KL, average keep ratio, policy loss, and best reward.

---

## Step 4: Export a structurally pruned model

After policy training, convert the learned mask into an actual pruned Hugging Face model:

```bash
python scripts/export_pruned_model.py \
  --model_name_or_path codellama/CodeLlama-7b-Python-hf \
  --out_dir runs/codellama_7b_secrl_prune_20p \
  --keep_target 0.8 \
  --policy_ckpt runs/secrl_prune_20p/pruner_policy.pt \
  --group_size 16 \
  --device cuda \
  --dtype float16 \
  --save_tokenizer
```

The output directory is a standard Hugging Face model folder and can be loaded with:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "runs/codellama_7b_secrl_prune_20p"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
```

---

## Random-pruning baseline

To export a random structured pruning baseline without a learned policy checkpoint:

```bash
python scripts/export_pruned_model.py \
  --model_name_or_path codellama/CodeLlama-7b-Python-hf \
  --out_dir runs/codellama_7b_random_prune_20p \
  --keep_target 0.8 \
  --group_size 16 \
  --device cuda \
  --dtype float16 \
  --save_tokenizer
```

---

## Important arguments

| Argument | Used in | Meaning |
|---|---|---|
| `--top_k` / `--topk` | teacher collection / training | Number of teacher tokens kept per step |
| `--keep_target` | training / export | Fraction of FFN channels to keep |
| `--group_size` | training / export | Number of FFN units grouped into one pruning action |
| `--max_gen_steps` | training | Maximum teacher-forced KD rollout length |
| `--sparsity_coef` | training | Penalty strength for deviating from target keep ratio |
| `--trust_remote_code` | all model-loading scripts | Needed for some Hugging Face models such as Qwen |

Compression examples:

| Desired pruning ratio | `keep_target` |
|---:|---:|
| 10% pruned | `0.9` |
| 20% pruned | `0.8` |
| 30% pruned | `0.7` |

---

## Notes and limitations

- The current implementation targets LLaMA/CodeLLaMA-style MLPs with `gate_proj`, `up_proj`, and `down_proj`.
- The teacher and student should use the same tokenizer settings during teacher collection and policy training.
- Do not commit large generated files such as `kd_targets.jsonl`, model checkpoints, or exported Hugging Face model folders.
- Use Git LFS only if you intentionally want to release large checkpoints.
- For reproducible experiments, report the base model, dataset, `top_k`, `keep_target`, `group_size`, seed, and evaluation benchmark.

---

## Citation

If you use this repository, please cite:

bibtext
@inproceedings{memarzadeh2026secrlprune,
  title     = {SecRL-Prune: Structured Reinforcement Learning-Based Pruning of CodeLLMs for Preserving Adversarial Code Mutation},
  author    = {Memarzadehsaghezi, Parsa and Madani, Pooria and El-Khatib, Khalil},
  booktitle = {Proceedings of the Sixteenth ACM Conference on Data and Application Security and Privacy},
  year      = {2026},
  publisher = {ACM},
  address   = {Frankfurt am Main, Germany},
  doi       = {10.1145/3800506.3803508}
}

---

## Acknowledgments

This repository is released for research on structured CodeLLM compression and teacher-preserving pruning.
