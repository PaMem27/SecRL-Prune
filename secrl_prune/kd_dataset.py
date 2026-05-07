# experiments/kd_dataset.py
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_kd_records(path: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def prepare_example(rec: Dict[str, Any], tokenizer, topk: int):
    """
    Builds one training example for KD/pruning with RAW teacher mass.

    Returns:
      - input_ids: LongTensor [1, L]
      - attention_mask: LongTensor [1, L]
      - teacher_topk_ids_seq:  List[LongTensor[k]] length T
      - teacher_topk_probs_seq: List[FloatTensor[k]] length T   (RAW, UN-RENORMALIZED)
      - gen_len: int (T)
      - prompt: str
      - task_id: Any
    """
    if "prompt" not in rec or not rec["prompt"]:
        raise ValueError(f"Missing 'prompt' for task_id={rec.get('task_id')}")

    prompt = rec["prompt"]

    # IMPORTANT: align with collection (BOS on, EOS off; add_special_tokens=True)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]                         # [1, L]
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    # KD sequences (must be present)
    if "topk_ids" not in rec or "topk_probs" not in rec:
        raise ValueError(f"Missing topk_ids/topk_probs for task_id={rec.get('task_id')}")

    ids_seq   = rec["topk_ids"]                          # list length T
    probs_seq = rec["topk_probs"]                        # list length T (RAW, not renormed)
    T = int(rec.get("gen_len", len(ids_seq)))

    # Basic consistency
    if len(ids_seq) != len(probs_seq):
        raise ValueError(f"topk_ids vs topk_probs length mismatch for task_id={rec.get('task_id')}")
    if T != len(ids_seq):
        T = len(ids_seq)

    if T == 0:
        raise ValueError(f"No KD steps for task_id={rec.get('task_id')}")

    # Enforce a fixed k across all steps (smallest available per-step)
    per_step_k = [len(step) for step in ids_seq[:T]]
    if any(k_i == 0 for k_i in per_step_k):
        raise ValueError(f"Found an empty top-k list in KD record for task_id={rec.get('task_id')}")
    k_all_steps = min(per_step_k)
    k = min(topk, k_all_steps)

    # Convert to fixed-k tensors per step (NO renormalization)
    ids_list: List[torch.Tensor] = []
    probs_list: List[torch.Tensor] = []
    for t in range(T):
        ids_t = ids_seq[t]
        probs_t = probs_seq[t]
        if len(ids_t) != len(probs_t):
            raise ValueError(
                f"Per-step ids/probs length mismatch at t={t} for task_id={rec.get('task_id')}"
            )
        ids_list.append(torch.tensor(ids_t[:k], dtype=torch.long))
        probs_list.append(torch.tensor(probs_t[:k], dtype=torch.float))

    # Optional sanity: if the collection stored input_len, check it
    if "input_len" in rec:
        stored_L = int(rec["input_len"])
        got_L = int(input_ids.shape[-1])
        if stored_L != got_L:
            print(f"[warn] task_id={rec.get('task_id')}: stored input_len={stored_L}, "
                  f"tokenized now={got_L}. (Different tokenizer settings?)")

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        teacher_topk_ids_seq=ids_list,            # [T] of [k]
        teacher_topk_probs_seq=probs_list,        # [T] of [k], RAW (unrenormalized)
        gen_len=T,
        prompt=prompt,
        task_id=rec.get("task_id"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd_targets_path", required=True, default="../kd_targets1.jsonl",
                    help="Path to kd_targets.jsonl (must contain raw, unrenormalized top-K probs).")
    ap.add_argument("--model_name_or_path", required=True,
                    default="codellama/CodeLlama-7b-Python-hf")
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    path = Path(args.kd_targets_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Make BOS/EOS explicit (matches typical CodeLlama collection)
    if hasattr(tok, "add_bos_token"):
        tok.add_bos_token = True
    if hasattr(tok, "add_eos_token"):
        tok.add_eos_token = False

    recs = load_kd_records(str(path))
    print(f"[info] loaded {len(recs)} KD records")

    # Show one sample (index 0)
    ex = prepare_example(recs[0], tok, args.topk)
    L = ex["input_ids"].shape[-1]
    T = ex["gen_len"]
    k = ex["teacher_topk_ids_seq"][0].numel()
    print(f"[sanity] prompt tokens L={L}, timesteps T={T}, topk={k}")

    # Inspect sums of top-K mass (should be <= 1; <1 means OTHER>0, which is expected)
    probe_ts = [0, min(1, T - 1), T - 1] if T > 1 else [0]
    for t in probe_ts:
        sum_k = float(ex["teacher_topk_probs_seq"][t].sum().item())
        other = max(0.0, 1.0 - sum_k)
        print(f"[sanity] t={t}: sum(topK)={sum_k:.6f}, OTHER=1-sum(topK)={other:.6f}")
        if sum_k > 1.0 + 1e-4:
            print("  [warn] sum(topK) > 1.0 — KD targets look inconsistent at this step.")
        if abs(sum_k - 1.0) < 1e-4:
            print("  [note] sum(topK)≈1 → looks renormalized on K; "
                  "the recommended KL (Top-K + OTHER) will have no OTHER mass at this step.")

    # Shape checks
    assert len(ex["teacher_topk_ids_seq"]) == T
    assert len(ex["teacher_topk_probs_seq"]) == T
    assert ex["teacher_topk_ids_seq"][0].shape == ex["teacher_topk_probs_seq"][0].shape

    print("[ok] kd_dataset sanity checks passed.")


if __name__ == "__main__":
    main()
