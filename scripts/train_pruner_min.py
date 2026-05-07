#!/usr/bin/env python3
# experiments/train_pruner_min.py

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Project utilities (unchanged)
from secrl_prune.kd_dataset import load_kd_records, prepare_example
from secrl_prune.kd_penalty import kd_kl_on_topk
from secrl_prune.pruned_view import PrunedMLPView
from secrl_prune.pruner import MLPPrunerPolicy


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_seq_to_device(seq: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [x.to(device) for x in seq]


@torch.inference_mode()
def forward_student_sequence(
    student: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    ids_seq: List[torch.Tensor],
    T: int
) -> List[torch.Tensor]:
    """Teacher-forced rollout for T steps using KV-cache; returns last-token logits per step (fp32).

    Much faster than repeatedly re-running the full prefix (O(T) vs O(T^2)).
    """
    logits_seq: List[torch.Tensor] = []
    device = input_ids.device
    dtype = input_ids.dtype  # should be torch.long for input_ids

    # Step 0: run on the full prompt with cache
    out = student(input_ids=input_ids, use_cache=True)
    logits_seq.append(out.logits[0, -1].float())
    past = out.past_key_values

    # Subsequent steps: feed one teacher token at a time using cache
    for t in range(T - 1):
        next_id = ids_seq[t][0].item()  # feed top-1 teacher token
        next_tok = torch.tensor([[next_id]], device=device, dtype=dtype)
        out = student(input_ids=next_tok, use_cache=True, past_key_values=past)
        logits_seq.append(out.logits[0, -1].float())
        past = out.past_key_values

    return logits_seq


# ---------------------------
# Main training
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Minimal pruner trainer (REINFORCE + KD top-k)")
    ap.add_argument("--kd_targets_path", type=str, required=True,
                    help="Path to kd_targets.jsonl (prompt + topk ids/probs).")
    ap.add_argument("--model_name_or_path", type=str, required=True,
                    help="Student model (same tokenizer as KD collection).")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory (pruner checkpoint).")

    # Minimal knobs
    ap.add_argument("--steps", type=int, default=10, help="Policy update steps.")
    ap.add_argument("--batch_size", type=int, default=100, help="KD examples per step.")
    ap.add_argument("--lr", type=float, default=3e-4, help="Policy learning rate.")
    ap.add_argument("--temperature", type=float, default=1.0, help="KD temperature for student.")
    ap.add_argument("--topk", type=int, default=124, help="Use first k teacher tokens/step.")
    ap.add_argument("--max_gen_steps", type=int, default=64,
                    help="Cap teacher-forced rollout length per example for speed.")
    ap.add_argument("--keep_target", type=float, default=0.8,
                    help="Target FFN keep ratio (0..1) used only in loss penalty.")
    ap.add_argument("--sparsity_coef", type=float, default=50.0,
                    help="Penalty strength for (keep - keep_target)^2.")
    ap.add_argument("--group_size", type=int, default=1, help="Group FFN units for action space.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Pass trust_remote_code=True to HF loaders (needed for some models like Qwen).")

    args = ap.parse_args()
    args.keep_target = float(max(0.0, min(1.0, args.keep_target)))

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint paths
    best_ckpt_path = out_dir / "pruner_policy.pt"

    # Log file
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")

    # --- Load tokenizer & student (frozen) ---
    auto_trust_remote = args.trust_remote_code or ("qwen" in args.model_name_or_path.lower())
    tokenizer_kwargs = {"use_fast": True}
    if auto_trust_remote:
        tokenizer_kwargs["trust_remote_code"] = True
    try:
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    except Exception as fast_err:
        if tokenizer_kwargs.get("use_fast", True):
            tokenizer_kwargs["use_fast"] = False
            try:
                tok = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
            except Exception:
                raise fast_err
        else:
            raise fast_err

    eos_token_val = tok.eos_token_id
    added_pad_token = False
    if tok.pad_token_id is None:
        if eos_token_val is not None:
            tok.pad_token_id = eos_token_val if isinstance(eos_token_val, int) else eos_token_val[0]
            if tok.pad_token is None and tok.eos_token is not None:
                tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True

    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "device_map": None,
    }
    if auto_trust_remote:
        model_kwargs["trust_remote_code"] = True
    student: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    if added_pad_token:
        student.resize_token_embeddings(len(tok))
    student = student.to(device)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)

    # --- Build KD examples in memory ---
    recs = load_kd_records(args.kd_targets_path)
    if len(recs) == 0:
        raise RuntimeError(f"No KD records at: {args.kd_targets_path}")

    examples: List[Dict[str, Any]] = []
    for rec in recs:
        try:
            ex = prepare_example(rec, tok, args.topk)
            examples.append(ex)
        except Exception as e:
            print(f"[skip] task_id={rec.get('task_id')} reason={e}")
    if len(examples) == 0:
        raise RuntimeError("All KD records failed to prepare.")

    print(f"[info] usable KD examples: {len(examples)}")

    # --- Initialize pruner policy ---
    policy = MLPPrunerPolicy(
        model=student,
        group_size=args.group_size,
        init_keep_prob=args.keep_target,  # simple init near target
        device=device,
        keep_target=None,
    )
    opt = AdamW(policy.parameters(), lr=args.lr)

    # (No per-layer EMA baselines in this minimal trainer)

    # --- Attach pruning view once ---
    view = PrunedMLPView(student)
    view.attach()

    try:
        pbar = tqdm(range(1, args.steps + 1), desc="train_pruner_min")
        best_mean_reward = float("-inf")
        for step in pbar:
            # Mini-batch sample
            batch = [random.choice(examples) for _ in range(max(1, args.batch_size))]

            logprobs = []
            rewards = []
            kd_vals = []
            keeps = []

            # (No per-layer accumulators in this minimal trainer)
            # For reporting, track the max rollout length used in this batch
            T_used = 0

            for bi, ex in enumerate(batch):
                input_ids = ex["input_ids"].to(device)      # [1, L]
                T_full = int(ex["gen_len"])
                T = max(1, min(T_full, args.max_gen_steps))
                T_used = max(T_used, T)

                ids_seq_full = ex["teacher_topk_ids_seq"]
                probs_seq_full = ex["teacher_topk_probs_seq"]

                ids_seq = move_seq_to_device(ids_seq_full[:T], device)     # len T, each [k]
                probs_seq = move_seq_to_device(probs_seq_full[:T], device)  # len T, each [k]

                # Sample Bernoulli masks
                view.clear_all_masks()
                # sample_masks returns per-layer logprobs
                mask_dict, logprob, _entropy_ignored, keep_ratio, layer_logprobs = policy.sample_masks(
                    training=True, fixed_keep=args.keep_target
                )
                # Apply masks without tracking gradients
                view.set_global_masks({k: v.detach() for k, v in mask_dict.items()})

                # KD loss (student vs teacher top-k)
                student_logits_seq = forward_student_sequence(student, input_ids, ids_seq, T)
                kd_loss = kd_kl_on_topk(
                    student_logits_seq=student_logits_seq,
                    teacher_topk_ids_seq=ids_seq,
                    teacher_topk_probs_seq=probs_seq,
                    temp=args.temperature,
                    reduction="mean",
                )

                # Reward: prefer low KD and keep near target
                kr = keep_ratio if torch.is_tensor(keep_ratio) else torch.tensor(keep_ratio, device=device)
                # Reward should not backprop through keep penalty either
                keep_penalty = (kr - args.keep_target).detach() ** 2
                reward = (-kd_loss.detach() - args.sparsity_coef * keep_penalty).float()

                logprobs.append(logprob.float())
                rewards.append(reward)
                kd_vals.append(float(kd_loss))
                keeps.append(float(kr))

                # (No per-layer reward/logprob usage in this minimal trainer)

            # REINFORCE with simple batch baseline
            logprobs_t = torch.stack(logprobs)          # [B]
            rewards_t = torch.stack(rewards)            # [B]
            advantages = rewards_t - rewards_t.mean()   # center to reduce variance
            policy_loss = (-(advantages * logprobs_t)).mean()

            opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            opt.step()
            # (No EMA baseline update in this minimal trainer)

            # Compute mean reward for the batch and save checkpoints
            mean_reward = float(rewards_t.mean().item())

            # Build a checkpoint payload for this step (only keep the best)
            ckpt = {
                "step": step,
                "policy_state": policy.state_dict(),
                "optimizer_state": opt.state_dict(),
                "batch_mean_reward": mean_reward,
                "config": {
                    "group_size": args.group_size,
                    "keep_target": args.keep_target,
                    "model_name_or_path": args.model_name_or_path,
                    "topk": args.topk,
                    "temperature": args.temperature,
                    "sparsity_coef": args.sparsity_coef,
                    "seed": args.seed,
                },
            }

            # If this step improves the best reward, update best checkpoint
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                ckpt_best = dict(ckpt)
                ckpt_best["best_mean_reward"] = best_mean_reward
                # Save/overwrite stable "best" checkpoint
                try:
                    torch.save(ckpt_best, best_ckpt_path)
                except Exception:
                    pass
            

            pbar.set_postfix({
                "kd": f"{sum(kd_vals)/len(kd_vals):.4f}",
                "keep": f"{sum(keeps)/len(keeps):.3f}",
                "T": T_used,
                "B": len(batch),
                "bestR": f"{best_mean_reward:.4f}",
            })

            # Write JSONL log for this step
            log_row = {
                "ts": time.time(),
                "step": step,
                "batch_mean_reward": mean_reward,
                "policy_loss": float(policy_loss.detach().item()),
                "avg_kd": float(sum(kd_vals) / max(1, len(kd_vals))),
                "avg_keep": float(sum(keeps) / max(1, len(keeps))),
                "T": int(T_used),
                "B": int(len(batch)),
                "best_mean_reward": best_mean_reward,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            try:
                log_f.write(json.dumps(log_row) + "\n")
                log_f.flush()
            except Exception:
                pass

        # --- Final notice ---
        print(f"[ok] best policy so far at: {best_ckpt_path}")

    finally:
        with torch.no_grad():
            view.clear_all_masks()
            view.detach()
        try:
            log_f.close()
        except Exception:
            pass
        print("[done] minimal pruner training finished.")


if __name__ == "__main__":
    main()
