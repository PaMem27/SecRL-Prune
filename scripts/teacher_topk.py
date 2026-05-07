#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


def read_jsonl(path: Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, obj: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def trim_on_first_marker(text: str, markers: List[str]) -> Tuple[str, Optional[str]]:
    best_i = None
    best_m = None
    for m in markers:
        if not m:
            continue
        i = text.find(m)
        if i != -1 and (best_i is None or i < best_i):
            best_i = i
            best_m = m
    if best_i is None:
        return text, None
    return text[:best_i], best_m


def write_solution_file(root: Path, task_id: str, prompt: str, continuation: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{task_id}.py"
    prefix = prompt if prompt.endswith("\n") else (prompt + "\n")
    full_text = prefix + continuation
    with open(p, "w", encoding="utf-8") as f:
        f.write(full_text.rstrip() + "\n")
    return p


@torch.no_grad()
def softmax_then_topk(
    logits_last: torch.Tensor, k: int, T: float = 1.0
) -> Tuple[List[int], List[float], List[float]]:
    k = min(k, logits_last.numel())
    probs = torch.softmax(logits_last / max(T, 1e-6), dim=-1)
    vals, idx = torch.topk(probs, k, dim=-1)
    logvals = torch.log(vals.clamp_min(1e-45))
    return idx.tolist(), vals.tolist(), logvals.tolist()


@torch.no_grad()
def collect_teacher_topk_for_prompt(
    model,
    tokenizer,
    prompt: str,
    topk: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    stop_on_eos: bool = True,
    stop_markers: Optional[List[str]] = None,
    min_gen_tokens: int = 8,
    exec_device: torch.device = torch.device("cpu"),
    eos_token_ids: Optional[List[int]] = None,
) -> Dict:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(exec_device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(exec_device)
    if eos_token_ids is None:
        eos_val = tokenizer.eos_token_id
        if eos_val is None:
            eos_token_ids = []
        elif isinstance(eos_val, int):
            eos_token_ids = [eos_val]
        else:
            eos_token_ids = list(eos_val)

    collected_ids: List[List[int]] = []
    collected_probs: List[List[float]] = []
    collected_logprobs: List[List[float]] = []

    past_key_values = None
    cur_len = int(input_ids.shape[1])
    generated_ids: List[int] = []
    continuation_text: str = ""
    stop_reason: str = ""

    if stop_markers is None:
        stop_markers = [
            "\n\n",
            "\n# Tests",
            '\nif __name__ == "__main__":',
            "\nclass ",
            "\n\ndef ",
        ]

    for step in range(max_new_tokens):
        if past_key_values is None:
            out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
        else:
            next_inp = torch.tensor([[generated_ids[-1]]], device=exec_device)
            attn_mask = torch.cat([attn_mask, torch.ones_like(attn_mask[:, :1])], dim=1)
            out = model(
                input_ids=next_inp,
                attention_mask=attn_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = out.past_key_values
        last_logits = out.logits[0, -1]

        # Log top-K (softmax→topk) for KD
        ids, probsK, logpsK = softmax_then_topk(last_logits, k=topk, T=temperature)
        collected_ids.append(ids)
        collected_probs.append(probsK)
        collected_logprobs.append(logpsK)

        # Greedy next token
        next_id = int(torch.argmax(last_logits).item())
        generated_ids.append(next_id)

        # IMPORTANT: decode the ENTIRE generated sequence (not per-token), preserving spaces
        continuation_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,  # keep spaces exactly as coded
        )

        # Early stop checks
        if stop_on_eos and eos_token_ids and next_id in eos_token_ids:
            stop_reason = "eos_token"
            break
        if step + 1 >= min_gen_tokens:
            tail = continuation_text[-256:]
            hit_marker = None
            for m in stop_markers:
                if m and m in tail:
                    hit_marker = m
                    break
            if hit_marker is not None:
                stop_reason = f"marker:{hit_marker.encode('unicode_escape').decode()}"
                break

    # Final trim at first marker for a clean file
    trimmed_continuation, first_marker = trim_on_first_marker(continuation_text, stop_markers)
    if first_marker and not stop_reason.startswith("marker:"):
        stop_reason = f"marker:{first_marker.encode('unicode_escape').decode()}"

    return {
        "prompt": prompt,
        "continuation": trimmed_continuation,
        "topk_ids": collected_ids,
        "topk_probs": collected_probs,
        "topk_logprobs": collected_logprobs,
        "input_len": cur_len,
        "gen_len": len(generated_ids),
        "stop_reason": stop_reason or "max_new_tokens",
    }


def main():
    ap = argparse.ArgumentParser(description="Compute teacher softmax→top-k targets and save solutions.")
    ap.add_argument("--dataset", type=Path, default=Path("dataset.jsonl"),
                    help="Input dataset JSONL with fields {task_id, prompt}.")
    ap.add_argument("--out", type=Path, default=Path("kd_targets.jsonl"),
                    help="Output JSONL for KD targets.")
    ap.add_argument("--solutions_dir", type=Path, default=Path("solutions"),
                    help="Directory to write generated .py solutions (one per task).")
    ap.add_argument("--model", type=str, default="codellama/CodeLlama-7b-Python-hf",
                    help="Teacher model name or local path.")
    ap.add_argument("--top_k", type=int, default=124, help="Top-k size.")
    ap.add_argument("--max_new_tokens", type=int, default=192, help="Max generation steps.")
    ap.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature before top-k.")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                    help="Device selection.")
    ap.add_argument("--min_gen_tokens", type=int, default=8,
                    help="Min tokens before marker-based early stop.")
    ap.add_argument("--stop_markers", type=str, nargs="*", default=[
        "\n\n",
        "\n# Tests",
        'if __name__ == "__main__":',
        "\nclass ",
        "\n\ndef ",
    ], help="Strings that trigger early stop.")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Pass trust_remote_code=True to HF loaders (needed for some models like Qwen).")
    args = ap.parse_args()

    # Device / dtype
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and torch.cuda.is_available():
        major_cc = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
    else:
        dtype = torch.float32

    exec_device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Load teacher
    auto_trust_remote = args.trust_remote_code or ("qwen" in args.model.lower())
    tokenizer_kwargs = {"use_fast": True}
    if auto_trust_remote:
        tokenizer_kwargs["trust_remote_code"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    except Exception as fast_err:
        if tokenizer_kwargs.get("use_fast", True):
            tokenizer_kwargs["use_fast"] = False
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
            except Exception:
                raise fast_err
        else:
            raise fast_err

    eos_token_val = tokenizer.eos_token_id
    if isinstance(eos_token_val, int):
        eos_token_ids = [eos_token_val]
    elif eos_token_val is None:
        eos_token_ids = []
    else:
        eos_token_ids = list(eos_token_val)

    added_pad_token = False
    if tokenizer.pad_token_id is None:
        if eos_token_ids:
            tokenizer.pad_token_id = eos_token_ids[0]
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True
            eos_token_ids = [tokenizer.pad_token_id]

    model_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    if auto_trust_remote:
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Resume support: skip existing task_ids
    done_ids = set()
    if args.out.exists():
        for row in read_jsonl(args.out):
            tid = row.get("task_id")
            if tid is not None:
                done_ids.add(tid)

    # Iterate dataset
    ds = list(read_jsonl(args.dataset))
    pbar = tqdm(ds, total=len(ds))
    n_written = 0
    with open(args.out, "a", encoding="utf-8") as fout:
        for ex in pbar:
            tid = ex.get("task_id")
            prompt = ex.get("prompt", "")
            if tid in done_ids:
                pbar.set_description(f"skip {tid}")
                continue
            try:
                rec = collect_teacher_topk_for_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    topk=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    stop_on_eos=True,
                    stop_markers=args.stop_markers,
                    min_gen_tokens=args.min_gen_tokens,
                    exec_device=exec_device,
                    eos_token_ids=eos_token_ids,
                )

                # Save the .py solution
                sol_path = write_solution_file(args.solutions_dir, str(tid), rec["prompt"], rec["continuation"])

                row = {
                    "task_id": tid,
                    "prompt": rec["prompt"],
                    "continuation": rec["continuation"],
                    "solution_file": str(sol_path),
                    "topk_ids": rec["topk_ids"],
                    "topk_probs": rec["topk_probs"],
                    "topk_logprobs": rec["topk_logprobs"],
                    "input_len": rec["input_len"],
                    "gen_len": rec["gen_len"],
                    "stop_reason": rec["stop_reason"],
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                pbar.set_description(f"added {tid} (gen_len={row['gen_len']}, stop={row['stop_reason']})")
                if n_written % 16 == 0 and device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                pbar.set_description(f"ERROR {tid}: {e}")
                continue

    print(f"Done. Newly added: {n_written}, total rows now: {len(done_ids) + n_written}")


if __name__ == "__main__":
    main()
