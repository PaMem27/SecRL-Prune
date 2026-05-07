#!/usr/bin/env python3
# experiments/export_pruned_model.py
import os
import argparse
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from secrl_prune.pruner import MLPPrunerPolicy
from secrl_prune.pruner import _discover_mlp_layers as discover_mlps


def _unwrap_mlp_obj(x):
    """Return the actual MLP module regardless of whether `x` is a _LayerSpec or the MLP itself."""
    if hasattr(x, "up_proj") and hasattr(x, "down_proj"):
        return x
    mlp = getattr(x, "mlp", None)
    if mlp is None:
        raise AttributeError("Object does not have expected MLP attributes (up_proj/down_proj) or .mlp")
    return mlp


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _structured_prune_one_mlp(mlp: nn.Module, keep_idx: torch.Tensor) -> None:
    """
    In-place replace up_proj, gate_proj, down_proj with pruned Linear layers.

    keep_idx: 1D LongTensor of indices to KEEP in the intermediate dimension.
    """
    assert hasattr(mlp, "up_proj") and hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj")
    up: nn.Linear = mlp.up_proj
    gate: nn.Linear = mlp.gate_proj
    down: nn.Linear = mlp.down_proj

    device = up.weight.device
    dtype = up.weight.dtype

    # --- sizes before ---
    d_in = up.in_features          # model hidden size
    I = up.out_features            # intermediate size
    K = int(keep_idx.numel())

    # sanity
    assert gate.out_features == I and down.in_features == I, "MLP wiring mismatch"

    # Create new layers with pruned shapes
    new_up   = nn.Linear(d_in, K, bias=(up.bias is not None)).to(device=device, dtype=dtype)
    new_gate = nn.Linear(d_in, K, bias=(gate.bias is not None)).to(device=device, dtype=dtype)
    new_down = nn.Linear(K, d_in, bias=(down.bias is not None)).to(device=device, dtype=dtype)

    # Copy weights/biases
    with torch.no_grad():
        new_up.weight.copy_(up.weight[keep_idx, :])
        if up.bias is not None:
            new_up.bias.copy_(up.bias[keep_idx])

        new_gate.weight.copy_(gate.weight[keep_idx, :])
        if gate.bias is not None:
            new_gate.bias.copy_(gate.bias[keep_idx])

        new_down.weight.copy_(down.weight[:, keep_idx])
        if down.bias is not None:
            new_down.bias.copy_(down.bias)

    # Swap into the module
    mlp.up_proj = new_up
    mlp.gate_proj = new_gate
    mlp.down_proj = new_down

    # Best effort: if the MLP stores an attribute called 'intermediate_size', update it
    if hasattr(mlp, "intermediate_size"):
        try:
            setattr(mlp, "intermediate_size", K)
        except Exception:
            pass


def _build_keep_indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: float/bool [I] where >0.5 means keep
    returns LongTensor indices
    """
    keep = mask > 0.5 if mask.dtype != torch.bool else mask
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1).long()
    return idx


def _random_keep_indices(I: int, keep_target: float, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        perm = torch.randperm(I, generator=gen, device=device)
    else:
        perm = torch.randperm(I, device=device)
    K = int(round(keep_target * I))
    K = max(0, min(K, I))
    return perm[:K].sort().values.long()


def _uniform_check_same_intermediate(mlps: List[nn.Module]) -> int:
    I0 = None
    for m in mlps:
        m = _unwrap_mlp_obj(m)
        I = m.up_proj.out_features
        if I0 is None:
            I0 = I
        else:
            if I != I0:
                raise RuntimeError("This script expects all layers to share the same intermediate_size.")
    return int(I0)


def _adapt_ckpt_input_dim_to_policy(state: Dict[str, torch.Tensor], policy: MLPPrunerPolicy) -> Dict[str, torch.Tensor]:
    """
    Make checkpoint's first Linear (scorer.0.weight) match the current policy's expected
    input feature dim by zero-padding or slicing. No other keys need changing.
    """
    key = "scorer.0.weight"
    if key not in state:
        return state

    w = state[key]
    # first Linear input dim of the scorer (5 or 6 depending on variant)
    target_in = int(policy.scorer[0].weight.shape[1])
    cur_in = int(w.shape[1])

    if cur_in == target_in:
        return state

    with torch.no_grad():
        if cur_in < target_in:
            # pad extra columns with zeros
            pad_cols = target_in - cur_in
            pad = torch.zeros(w.shape[0], pad_cols, dtype=w.dtype)
            state[key] = torch.cat([w, pad], dim=1)
        else:
            # slice extra columns
            state[key] = w[:, :target_in]

    return state


def main():
    ap = argparse.ArgumentParser(description="Export a structurally pruned HF model (CodeLlama/LLaMA-style MLPs).")
    ap.add_argument("--model_name_or_path", required=True, help="Base model to prune (HF id or local path).")
    ap.add_argument("--out_dir", required=True, help="Where to save the pruned HF model.")
    ap.add_argument("--keep_target", type=float, required=True,
                    help="Fraction of FFN units to KEEP per layer (0..1). Example: 0.7 keeps 70% (prunes 30%).")
    ap.add_argument("--policy_ckpt", type=str, default=None,
                    help="Optional path to policy checkpoint (.pt). If omitted, random pruning is used.")
    ap.add_argument("--group_size", type=int, default=None,
                    help="Group size used in training (if not inferable from checkpoint). Default: 16 if needed.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for turning policy probabilities into 0/1 when not using fixed_keep (unused here).")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=123, help="Seed for random pruning.")
    ap.add_argument("--save_tokenizer", action="store_true", help="Also save the tokenizer next to the model.")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Pass trust_remote_code=True to HF loaders (needed for models like Qwen).")
    args = ap.parse_args()

    # clamp keep_target
    keep_target = float(max(0.0, min(1.0, args.keep_target)))

    # device/dtype
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.dtype == "auto":
        torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model (newer HF prefers dtype=)
    print(f"[load] model: {args.model_name_or_path}")
    auto_trust_remote = args.trust_remote_code or ("qwen" in args.model_name_or_path.lower())
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "device_map": None,
    }
    if auto_trust_remote:
        model_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    ).to(device)

    # Discover MLPs and sanity check
    mlps = discover_mlps(model)
    if not mlps:
        raise RuntimeError("Could not find LLaMA-style MLPs (up_proj/gate_proj/down_proj).")
    # Accept _LayerSpec or raw modules; normalize to modules afterwards
    I = _uniform_check_same_intermediate(mlps)
    print(f"[info] #layers={len(mlps)}, intermediate_size={I}, keep_target={keep_target:.3f}")
    mlp_modules: List[nn.Module] = [_unwrap_mlp_obj(m) for m in mlps]

    # Determine masks
    if args.policy_ckpt is not None and args.policy_ckpt.strip():
        ckpt = torch.load(args.policy_ckpt, map_location="cpu")
        state = ckpt.get("policy_state", {})
        ckpt_args: Dict = ckpt.get("args", {}) or ckpt.get("config", {}) or {}

        # Resolve group size & init prob (CLI > checkpoint > default)
        gs = int(args.group_size) if args.group_size is not None else int(ckpt_args.get("group_size", 16))
        ikp = float(ckpt_args.get("init_keep_prob", 0.9))

        print(f"[load] policy: {args.policy_ckpt} (group_size={gs})")
        policy = MLPPrunerPolicy(
            model=model,
            group_size=gs,
            init_keep_prob=ikp,
            device=torch.device(device),
            keep_target=None,  # pass fixed_keep at sample time
        )

        # Adapt 5-feature vs 6-feature checkpoints to the current policy
        state = _adapt_ckpt_input_dim_to_policy(state, policy)

        # Load (tolerate missing/unexpected keys gracefully)
        incompat = policy.load_state_dict(state, strict=False)
        missing = list(getattr(incompat, "missing_keys", []))
        unexpected = list(getattr(incompat, "unexpected_keys", []))
        if missing or unexpected:
            print(f"[warn] policy state_dict missing={missing} unexpected={unexpected}")

        # Deterministic greedy Top-M per layer for reproducible export
        with torch.no_grad():
            mask_dict, _, _, keep_ratio, _ = policy.sample_masks(
                training=False,
                fixed_keep=keep_target  # exact fraction per layer
            )

        # Convert to indices per layer
        layer_keep_indices: List[torch.Tensor] = []
        for li in range(len(mlp_modules)):
            m = mask_dict[li].to(device=model.device)
            idx = _build_keep_indices_from_mask(m)
            if idx.numel() == 0:
                print(f"[warn] layer {li}: empty keep set; keeping one unit to avoid shape errors.")
                idx = torch.tensor([0], device=model.device, dtype=torch.long)
            layer_keep_indices.append(idx)
        print(f"[info] policy implied global keep_ratio ~ {float(keep_ratio):.4f}")

        # Ensure uniform K across layers (HF config requires one intermediate_size)
        Ks = [int(x.numel()) for x in layer_keep_indices]
        if len(set(Ks)) != 1:
            K_target = int(round(keep_target * I))
            print(f"[warn] per-layer K not uniform; enforcing K={K_target} across all layers.")
            for i, idx in enumerate(layer_keep_indices):
                if idx.numel() > K_target:
                    layer_keep_indices[i] = idx[:K_target]
                elif idx.numel() < K_target:
                    # pad with smallest missing indices
                    mask_missing = torch.ones(I, dtype=torch.bool, device=idx.device)
                    mask_missing[idx] = False
                    add_idx = torch.nonzero(mask_missing, as_tuple=False).squeeze(-1)[:(K_target - idx.numel())]
                    layer_keep_indices[i] = torch.cat([idx, add_idx]).sort().values

        K = int(layer_keep_indices[0].numel())

    else:
        # Random pruning path
        print("[info] no policy checkpoint provided => random pruning.")
        K = int(round(keep_target * I))
        layer_keep_indices = []
        for _ in range(len(mlp_modules)):
            idx = _random_keep_indices(I, keep_target, device=torch.device(device), seed=args.seed)
            if idx.numel() == 0:
                idx = torch.tensor([0], device=device, dtype=torch.long)
            layer_keep_indices.append(idx)
        print(f"[info] random K={K}/{I} per layer (keep_target={keep_target:.3f})")

    # Apply structural pruning
    print("[apply] pruning all MLPs...")
    for li, (mlp, idx) in enumerate(zip(mlp_modules, layer_keep_indices)):
        _structured_prune_one_mlp(mlp, idx.to(mlp.up_proj.weight.device))
        if li % 8 == 0 or li == len(mlp_modules) - 1:
            print(f"  - layer {li:02d}: kept {idx.numel()}/{I} ({idx.numel()/I:6.2%})")

    # Update model config (single intermediate_size shared across layers)
    model.config.intermediate_size = int(layer_keep_indices[0].numel())
    print(f"[config] intermediate_size -> {model.config.intermediate_size}")

    # Save pruned model (HF-compatible)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save] writing pruned model to: {out_dir}")
    model.save_pretrained(out_dir)

    if args.save_tokenizer:
        tokenizer_kwargs = {}
        if auto_trust_remote:
            tokenizer_kwargs["trust_remote_code"] = True
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        tok.save_pretrained(out_dir)

    # Write a small metadata file
    meta = {
        "base_model": args.model_name_or_path,
        "keep_target": keep_target,
        "policy_ckpt": args.policy_ckpt,
        "group_size": (args.group_size if args.group_size is not None else "infer/16"),
        "intermediate_size_original": I,
        "intermediate_size_pruned": int(model.config.intermediate_size),
        "layers": len(mlps),
        "dtype": str(model.dtype),
    }
    torch.save(meta, out_dir / "prune_meta.pt")
    print("[done] pruning complete.")


if __name__ == "__main__":
    main()
