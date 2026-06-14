"""
print_weights.py
----------------
Load checkpoint và in ra tất cả weights liên quan đến text/audio fusion.
Bước 1: Phân tích static weights (không cần data).
Bước 2: Build model + forward hook để capture live gate values.

Usage:
    python ablation_weight_text_audio/print_weights.py
    python ablation_weight_text_audio/print_weights.py \
        --ckpt Model/Model/checkpoints_fluency/model_best_mae_fluency_fusion_only_from_final_ckpt.pth \
        --config config/config_fluency.yaml
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from pprint import pprint

# Add Model/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from model import ESLGradingModelByCandidatesWithAudio
import inspect


# ============================================================
MODALITY_NAMES = [
    "text_self",       # [B, d_fuse] – text self-attention pooled
    "audio_self",      # [B, d_fuse] – audio self-attention pooled
    "t2a",             # [B, d_fuse] – text→audio cross-attention pooled
    "a2t",             # [B, d_fuse] – audio→text cross-attention pooled
    "audio_mean",      # [B, d_fuse] – global audio mean pooled
]
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="/home/user06/Interspeech_2026/Model/Model/checkpoints_fluency/"
                "model_best_mae_fluency_fusion_only_from_final_ckpt.pth",
    )
    parser.add_argument(
        "--config",
        default="/home/user06/Interspeech_2026/Model/config/config_fluency.yaml",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/user06/Interspeech_2026/Model/ablation_weight_text_audio/results",
    )
    return parser.parse_args()


# ------------------------------------------------------------------ helpers --

def load_checkpoint_shape_safe(model, state_dict):
    current = model.state_dict()
    matched, skipped = {}, []
    for k, v in state_dict.items():
        if k in current and current[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)
    current.update(matched)
    model.load_state_dict(current)
    if skipped:
        print(f"  ⚠ Skipped {len(skipped)} keys (shape/name mismatch). First 5: {skipped[:5]}")


def sep(title="", width=80):
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"\n{'─'*1} {title} {'─'*pad}")
    else:
        print("─" * width)


# --------------------------------------------------------- static analysis ---

def analyze_static_weights(state_dict, out: dict):
    """
    Phân tích weights GatedMultimodalFusion, audio/text projections,
    regression head, và gate linear layer (text half vs audio half).
    """

    sep("1. ALL KEYS – grouped by top-level module")
    modules = {}
    for k in state_dict:
        top = k.split(".")[0]
        modules.setdefault(top, []).append(k)
    for mod, keys in sorted(modules.items()):
        n_params = sum(state_dict[k].numel() for k in keys)
        print(f"  {mod:<45} {len(keys):>4} keys  {n_params:>12,} params")

    # ---- Fusion weights ----
    sep("2. GatedMultimodalFusion – all weights")
    fusion_stats = {}
    for k, v in state_dict.items():
        if "gated_fusion" not in k and "fusion" not in k:
            continue
        arr = v.detach().cpu().float().numpy().ravel()
        s = {
            "shape": list(v.shape),
            "mean":  float(arr.mean()),
            "std":   float(arr.std()),
            "abs_mean": float(np.abs(arr).mean()),
            "min":   float(arr.min()),
            "max":   float(arr.max()),
            "norm":  float(np.linalg.norm(arr)),
        }
        fusion_stats[k] = s
        print(f"  {k}")
        print(f"    shape={s['shape']}  mean={s['mean']:.5f}  std={s['std']:.5f}  "
              f"abs_mean={s['abs_mean']:.5f}  norm={s['norm']:.4f}")
    out["fusion_weights"] = fusion_stats

    # ---- Gate net final linear: decompose text vs audio half ----
    sep("3. Gate Linear – TEXT half vs AUDIO half (Frobenius norm)")
    # GatedMultimodalFusion.gate_net:
    #   Linear(5*d_fuse → 2*d_fuse) → GELU → Linear(2*d_fuse → 5) → Softmax
    # The first Linear W has shape [2*d_fuse, 5*d_fuse].
    # The 5*d_fuse input is the concat of 5 modalities each d_fuse.
    # Columns [0 : d_fuse]   → text_self
    # Columns [d_fuse : 2*d_fuse] → audio_self
    # Columns [2*d_fuse : 3*d_fuse] → t2a
    # Columns [3*d_fuse : 4*d_fuse] → a2t
    # Columns [4*d_fuse : 5*d_fuse] → audio_mean

    gate_decomp = {}
    for k, v in state_dict.items():
        if "gate_net" not in k and "gate" not in k.lower():
            continue
        if "weight" not in k or v.dim() != 2:
            continue
        d_out, d_in = v.shape
        arr = v.detach().cpu().float().numpy()
        # Check if this looks like the 5-modality input (d_in divisible by 5)
        if d_in % 5 == 0:
            chunk = d_in // 5
            norms = []
            for i, name in enumerate(MODALITY_NAMES):
                col_block = arr[:, i*chunk:(i+1)*chunk]
                n = float(np.linalg.norm(col_block))
                norms.append(n)
            total_norm = sum(norms)
            print(f"\n  Key: {k}  shape={list(v.shape)}")
            print(f"  {'Modality':<20} {'‖W[:, block]‖':>14}  {'%':>7}")
            for i, name in enumerate(MODALITY_NAMES):
                pct = 100.0 * norms[i] / total_norm if total_norm > 0 else 0.0
                marker = " ◄ TEXT" if "text" in name else (" ◄ AUDIO" if "audio" in name else "")
                print(f"  {name:<20} {norms[i]:>14.4f}  {pct:>6.2f}%{marker}")

            # Text vs Audio summary
            text_norms  = [norms[0], norms[2]]   # text_self, t2a
            audio_norms = [norms[1], norms[3], norms[4]]  # audio_self, a2t, audio_mean
            text_total  = sum(text_norms)
            audio_total = sum(audio_norms)
            grand_total = text_total + audio_total
            print(f"\n  ── Summary ──")
            print(f"  TEXT  (text_self + t2a):              "
                  f"{text_total:>10.4f}  {100*text_total/grand_total:>5.1f}%")
            print(f"  AUDIO (audio_self + a2t + audio_mean):"
                  f" {audio_total:>10.4f}  {100*audio_total/grand_total:>5.1f}%")

            gate_decomp[k] = {
                "shape": list(v.shape),
                "modality_norms": dict(zip(MODALITY_NAMES, norms)),
                "text_total_norm": text_total,
                "audio_total_norm": audio_total,
                "text_pct": 100 * text_total / grand_total,
                "audio_pct": 100 * audio_total / grand_total,
            }
        else:
            # Second linear: [5, 2*d_fuse] → final gate scores per modality
            print(f"\n  Key: {k}  shape={list(v.shape)}")
            if d_out == 5:
                # Each row corresponds to one modality gate score
                row_norms = [float(np.linalg.norm(arr[i])) for i in range(5)]
                total = sum(row_norms)
                print(f"  Per-modality output norm (∝ gate sensitivity):")
                for i, name in enumerate(MODALITY_NAMES):
                    pct = 100 * row_norms[i] / total if total > 0 else 0.0
                    print(f"    {name:<20} {row_norms[i]:>10.4f}  {pct:>5.1f}%")
    out["gate_decomp"] = gate_decomp

    # ---- Audio / Text projections ----
    sep("4. Audio Projection / Adapter weights")
    proj_stats = {}
    for k, v in state_dict.items():
        if any(tag in k for tag in ["audio_proj", "audio_norm", "audio_adapter",
                                     "text_proj", "text_norm"]):
            arr = v.detach().cpu().float().numpy().ravel()
            s = {"shape": list(v.shape),
                 "mean": float(arr.mean()), "std": float(arr.std()),
                 "abs_mean": float(np.abs(arr).mean()),
                 "norm": float(np.linalg.norm(arr))}
            proj_stats[k] = s
            print(f"  {k:<60} norm={s['norm']:.4f}  abs_mean={s['abs_mean']:.5f}")
    out["projection_weights"] = proj_stats

    # ---- Part importance weights (learnable) ----
    sep("5. Learnable Part Importance Weights (audio parts)")
    for k, v in state_dict.items():
        if "part_importance" in k:
            arr = v.detach().cpu().float().numpy()
            softmaxed = np.exp(arr) / np.exp(arr).sum()
            print(f"  {k}: raw={arr.tolist()}")
            print(f"  Softmax: {np.round(softmaxed, 4).tolist()}")
            out["part_importance_raw"] = arr.tolist()
            out["part_importance_softmax"] = softmaxed.tolist()

    # ---- Regression head ----
    sep("6. Regression Head layer norms")
    for k, v in state_dict.items():
        if "reg_head" in k and "weight" in k and v.dim() == 2:
            arr = v.detach().cpu().float().numpy()
            print(f"  {k:<60} shape={list(v.shape)}  norm={np.linalg.norm(arr):.4f}")

    # ---- QuestionAwareEncoder gate ----
    sep("7. QuestionAwareEncoder gate weights")
    for k, v in state_dict.items():
        if "question_aware_encoder" in k and "gate" in k and "weight" in k and v.dim() == 2:
            arr = v.detach().cpu().float().numpy()
            d_out, d_in = arr.shape
            half = d_in // 2
            q_half = arr[:, :half]
            r_half = arr[:, half:]
            q_norm = float(np.linalg.norm(q_half))
            r_norm = float(np.linalg.norm(r_half))
            total  = q_norm + r_norm
            print(f"  {k}  shape={list(v.shape)}")
            print(f"    ‖question_half‖ = {q_norm:.4f}  ({100*q_norm/total:.1f}%)")
            print(f"    ‖response_half‖ = {r_norm:.4f}  ({100*r_norm/total:.1f}%)")
            out["qa_gate_question_pct"] = round(100 * q_norm / total, 2)
            out["qa_gate_response_pct"] = round(100 * r_norm / total, 2)


# ------------------------------------------------------- live gate capture ---

def register_gate_hooks(model):
    """
    Đăng ký forward hooks lên gate_net để capture actual gate weights
    (giá trị sau Softmax) khi forward pass.
    Returns captured list (mutable) and hook handles.
    """
    captured = []

    def hook_fn(module, input, output):
        # output shape: [B, 5] after Softmax
        captured.append(output.detach().cpu().float())

    hooks = []
    if hasattr(model, "gated_fusion") and hasattr(model.gated_fusion, "gate_net"):
        h = model.gated_fusion.gate_net.register_forward_hook(hook_fn)
        hooks.append(h)
        print("  ✓ Hook registered on model.gated_fusion.gate_net")
    else:
        print("  ⚠ gated_fusion.gate_net not found – no hooks registered.")

    return captured, hooks


def run_dummy_forward(model, cfg, device="cpu"):
    """
    Tạo dummy input và chạy một forward pass để trigger hooks.
    Không cần real data – chỉ để capture gate output structure.
    """
    B = 2
    seq_len = 64
    num_chunks = cfg.audio.eval_num_chunks or cfg.audio.num_chunks

    if cfg.model.use_question_encoder:
        q_ids  = torch.zeros(B, 32, dtype=torch.long).to(device)
        q_mask = torch.ones(B, 32, dtype=torch.long).to(device)
        r_ids  = torch.zeros(B, seq_len, dtype=torch.long).to(device)
        r_mask = torch.ones(B, seq_len, dtype=torch.long).to(device)
        kwargs = dict(
            question_input_ids=q_ids, question_attention_mask=q_mask,
            response_input_ids=r_ids, response_attention_mask=r_mask,
        )
    else:
        ids  = torch.zeros(B, seq_len, dtype=torch.long).to(device)
        mask = torch.ones(B, seq_len, dtype=torch.long).to(device)
        kwargs = dict(input_ids=ids, attention_mask=mask)

    # Whisper input: [B, num_chunks, mel_bins, 3000]
    if cfg.model.audio_encoder_type.lower() == "whisper":
        mel_bins = model.audio_encoder.get_num_mel_bins()
        audio = torch.zeros(B, num_chunks, mel_bins, 3000).to(device)
    else:
        audio = torch.zeros(B, num_chunks, cfg.audio.max_waveform_len).to(device)

    kwargs["audio"] = audio

    with torch.no_grad():
        model(**kwargs)


def build_model(cfg):
    sig = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in cfg.model.__dict__.items() if k in valid_params}
    return ESLGradingModelByCandidatesWithAudio(**kwargs)


# -------------------------------------------------------------------  main ---

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🔍  Checkpoint : {args.ckpt}")
    print(f"⚙️   Config     : {args.config}")
    print("=" * 80)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    # Basic checkpoint info
    sep("Checkpoint Metadata")
    print(f"  val_mae : {ckpt.get('val_mae', 'N/A')}")
    print(f"  epoch   : {ckpt.get('epoch', 'N/A')}")
    print(f"  monitor : {ckpt.get('monitor', 'N/A')}")
    ckpt_cfg = ckpt.get("config", {})
    if ckpt_cfg:
        print(f"  config keys: {list(ckpt_cfg.keys())}")

    results = {
        "checkpoint": args.ckpt,
        "val_mae": ckpt.get("val_mae"),
        "epoch":   ckpt.get("epoch"),
    }

    # ---- Static analysis ----
    analyze_static_weights(state_dict, results)

    # ---- Build model + hook-based live capture ----
    sep("Building Model for Live Gate Capture (dummy forward)")
    cfg = Config.from_yaml(args.config)
    model = build_model(cfg)
    load_checkpoint_shape_safe(model, state_dict)
    model.eval()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.2f}%)")

    # Register hooks
    captured, hooks = register_gate_hooks(model)

    # Dummy forward
    try:
        run_dummy_forward(model, cfg, device="cpu")
        print(f"  ✓ Dummy forward complete – captured {len(captured)} gate batch(es)")
    except Exception as e:
        print(f"  ⚠ Dummy forward failed: {e}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # ---- Summarize captured gate weights ----
    sep("Live Gate Output (dummy input – structural, not data-dependent)")
    if captured:
        all_gates = torch.cat(captured, dim=0)  # [N_samples, 5]
        mean_weights = all_gates.mean(dim=0).numpy()
        std_weights  = all_gates.std(dim=0).numpy()

        print(f"\n  Gate weights (mean over {all_gates.shape[0]} samples):")
        print(f"\n  {'Modality':<20} {'Mean':>10}  {'Std':>10}  {'%':>8}")
        print(f"  {'─'*55}")
        for i, name in enumerate(MODALITY_NAMES):
            pct = 100 * mean_weights[i]
            marker = " ◄ TEXT" if "text" in name else (" ◄ AUDIO" if "audio" in name else "")
            print(f"  {name:<20} {mean_weights[i]:>10.4f}  {std_weights[i]:>10.4f}  "
                  f"{pct:>7.2f}%{marker}")

        text_weight  = float(mean_weights[0] + mean_weights[2])           # text_self + t2a
        audio_weight = float(mean_weights[1] + mean_weights[3] + mean_weights[4])  # audio_self + a2t + audio_mean
        total_w = text_weight + audio_weight
        print(f"\n  ── TEXT  vs AUDIO (dummy input) ──")
        print(f"  TEXT  (text_self + t2a)              : {text_weight:.4f}  "
              f"({100*text_weight/total_w:.1f}%)")
        print(f"  AUDIO (audio_self + a2t + audio_mean): {audio_weight:.4f}  "
              f"({100*audio_weight/total_w:.1f}%)")
        print()
        print("  ⚠ NOTE: Dummy forward uses zero inputs → weights are biased.")
        print("    Run ablation_fusion.py with real data for meaningful gate values.")

        results["dummy_gate_weights"] = {
            name: {"mean": float(mean_weights[i]), "std": float(std_weights[i]),
                   "pct": float(100 * mean_weights[i])}
            for i, name in enumerate(MODALITY_NAMES)
        }
        results["dummy_gate_text_pct"]  = 100 * text_weight / total_w
        results["dummy_gate_audio_pct"] = 100 * audio_weight / total_w

    # ---- Save JSON ----
    out_path = out_dir / "weights_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅  Saved → {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
