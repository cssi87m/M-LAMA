"""
ablation_fusion.py
------------------
Ablation study: Phân tích đóng góp của text vs audio bằng 2 phương pháp:

METHOD A – Live Gate Capture (Forward Hook):
    - Chạy inference trên real data
    - Capture gate weights [B, 5] sau Softmax tại mỗi forward pass
    - Average để biết modality nào được model ưu tiên

METHOD B – Alpha Scaling (Monkey-patch):
    - Scale text/audio features TRƯỚC khi vào gated_fusion
    - So sánh MAE/QWK khi tắt/giảm từng modality
    - Quantify: "nếu không có audio, MAE tăng bao nhiêu?"

Không sửa model.py hay training code.

Usage:
    python ablation_weight_text_audio/ablation_fusion.py
    python ablation_weight_text_audio/ablation_fusion.py \
        --ckpt  Model/Model/checkpoints_fluency/model_best_mae_fluency_fusion_only_from_final_ckpt.pth \
        --config config/config_fluency.yaml \
        --split test \
        --limit 200
"""

import argparse
import json
import math
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperProcessor, Wav2Vec2Processor
import inspect

# Add Model/ to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from model import ESLGradingModelByCandidatesWithAudio
from dataloader import ESLDatasetByCandidatesWithAudio, get_collate_fn_bycandidates_with_audio

# ============================================================
MODALITY_NAMES = ["text_self", "audio_self", "t2a", "a2t", "audio_mean"]

# Ablation configs: (text_alpha, audio_alpha, name)
# alpha=1.0 → modality không đổi
# alpha=0.0 → triệt tiêu hoàn toàn modality đó
ABLATION_CONFIGS = [
    (1.0, 1.0,  "01_baseline"),
    (0.0, 1.0,  "02_audio_only__text=0"),
    (1.0, 0.0,  "03_text_only__audio=0"),
    (0.5, 1.0,  "04_text_x0.5__audio_full"),
    (0.25, 1.0, "05_text_x0.25__audio_full"),
    (0.1, 1.0,  "06_text_x0.1__audio_full"),
    (1.0, 0.5,  "07_text_full__audio_x0.5"),
    (1.0, 0.25, "08_text_full__audio_x0.25"),
    (1.0, 0.1,  "09_text_full__audio_x0.1"),
    (2.0, 1.0,  "10_text_x2__audio_full"),
    (1.0, 2.0,  "11_text_full__audio_x2"),
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
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which data split to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (quick test)",
    )
    parser.add_argument(
        "--skip_ablation",
        action="store_true",
        help="Only run live gate capture, skip alpha scaling ablation",
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
        print(f"  ⚠ Skipped {len(skipped)} keys. First 5: {skipped[:5]}")


def build_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    tok.padding_side = "right"
    return tok


def build_loader(csv_path, cfg, tokenizer, audio_processor, limit=None):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    num_chunks = cfg.audio.eval_num_chunks or cfg.audio.num_chunks
    dataset = ESLDatasetByCandidatesWithAudio(
        df,
        criteria=cfg.data.criteria,
        audio_processor=audio_processor,
        encoder_type=cfg.model.audio_encoder_type,
        num_chunks=num_chunks,
        chunk_length_sec=cfg.audio.chunk_length_sec,
        separate_question_response=cfg.model.use_question_encoder,
    )
    collate_fn = get_collate_fn_bycandidates_with_audio(
        tokenizer,
        max_length=cfg.data.max_length,
        max_audio_chunks=cfg.audio.max_audio_chunks,
        max_waveform_len=cfg.audio.max_waveform_len,
        separate_tokenize=cfg.model.use_question_encoder,
    )
    batch_size = cfg.training.eval_batch_size or cfg.training.batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=True,
    )


def quadratic_weighted_kappa(y_true, y_pred):
    bins_true = np.round(y_true * 2) / 2.0
    bins_pred = np.round(y_pred * 2) / 2.0
    ints_true = np.round(bins_true * 2).astype(int)
    ints_pred = np.round(bins_pred * 2).astype(int)
    return float(cohen_kappa_score(ints_true, ints_pred, weights="quadratic"))


def build_batch_kwargs(batch, cfg, device):
    audio = batch["audio"].to(device)
    if cfg.model.use_question_encoder:
        return dict(
            question_input_ids=batch["question_input_ids"].to(device),
            question_attention_mask=batch["question_attention_mask"].to(device),
            response_input_ids=batch["response_input_ids"].to(device),
            response_attention_mask=batch["response_attention_mask"].to(device),
            audio=audio,
        )
    else:
        return dict(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio=audio,
        )


def compute_metrics(y_true, y_pred_raw, edge_thr=3.5):
    y_pred = np.clip(np.round(y_pred_raw * 2) / 2.0, 0, 10)
    err = y_pred - y_true
    edge_mask = (y_true <= edge_thr) | (y_true >= (10 - edge_thr))
    mid_mask  = ~edge_mask
    return {
        "mae":          float(np.mean(np.abs(err))),
        "rmse":         float(math.sqrt(np.mean(err ** 2))),
        "qwk":          float(quadratic_weighted_kappa(y_true, y_pred)),
        "acc_0.5":      float(np.mean(np.abs(err) <= 0.5)),
        "acc_1.0":      float(np.mean(np.abs(err) <= 1.0)),
        "edge_mae":     float(np.mean(np.abs(err[edge_mask]))) if edge_mask.any() else float("nan"),
        "mid_mae":      float(np.mean(np.abs(err[mid_mask])))  if mid_mask.any()  else float("nan"),
        "n":            int(len(y_true)),
    }


# ======================================================== METHOD A: Hook ===

def run_live_gate_capture(model, loader, cfg, device):
    """
    Đăng ký forward hook lên gate_net để capture live gate weights.
    Returns:
        gate_matrix: np.ndarray [N_samples, 5]
    """
    captured = []

    def _hook(module, inp, out):
        # out: [B, 5] Softmax weights
        captured.append(out.detach().cpu().float())

    hook = None
    if hasattr(model, "gated_fusion") and hasattr(model.gated_fusion, "gate_net"):
        hook = model.gated_fusion.gate_net.register_forward_hook(_hook)
    else:
        print("  ⚠ gated_fusion.gate_net not found. Skipping live capture.")
        return None

    model.eval()
    all_true = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Live Gate Capture"):
            kwargs = build_batch_kwargs(batch, cfg, device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                model(**kwargs)
            all_true.extend(batch["score"].numpy().tolist())

    if hook:
        hook.remove()

    if not captured:
        return None, np.array(all_true)

    gate_matrix = torch.cat(captured, dim=0).numpy()  # [N, 5]
    return gate_matrix, np.array(all_true)


def print_gate_summary(gate_matrix, y_true, out_dir):
    """In bảng tóm tắt gate weights từ real data."""
    N = gate_matrix.shape[0]
    mean_w = gate_matrix.mean(axis=0)
    std_w  = gate_matrix.std(axis=0)
    med_w  = np.median(gate_matrix, axis=0)

    print(f"\n{'─'*70}")
    print(f"  📊 LIVE GATE WEIGHTS  (N={N} real samples)")
    print(f"{'─'*70}")
    print(f"  {'Modality':<20} {'Mean':>9}  {'Median':>9}  {'Std':>9}  {'%':>7}")
    print(f"  {'─'*60}")
    for i, name in enumerate(MODALITY_NAMES):
        pct = 100 * mean_w[i]
        marker = " ◄ TEXT" if "text" in name else (" ◄ AUDIO" if "audio" in name else "")
        print(f"  {name:<20} {mean_w[i]:>9.4f}  {med_w[i]:>9.4f}  "
              f"{std_w[i]:>9.4f}  {pct:>6.2f}%{marker}")

    text_w  = mean_w[0] + mean_w[2]            # text_self + t2a
    audio_w = mean_w[1] + mean_w[3] + mean_w[4]  # audio_self + a2t + audio_mean
    total_w = text_w + audio_w
    print(f"\n  ── TEXT vs AUDIO (real data) ──")
    print(f"  TEXT  branches (text_self + t2a)              : "
          f"{text_w:.4f}  ({100*text_w/total_w:.1f}%)")
    print(f"  AUDIO branches (audio_self + a2t + audio_mean): "
          f"{audio_w:.4f}  ({100*audio_w/total_w:.1f}%)")

    # Correlation: gate weight vs true score
    print(f"\n  Pearson corr(gate_weight_i, true_score):")
    for i, name in enumerate(MODALITY_NAMES):
        col = gate_matrix[:, i]
        if col.std() > 1e-8 and y_true.std() > 1e-8:
            corr = float(np.corrcoef(col, y_true)[0, 1])
        else:
            corr = float("nan")
        print(f"    {name:<20} r = {corr:+.4f}")

    # Save raw gate values
    df_gates = pd.DataFrame(gate_matrix, columns=MODALITY_NAMES)
    df_gates["true_score"] = y_true
    df_gates.to_csv(out_dir / "gate_values_per_sample.csv", index=False)
    print(f"\n  Saved per-sample gate values → gate_values_per_sample.csv")

    summary = {
        "n_samples": N,
        "modality_mean": dict(zip(MODALITY_NAMES, mean_w.tolist())),
        "modality_std":  dict(zip(MODALITY_NAMES, std_w.tolist())),
        "modality_median": dict(zip(MODALITY_NAMES, med_w.tolist())),
        "text_pct":  float(100 * text_w / total_w),
        "audio_pct": float(100 * audio_w / total_w),
    }
    return summary


# ==================================================== METHOD B: Alpha Scale ==

def run_inference_with_alpha(model, loader, cfg, device, text_alpha, audio_alpha):
    """
    Monkey-patch gated_fusion.forward để scale text/audio TRƯỚC khi fuse.
    Text branches: text_self (arg0) + t2a (arg2)
    Audio branches: audio_self (arg1) + a2t (arg3) + audio_mean (arg4)
    """
    original_forward = None
    if hasattr(model, "gated_fusion"):
        original_forward = model.gated_fusion.forward

        def patched_forward(text_self, audio_self, t2a, a2t, audio_mean):
            return original_forward(
                text_self  * text_alpha,
                audio_self * audio_alpha,
                t2a        * text_alpha,
                a2t        * audio_alpha,
                audio_mean * audio_alpha,
            )
        model.gated_fusion.forward = patched_forward

    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in tqdm(loader,
                          desc=f"α_text={text_alpha:.2f} α_audio={audio_alpha:.2f}",
                          leave=False):
            kwargs = build_batch_kwargs(batch, cfg, device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(**kwargs)
            all_preds.append(outputs["expected_score"].detach().cpu().numpy())
            all_true.append(batch["score"].numpy())

    # Restore
    if original_forward is not None:
        model.gated_fusion.forward = original_forward

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    return y_pred, y_true


def run_ablation_study(model, loader, cfg, device, baseline_mae, edge_thr):
    print(f"\n{'─'*80}")
    print("  📊 ABLATION – Alpha Scaling Text / Audio Branches")
    print(f"{'─'*80}")
    print(f"  {'Config':<40} {'MAE':>7} {'RMSE':>7} {'QWK':>7} {'Acc@1.0':>8} {'ΔMAE':>8}")
    print(f"  {'─'*75}")

    results = []
    for text_alpha, audio_alpha, name in ABLATION_CONFIGS:
        y_pred_raw, y_true = run_inference_with_alpha(
            model, loader, cfg, device, text_alpha, audio_alpha
        )
        m = compute_metrics(y_true, y_pred_raw, edge_thr)
        m.update({"name": name, "text_alpha": text_alpha, "audio_alpha": audio_alpha})

        delta = m["mae"] - baseline_mae
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<40} {m['mae']:>7.4f} {m['rmse']:>7.4f} "
              f"{m['qwk']:>7.4f} {m['acc_1.0']:>8.4f} {sign}{delta:>7.4f}")
        results.append(m)

    print(f"  {'─'*75}")
    print(f"\n  📌 Baseline (α_text=1, α_audio=1) MAE = {baseline_mae:.4f}")

    # Text vs Audio impact
    audio_only = next((r for r in results if r["name"] == "02_audio_only__text=0"), None)
    text_only  = next((r for r in results if r["name"] == "03_text_only__audio=0"), None)
    if audio_only and text_only:
        print(f"\n  ── Modality Importance ──")
        print(f"  Removing TEXT  (audio only): MAE = {audio_only['mae']:.4f}  "
              f"(Δ = +{audio_only['mae'] - baseline_mae:.4f})")
        print(f"  Removing AUDIO (text only) : MAE = {text_only['mae']:.4f}  "
              f"(Δ = +{text_only['mae'] - baseline_mae:.4f})")
        if audio_only["mae"] > text_only["mae"]:
            print(f"  → TEXT is more important (removing it hurts more)")
        else:
            print(f"  → AUDIO is more important (removing it hurts more)")

    return results


# -------------------------------------------------------------------  main ---

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬  ABLATION: Text vs Audio Weight Analysis")
    print(f"    Checkpoint : {args.ckpt}")
    print(f"    Config     : {args.config}")
    print(f"    Split      : {args.split}  | Limit: {args.limit or 'all'}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ---- Config & processors ----
    cfg = Config.from_yaml(args.config)
    tokenizer = build_tokenizer(cfg.model.model_name)
    if cfg.model.audio_encoder_type.lower() == "whisper":
        audio_processor = WhisperProcessor.from_pretrained(cfg.model.audio_encoder_id)
    else:
        audio_processor = Wav2Vec2Processor.from_pretrained(cfg.model.audio_encoder_id)

    # ---- Load model ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    sig = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in cfg.model.__dict__.items() if k in valid_params}
    model = ESLGradingModelByCandidatesWithAudio(**model_kwargs).to(device)
    load_checkpoint_shape_safe(model, state_dict)
    model.eval()
    print(f"  ✅ Model loaded. Checkpoint val_mae: {ckpt.get('val_mae', 'N/A')}")

    # ---- Build loader ----
    csv_path = cfg.data.test_path if args.split == "test" else cfg.data.val_path
    print(f"\n  Loading {args.split} data: {csv_path}")
    loader = build_loader(csv_path, cfg, tokenizer, audio_processor, limit=args.limit)
    print(f"  Samples: {len(loader.dataset)}")

    edge_thr = cfg.training.edge_threshold
    all_output = {
        "checkpoint": args.ckpt,
        "split": args.split,
        "val_mae_ckpt": ckpt.get("val_mae"),
        "criteria": cfg.data.criteria,
    }

    # =========================================================
    # METHOD A: Live Gate Capture
    # =========================================================
    print(f"\n{'='*80}")
    print("  METHOD A – Live Gate Capture (real data)")
    print(f"{'='*80}")
    gate_matrix, y_true_gate = run_live_gate_capture(model, loader, cfg, device)
    if gate_matrix is not None:
        gate_summary = print_gate_summary(gate_matrix, y_true_gate, out_dir)
        all_output["live_gate_summary"] = gate_summary

    # =========================================================
    # METHOD B: Ablation Alpha Scaling
    # =========================================================
    if not args.skip_ablation:
        print(f"\n{'='*80}")
        print("  METHOD B – Alpha Scaling Ablation")
        print(f"{'='*80}")

        # Baseline first
        print(f"\n  Running baseline inference...")
        y_pred_base, y_true_base = run_inference_with_alpha(model, loader, cfg, device, 1.0, 1.0)
        baseline_metrics = compute_metrics(y_true_base, y_pred_base, edge_thr)
        baseline_mae = baseline_metrics["mae"]
        all_output["baseline_metrics"] = baseline_metrics

        ablation_results = run_ablation_study(model, loader, cfg, device, baseline_mae, edge_thr)
        all_output["ablation_results"] = ablation_results

        # Save CSV
        df_abl = pd.DataFrame(ablation_results)
        df_abl["delta_mae"] = df_abl["mae"] - baseline_mae
        df_abl = df_abl.sort_values("mae")
        csv_path_out = out_dir / f"ablation_results_{cfg.data.criteria}.csv"
        df_abl.to_csv(csv_path_out, index=False)
        print(f"\n  Saved ablation CSV → {csv_path_out}")

    # ---- Save JSON ----
    json_path = out_dir / f"ablation_full_{cfg.data.criteria}.json"
    with open(json_path, "w") as f:
        json.dump(all_output, f, indent=2, default=str)
    print(f"\n✅  All results saved → {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
