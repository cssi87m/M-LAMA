"""
ablation_qaware.py
------------------
Ablation Study: Does Question-Aware Cross-Attention help scoring?

EXPERIMENT (single, inference-only, no retraining):
────────────────────────────────────────────────────
Same checkpoint · Same weights · Same test set · ONE change only

  [A] BASELINE  – QuestionAwareEncoder normal:
        cross_attn(Q=response, K=question, V=question)
        → Response READS question context before pooling

  [B] ABLATED   – QuestionAwareEncoder FULLY BYPASSED:
        output = AttentionPool(response_features)  only
        → Question is COMPLETELY IGNORED
        → No question self-attn, no cross-attn, no gate
        → Pure response pooling only

WHY FULL BYPASS?
  Strongest possible ablation: model receives ZERO question signal.
  Uses the same response_pool weights already in the module.
  Any ΔMAE > 0 directly proves question context is useful.

RESEARCH QUESTION ANSWERED:
  ΔMAE = MAE(ABLATED) - MAE(BASELINE)
  ΔMAE > 0  →  removing question context HURTS  →  QA module is useful
  ΔMAE ≈ 0  →  model ignores question anyway    →  QA module has no effect

Usage:
    cd /home/user06/Interspeech_2026/Model
    python ablation_Q_aware/ablation_qaware.py
    python ablation_Q_aware/ablation_qaware.py \\
        --ckpt   Model/Model/checkpoints_fluency/model_best_mae_fluency_fusion_only_from_final_ckpt.pth \\
        --config config/config_fluency.yaml \\
        --split  test
"""

import argparse
import json
import math
import sys
import types
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperProcessor, Wav2Vec2Processor
import inspect

# ── resolve Model/ regardless of cwd ───────────────────────────────────────
_MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MODEL_ROOT))

from config import Config
from model import ESLGradingModelByCandidatesWithAudio, QuestionAwareEncoder
from dataloader import (
    ESLDatasetByCandidatesWithAudio,
    get_collate_fn_bycandidates_with_audio,
)

# ───────────────────────────────────────────────── default paths ────────────
_CKPT_DEFAULT   = str(_MODEL_ROOT / "Model/checkpoints_fluency/"
                      "model_best_mae_fluency_fusion_only_from_final_ckpt.pth")
_CONFIG_DEFAULT = str(_MODEL_ROOT / "config/config_fluency.yaml")
_OUT_DEFAULT    = str(Path(__file__).resolve().parent / "results")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MONKEY-PATCH  –  ablate cross-attention inside QuestionAwareEncoder
# ══════════════════════════════════════════════════════════════════════════════

def _ablated_forward(self, question_features, response_features,
                     question_mask=None, response_mask=None):
    """
    FULL BYPASS of QuestionAwareEncoder.

    Completely skips:
      - Question self-attention
      - Cross-attention (response → question)
      - Gated fusion (question_pooled + response_pooled)

    Only does: AttentionPool(response_features)
    → output = pure response representation, ZERO question signal

    This is the strongest possible ablation to isolate question contribution.
    """
    # Bypass everything — pool response directly, ignore question entirely
    return self.response_pool(response_features, response_mask)


def patch_qa_encoder(model: torch.nn.Module, ablate: bool) -> list[str]:
    """
    Ablate=True  → replace forward with _ablated_forward (no question signal).
    Ablate=False → restore original forward.
    Returns list of patched module names.
    """
    patched = []
    for name, module in model.named_modules():
        if not isinstance(module, QuestionAwareEncoder):
            continue
        if ablate:
            if not hasattr(module, "_orig_forward"):
                module._orig_forward = module.forward
            module.forward = types.MethodType(_ablated_forward, module)
        else:
            if hasattr(module, "_orig_forward"):
                module.forward = module._orig_forward
        patched.append(name)
    return patched


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_ckpt_shape_safe(model, state_dict):
    cur = model.state_dict()
    ok, skip = {}, []
    for k, v in state_dict.items():
        if k in cur and cur[k].shape == v.shape:
            ok[k] = v
        else:
            skip.append(k)
    cur.update(ok)
    model.load_state_dict(cur)
    return len(ok), len(skip)


def build_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    tok.padding_side = "right"
    return tok


def build_loader(cfg: Config, tokenizer, audio_proc, split: str, limit=None):
    path = {"train": cfg.data.train_path,
             "val":   cfg.data.val_path,
             "test":  cfg.data.test_path}[split]
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)

    n_chunks = cfg.audio.eval_num_chunks or cfg.audio.num_chunks
    dataset  = ESLDatasetByCandidatesWithAudio(
        df,
        criteria=cfg.data.criteria,
        audio_processor=audio_proc,
        encoder_type=cfg.model.audio_encoder_type,
        num_chunks=n_chunks,
        chunk_length_sec=cfg.audio.chunk_length_sec,
        separate_question_response=cfg.model.use_question_encoder,
    )
    collate  = get_collate_fn_bycandidates_with_audio(
        tokenizer,
        max_length=cfg.data.max_length,
        max_audio_chunks=cfg.audio.max_audio_chunks,
        max_waveform_len=cfg.audio.max_waveform_len,
        separate_tokenize=cfg.model.use_question_encoder,
    )
    bs = cfg.training.eval_batch_size or cfg.training.batch_size
    return DataLoader(dataset, batch_size=bs, shuffle=False,
                      collate_fn=collate,
                      num_workers=cfg.data.num_workers,
                      prefetch_factor=cfg.data.prefetch_factor,
                      pin_memory=True)


def qwk(y_true, y_pred) -> float:
    bt = np.round(y_true * 2) / 2
    bp = np.round(y_pred * 2) / 2
    return float(cohen_kappa_score(
        np.round(bt * 2).astype(int),
        np.round(bp * 2).astype(int),
        weights="quadratic",
    ))


def metrics(y_true, y_pred_raw, edge_thr=3.5) -> dict:
    yp = np.clip(np.round(y_pred_raw * 2) / 2, 0, 10)
    e  = yp - y_true
    ae = np.abs(e)
    em = (y_true <= edge_thr) | (y_true >= 10 - edge_thr)
    mm = ~em
    return {
        "mae":       float(ae.mean()),
        "rmse":      float(math.sqrt((e**2).mean())),
        "qwk":       qwk(y_true, yp),
        "acc_0.5":   float((ae <= 0.5).mean()),
        "acc_1.0":   float((ae <= 1.0).mean()),
        "edge_mae":  float(ae[em].mean()) if em.any() else float("nan"),
        "mid_mae":   float(ae[mm].mean()) if mm.any() else float("nan"),
        "n":         int(len(y_true)),
        "n_edge":    int(em.sum()),
        "n_mid":     int(mm.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  INFERENCE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, loader, cfg: Config, device: str, label: str):
    model.eval()
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  [{label}]", leave=True):
            audio = batch["audio"].to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                if cfg.model.use_question_encoder:
                    out = model(
                        question_input_ids      =batch["question_input_ids"].to(device),
                        question_attention_mask =batch["question_attention_mask"].to(device),
                        response_input_ids      =batch["response_input_ids"].to(device),
                        response_attention_mask =batch["response_attention_mask"].to(device),
                        audio=audio,
                    )
                else:
                    out = model(
                        input_ids      =batch["input_ids"].to(device),
                        attention_mask =batch["attention_mask"].to(device),
                        audio=audio,
                    )
            preds.append(out["expected_score"].detach().cpu().numpy())
            trues.append(batch["score"].numpy())
            ids.extend(batch["candidate_id"])
    return np.concatenate(preds), np.concatenate(trues), ids


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(m_base, m_abl):
    metrics_order = ["mae", "rmse", "qwk", "acc_0.5", "acc_1.0",
                     "edge_mae", "mid_mae"]
    # for MAE/RMSE/edge/mid: higher ablated = QA helps
    # for QWK/acc: lower ablated = QA helps
    higher_is_worse = {"mae", "rmse", "edge_mae", "mid_mae"}

    print(f"\n  {'Metric':<14} {'BASELINE':>11} {'ABLATED':>11} "
          f"{'Δ (abl−base)':>14}  {'QA helps?':>10}")
    print(f"  {'─'*64}")
    for k in metrics_order:
        b = m_base[k]; a = m_abl[k]
        d = a - b
        if k in higher_is_worse:
            verdict = "✅ YES" if d > 0.005 else ("❌ NO" if d < -0.005 else "≈ equal")
        else:
            verdict = "✅ YES" if d < -0.005 else ("❌ NO" if d > 0.005 else "≈ equal")
        print(f"  {k:<14} {b:>11.4f} {a:>11.4f} {d:>+14.4f}  {verdict:>10}")


def score_range_breakdown(y_true, yp_base, yp_abl):
    ranges = [
        ("Low  (≤4.0)",  y_true <= 4.0),
        ("Mid  (4–7)",   (y_true > 4.0) & (y_true <= 7.0)),
        ("High (>7.0)",  y_true > 7.0),
    ]
    rows = []
    print(f"\n  {'Score Range':<15} {'N':>5}  {'MAE_base':>10}  "
          f"{'MAE_abl':>10}  {'Δ MAE':>10}  {'QA?':>8}")
    print(f"  {'─'*65}")
    for name, mask in ranges:
        if not mask.any():
            continue
        r_base = np.clip(np.round(yp_base[mask] * 2) / 2, 0, 10)
        r_abl  = np.clip(np.round(yp_abl[mask]  * 2) / 2, 0, 10)
        mb = float(np.abs(r_base - y_true[mask]).mean())
        ma = float(np.abs(r_abl  - y_true[mask]).mean())
        d  = ma - mb
        qa = "✅ YES" if d > 0.01 else ("❌ NO" if d < -0.01 else "≈ equal")
        print(f"  {name:<15} {mask.sum():>5}  {mb:>10.4f}  {ma:>10.4f}  {d:>+10.4f}  {qa:>8}")
        rows.append({"range": name, "n": int(mask.sum()),
                     "mae_baseline": mb, "mae_ablated": ma, "delta_mae": d})
    return rows


def per_sample_analysis(y_true, yp_base, yp_abl, out_dir: Path):
    r_base = np.clip(np.round(yp_base * 2) / 2, 0, 10)
    r_abl  = np.clip(np.round(yp_abl  * 2) / 2, 0, 10)
    ae_b   = np.abs(r_base - y_true)
    ae_a   = np.abs(r_abl  - y_true)
    # delta > 0 means ablated is WORSE → QA helped this sample
    delta  = ae_a - ae_b

    n_help = int((delta > 0).sum())
    n_hurt = int((delta < 0).sum())
    n_same = int((delta == 0).sum())
    pct    = 100 * n_help / max(len(delta), 1)

    print(f"\n  Per-sample breakdown  (N={len(y_true)}):")
    print(f"    QA helps  (ablated worse): {n_help:5d}  ({pct:.1f}%)")
    print(f"    QA hurts  (ablated better): {n_hurt:5d}  ({100*n_hurt/len(delta):.1f}%)")
    print(f"    No change               : {n_same:5d}")
    print(f"    Mean improvement / sample: {delta.mean():+.4f}")

    return {"n_helped": n_help, "n_hurt": n_hurt, "n_same": n_same,
            "pct_helped": pct, "mean_delta": float(delta.mean())}


def cross_attn_weight_stats(model):
    """
    Print the Frobenius norm of question_cross_attn in_proj weights –
    confirms how strongly the model has trained on question→response attention.
    """
    print(f"\n  question_cross_attn weight norms (larger = more trained):")
    for name, mod in model.named_modules():
        if isinstance(mod, QuestionAwareEncoder):
            for pname, param in mod.question_cross_attn.named_parameters():
                n = float(param.detach().cpu().float().norm())
                print(f"    {name}.question_cross_attn.{pname:<35}  norm={n:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default=_CKPT_DEFAULT)
    p.add_argument("--config",  default=_CONFIG_DEFAULT)
    p.add_argument("--split",   default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", default=_OUT_DEFAULT)
    p.add_argument("--limit",   type=int, default=None,
                   help="Limit N samples for quick debugging")
    p.add_argument("--skip_baseline", action="store_true",
                   help="Load baseline from existing qaware_predictions.csv instead of re-running")
    p.add_argument("--ablation_only", action="store_true",
                   help="Run ONLY the ablated condition (skip baseline inference entirely). "
                        "Outputs ablated metrics only — no ΔMAE comparison.")
    return p.parse_args()


def main():
    args  = parse_args()
    odir  = Path(args.out_dir); odir.mkdir(parents=True, exist_ok=True)
    dev   = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  ABLATION: Question-Aware Cross-Attention")
    print("  Single experiment  ·  inference only  ·  no retraining")
    print("=" * 70)
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Config     : {args.config}")
    print(f"  Split      : {args.split}  |  Limit: {args.limit or 'all'}")
    print(f"  Device     : {dev}")
    print("=" * 70)

    # ── config ──────────────────────────────────────────────────────────────
    cfg = Config.from_yaml(args.config)

    if not cfg.model.use_question_encoder:
        print("\n  ⚠  WARNING: use_question_encoder=False in config.")
        print("     QuestionAwareEncoder is not active → ablation will be a no-op.")

    # ── processors ──────────────────────────────────────────────────────────
    tokenizer   = build_tokenizer(cfg.model.model_name)
    audio_proc  = (WhisperProcessor.from_pretrained(cfg.model.audio_encoder_id)
                   if cfg.model.audio_encoder_type.lower() == "whisper"
                   else Wav2Vec2Processor.from_pretrained(cfg.model.audio_encoder_id))

    # ── load checkpoint & build model ───────────────────────────────────────
    print(f"\n  Loading checkpoint …")
    ckpt  = torch.load(args.ckpt, map_location="cpu")
    sd    = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    sig         = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
    valid       = set(sig.parameters) - {"self"}
    model_kw    = {k: v for k, v in cfg.model.__dict__.items() if k in valid}
    model       = ESLGradingModelByCandidatesWithAudio(**model_kw).to(dev)
    n_ok, n_sk  = load_ckpt_shape_safe(model, sd)
    model.eval()
    print(f"  Loaded {n_ok} keys  |  skipped {n_sk} keys")
    print(f"  Checkpoint val_mae={ckpt.get('val_mae','N/A')}  "
          f"epoch={ckpt.get('epoch','N/A')}")

    # ── verify QA modules ────────────────────────────────────────────────────
    qa_mods = [(n, m) for n, m in model.named_modules()
               if isinstance(m, QuestionAwareEncoder)]
    print(f"\n  Found {len(qa_mods)} QuestionAwareEncoder module(s): "
          f"{[n for n,_ in qa_mods]}")

    # Print weight norms for context
    cross_attn_weight_stats(model)

    # ── data loader (shared by both conditions) ───────────────────────────────
    print(f"\n  Loading {args.split} data …")
    loader = build_loader(cfg, tokenizer, audio_proc, args.split, args.limit)
    print(f"  Dataset: {len(loader.dataset)} samples")

    edge_thr = cfg.training.edge_threshold

    # ════════════════════════════════════════════════════════════════════════
    # FAST PATH: --ablation_only  (skip baseline inference entirely)
    # ════════════════════════════════════════════════════════════════════════
    if args.ablation_only:
        print(f"\n{'═'*70}")
        print("  MODE: ABLATION-ONLY  (baseline skipped)")
        print("  QuestionAwareEncoder FULLY BYPASSED")
        print("  output = AttentionPool(response_features) only")
        print(f"{'═'*70}")

        patched = patch_qa_encoder(model, ablate=True)
        if patched:
            print(f"  ✅ Patched: {patched}")
        else:
            print("  ⚠  No modules patched.")

        yp_abl, yt, cids = run_inference(model, loader, cfg, dev, "ABLATED")
        patch_qa_encoder(model, ablate=False)
        print(f"  ✅ Original forward restored.")

        m_abl = metrics(yt, yp_abl, edge_thr)
        print(f"\n{'═'*70}")
        print(f"  📊  ABLATED METRICS  (no question signal)")
        print(f"{'═'*70}")
        print(f"  MAE     = {m_abl['mae']:.4f}")
        print(f"  RMSE    = {m_abl['rmse']:.4f}")
        print(f"  QWK     = {m_abl['qwk']:.4f}")
        print(f"  Acc@0.5 = {m_abl['acc_0.5']:.4f}")
        print(f"  Acc@1.0 = {m_abl['acc_1.0']:.4f}")
        print(f"  edge_MAE= {m_abl['edge_mae']:.4f}")
        print(f"  mid_MAE = {m_abl['mid_mae']:.4f}")
        print(f"{'═'*70}")

        # Save
        yp_abl_r = np.clip(np.round(yp_abl * 2) / 2, 0, 10)
        pd.DataFrame({
            "candidate_id":     cids,
            "y_true":           yt,
            "pred_ablated_raw": yp_abl,
            "pred_ablated":     yp_abl_r,
            "err_ablated":      np.abs(yp_abl_r - yt),
        }).to_csv(odir / "ablated_predictions.csv", index=False)

        summary = {
            "experiment":      "Question-Aware Cross-Attention Ablation (ablation-only)",
            "ablation_method": "Full bypass: output = AttentionPool(response_features) only",
            "checkpoint":      args.ckpt,
            "ckpt_val_mae":    str(ckpt.get("val_mae", "N/A")),
            "split":           args.split,
            "n_samples":       int(len(yt)),
            "qa_modules":      [n for n, _ in qa_mods],
            "edge_threshold":  edge_thr,
            "metrics_ablated": m_abl,
        }
        jp = odir / "ablated_results.json"
        with open(jp, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n  ✅  Saved:")
        print(f"       {odir}/ablated_results.json")
        print(f"       {odir}/ablated_predictions.csv")
        print("=" * 70)
        return

    # ════════════════════════════════════════════════════════════════════════
    # CONDITION A – BASELINE (load from CSV or run inference)
    # ════════════════════════════════════════════════════════════════════════
    baseline_csv = odir / "qaware_predictions.csv"

    if args.skip_baseline and baseline_csv.exists():
        print(f"\n{'─'*70}")
        print(f"  [A] BASELINE  – Loading from existing CSV (skipping inference)")
        print(f"      {baseline_csv}")
        print(f"{'─'*70}")
        df_prev  = pd.read_csv(baseline_csv)
        yt       = df_prev["y_true"].values
        yp_base  = df_prev["pred_baseline_raw"].values
        cids     = df_prev["candidate_id"].tolist()
        m_base   = metrics(yt, yp_base, edge_thr)
        print(f"  Loaded {len(yt)} samples from CSV.")
    else:
        print(f"\n{'─'*70}")
        print(f"  [A] BASELINE  – QuestionAwareEncoder FULLY ON")
        print(f"      cross_attn(Q=response, K=question, V=question) + gate fusion")
        print(f"{'─'*70}")
        patch_qa_encoder(model, ablate=False)          # ensure clean state
        yp_base, yt, cids = run_inference(model, loader, cfg, dev, "BASELINE")
        m_base = metrics(yt, yp_base, edge_thr)

    print(f"\n  MAE={m_base['mae']:.4f}  RMSE={m_base['rmse']:.4f}  "
          f"QWK={m_base['qwk']:.4f}  Acc@0.5={m_base['acc_0.5']:.4f}  "
          f"Acc@1.0={m_base['acc_1.0']:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # CONDITION B – ABLATED (response self-attn, no question signal)
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  [B] ABLATED   – QuestionAwareEncoder FULLY BYPASSED")
    print("      output = AttentionPool(response_features) only")
    print("      → ZERO question signal reaches text_self_pooled")
    print(f"{'─'*70}")

    patched = patch_qa_encoder(model, ablate=True)
    if patched:
        print(f"  ✅ Patched: {patched}")
    else:
        print("  ⚠  No modules patched.")

    yp_abl, yt_abl, _ = run_inference(model, loader, cfg, dev, "ABLATED")

    # restore immediately
    patch_qa_encoder(model, ablate=False)
    print(f"  ✅ Original forward restored.")

    m_abl = metrics(yt_abl, yp_abl, edge_thr)

    print(f"\n  MAE={m_abl['mae']:.4f}  RMSE={m_abl['rmse']:.4f}  "
          f"QWK={m_abl['qwk']:.4f}  Acc@0.5={m_abl['acc_0.5']:.4f}  "
          f"Acc@1.0={m_abl['acc_1.0']:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  📊  COMPARISON: BASELINE  vs  ABLATED")
    print(f"{'═'*70}")
    print_comparison_table(m_base, m_abl)

    # Verdict
    d_mae = m_abl["mae"] - m_base["mae"]
    print(f"\n{'═'*70}")
    if   d_mae >  0.05: verdict = f"✅ STRONG  – QA cross-attention significantly helps  (ΔMAE=+{d_mae:.4f})"
    elif d_mae >  0.01: verdict = f"✅ MODERATE – QA cross-attention helps               (ΔMAE=+{d_mae:.4f})"
    elif d_mae < -0.01: verdict = f"❌ HURTS   – removing QA actually improves score     (ΔMAE={d_mae:.4f})"
    else:               verdict = f"⚠  NEUTRAL – QA cross-attention has minimal effect  (ΔMAE={d_mae:.4f})"
    print(f"  VERDICT: {verdict}")
    print(f"{'═'*70}")

    # ── per score-range breakdown ──────────────────────────────────────────
    print(f"\n  📊  SCORE-RANGE BREAKDOWN:")
    range_rows = score_range_breakdown(yt, yp_base, yp_abl)

    # ── per-sample analysis + save CSV ────────────────────────────────────
    sample_stats = per_sample_analysis(yt, yp_base, yp_abl, odir)

    # ── save full predictions CSV ─────────────────────────────────────────
    yp_base_r = np.clip(np.round(yp_base * 2) / 2, 0, 10)
    yp_abl_r  = np.clip(np.round(yp_abl  * 2) / 2, 0, 10)
    pd.DataFrame({
        "candidate_id":       cids,
        "y_true":             yt,
        "pred_baseline_raw":  yp_base,
        "pred_ablated_raw":   yp_abl,
        "pred_baseline":      yp_base_r,
        "pred_ablated":       yp_abl_r,
        "err_baseline":       np.abs(yp_base_r - yt),
        "err_ablated":        np.abs(yp_abl_r  - yt),
        "qa_improvement":     np.abs(yp_abl_r - yt) - np.abs(yp_base_r - yt),
    }).to_csv(odir / "qaware_predictions.csv", index=False)

    # ── save JSON summary ─────────────────────────────────────────────────
    summary = {
        "experiment": "Question-Aware Cross-Attention Ablation",
        "ablation_method": (
            "Full bypass of QuestionAwareEncoder: "
            "output = AttentionPool(response_features) only. "
            "Question self-attn, cross-attn, and gate are all skipped."
        ),
        "checkpoint":     args.ckpt,
        "ckpt_val_mae":   str(ckpt.get("val_mae", "N/A")),
        "split":          args.split,
        "n_samples":      int(len(yt)),
        "qa_modules":     [n for n, _ in qa_mods],
        "edge_threshold": edge_thr,
        "metrics_baseline": m_base,
        "metrics_ablated":  m_abl,
        "delta_mae":       d_mae,
        "verdict":         verdict,
        "score_range_analysis": range_rows,
        "per_sample_analysis":  sample_stats,
    }
    jp = odir / "qaware_results.json"
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  ✅  Saved:")
    print(f"       {odir}/qaware_results.json")
    print(f"       {odir}/qaware_predictions.csv")
    print(f"       {odir}/per_sample_comparison.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
