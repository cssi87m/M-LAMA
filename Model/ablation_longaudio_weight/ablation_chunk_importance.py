"""
Ablation Study: Long Audio Chunk Importance
===========================================

Goal: Identify which temporal chunk(s) of the audio input contribute most
to the model's predicted score.

Three complementary ablation strategies:

1. CHUNK-ZERO  — Individually zero out each chunk and measure score delta
                 (high |delta| -> that chunk matters a lot)

2. PART-ZERO   — Zero out all chunks belonging to Part 1 / 2 / 3 entirely
                 (shows which of the 3 spoken parts is most informative)

3. PROGRESSIVE-KEEP — Keep only the first N chunks (zero the rest) and
                      record metrics for N = 1, 2, ..., total_chunks
                      (reveals the temporal importance curve)

Outputs (saved to --output_dir):
  chunk_zero_deltas.csv       -- per-sample score deltas when each chunk is zeroed
  chunk_zero_summary.csv      -- mean |delta| per chunk position + std + rank
  part_zero_deltas.csv        -- per-sample score deltas when each part is zeroed
  part_zero_summary.csv       -- mean |delta| per part + rank
  progressive_metrics.csv     -- MAE / QWK vs. number of kept chunks
  ablation_summary.json       -- merged summary of all experiments

Usage:
  python ablation_chunk_importance.py \\
      --config ../config.yaml \\
      --checkpoint /path/to/model.pth \\
      --output_dir ./results \\
      [--split test|val|train] \\
      [--limit 200] \\
      [--skip_chunk_zero] \\
      [--skip_part_zero] \\
      [--skip_progressive_keep]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Wav2Vec2Processor, WhisperProcessor

# Make parent directory importable so config / model / dataloader can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from dataloader import ESLDatasetByCandidatesWithAudio, get_collate_fn_bycandidates_with_audio
from model import ESLGradingModelByCandidatesWithAudio


# =============================================================================
# Shared helpers
# =============================================================================

def load_checkpoint_shape_safe(model: torch.nn.Module, state_dict: dict):
    """Load state dict, skipping keys whose shape does not match."""
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
        print(f"[WARN] Skipped {len(skipped)} keys (shape/name mismatch). "
              f"First 5: {skipped[:5]}")


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    tok.padding_side = "right"
    return tok


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    bins_true = np.round(y_true * 2) / 2.0
    bins_pred = np.round(y_pred * 2) / 2.0
    ints_true = np.round(bins_true * 2).astype(int)
    ints_pred = np.round(bins_pred * 2).astype(int)
    return float(cohen_kappa_score(ints_true, ints_pred, weights="quadratic"))


def compute_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray, edge_thr: float) -> dict:
    """Round predictions to 0.5 steps then compute standard metrics."""
    y_pred     = np.clip(np.round(y_pred_raw * 2) / 2, 0, 10)
    err        = y_pred - y_true
    edge_mask  = (y_true <= edge_thr) | (y_true >= (10 - edge_thr))
    mid_mask   = ~edge_mask
    return {
        "mae":      float(np.mean(np.abs(err))),
        "rmse":     float(math.sqrt(np.mean(err ** 2))),
        "qwk":      quadratic_weighted_kappa(y_true, y_pred),
        "edge_mae": float(np.mean(np.abs(err[edge_mask]))) if edge_mask.any()  else float("nan"),
        "mid_mae":  float(np.mean(np.abs(err[mid_mask])))  if mid_mask.any()   else float("nan"),
        "n":        int(len(y_true)),
    }


def build_loader(csv_path: str, cfg: Config, tokenizer, audio_processor, limit=None):
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    num_chunks = (
        cfg.audio.eval_num_chunks
        if cfg.audio.eval_num_chunks is not None
        else cfg.audio.num_chunks
    )

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
    batch_size = (
        cfg.training.eval_batch_size
        if cfg.training.eval_batch_size is not None
        else cfg.training.batch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=True,
    )


# =============================================================================
# Core inference pass
# =============================================================================

@torch.no_grad()
def run_inference(model, loader, device, cfg, audio_override_fn=None):
    """
    Full inference pass over the loader.

    audio_override_fn : callable(audio_tensor) -> audio_tensor  [optional]
        Applied in-place to the audio batch before each forward pass.
        Use this to zero / mask specific chunks.

    Returns
    -------
    preds        : np.ndarray [n_samples]   raw (unrounded) predicted scores
    y_true       : np.ndarray [n_samples]
    candidate_ids: list[str]  [n_samples]
    """
    model.eval()
    all_preds, all_true, all_ids = [], [], []

    for batch in tqdm(loader):
        audio = batch["audio"].to(device)

        if audio_override_fn is not None:
            audio = audio_override_fn(audio)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            if cfg.model.use_question_encoder:
                outputs = model(
                    question_input_ids=batch["question_input_ids"].to(device),
                    question_attention_mask=batch["question_attention_mask"].to(device),
                    response_input_ids=batch["response_input_ids"].to(device),
                    response_attention_mask=batch["response_attention_mask"].to(device),
                    audio=audio,
                )
            else:
                outputs = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    audio,
                )

        all_preds.append(outputs["expected_score"].detach().cpu().numpy())
        all_true.append(batch["score"].numpy())
        all_ids.extend(batch["candidate_id"])

    return (
        np.concatenate(all_preds),
        np.concatenate(all_true),
        all_ids,
    )


# =============================================================================
# Ablation 1 — Chunk-level zeroing
# =============================================================================

def ablation_chunk_zero(model, loader, device, cfg, total_chunks, baseline_preds):
    """
    For each chunk index c in [0, total_chunks):
      - zero that chunk across ALL samples in the loader
      - record per-sample score delta = pred_zeroed - pred_baseline

    Returns
    -------
    delta_matrix : np.ndarray [n_samples, total_chunks]
    """
    n_samples    = len(baseline_preds)
    delta_matrix = np.zeros((n_samples, total_chunks), dtype=np.float32)

    print(f"\n[Chunk-Zero] Running {total_chunks} individual chunk masks ...")
    for c in tqdm(range(total_chunks), desc="Chunk-Zero"):

        def zero_chunk(audio, _c=c):
            a = audio.clone()
            a[:, _c] = 0.0
            return a

        preds_c, _, _ = run_inference(model, loader, device, cfg,
                                      audio_override_fn=zero_chunk)
        delta_matrix[:, c] = preds_c - baseline_preds

    return delta_matrix


# =============================================================================
# Ablation 2 — Part-level zeroing
# =============================================================================

def ablation_part_zero(model, loader, device, cfg,
                       num_parts, chunks_per_part, baseline_preds):
    """
    For each part p in [0, num_parts):
      - zero ALL chunks belonging to that part
      - record per-sample score delta

    Returns
    -------
    part_delta_matrix : np.ndarray [n_samples, num_parts]
    """
    n_samples         = len(baseline_preds)
    part_delta_matrix = np.zeros((n_samples, num_parts), dtype=np.float32)

    print(f"\n[Part-Zero] Running {num_parts} part masks "
          f"({chunks_per_part} chunks each) ...")
    for p in tqdm(range(num_parts), desc="Part-Zero"):
        start = p * chunks_per_part
        end   = start + chunks_per_part

        def zero_part(audio, _s=start, _e=end):
            a = audio.clone()
            a[:, _s:_e] = 0.0
            return a

        preds_p, _, _ = run_inference(model, loader, device, cfg,
                                      audio_override_fn=zero_part)
        part_delta_matrix[:, p] = preds_p - baseline_preds

    return part_delta_matrix


# =============================================================================
# Ablation 3 — Progressive keep
# =============================================================================

def ablation_progressive_keep(model, loader, device, cfg,
                               total_chunks, y_true, edge_thr):
    """
    Keep only the first N chunks (zero the rest) for N = 1 ... total_chunks.
    Computes full metrics at each N.

    Returns
    -------
    records : list[dict]  one entry per N value
    """
    records = []
    print(f"\n[Progressive-Keep] Sweeping N = 1 ... {total_chunks} ...")
    for n_keep in tqdm(range(1, total_chunks + 1), desc="Progressive-Keep"):

        def keep_n(audio, _n=n_keep):
            a = audio.clone()
            a[:, _n:] = 0.0
            return a

        preds_n, _, _ = run_inference(model, loader, device, cfg,
                                      audio_override_fn=keep_n)
        m = compute_metrics(y_true, preds_n, edge_thr)
        m["n_chunks_kept"] = n_keep
        records.append(m)

    return records


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation study: which audio chunks matter most?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_cfg = Path(__file__).resolve().parent.parent / "config.yaml"
    p.add_argument("--config",     type=str, default=str(default_cfg),
                   help="Path to config.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pth checkpoint (overrides config if given)")
    p.add_argument("--split",      type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Which CSV split to evaluate on")
    p.add_argument("--output_dir", type=str, default="./ablation_results",
                   help="Directory to write output CSVs and JSON")
    p.add_argument("--limit",      type=int, default=None,
                   help="Row limit for quick smoke-test (e.g. --limit 50)")
    # Flags to skip individual strategies (useful for quick partial runs)
    p.add_argument("--skip_chunk_zero",       action="store_true",
                   help="Skip chunk-level zeroing (runs total_chunks passes)")
    p.add_argument("--skip_part_zero",        action="store_true",
                   help="Skip part-level zeroing")
    p.add_argument("--skip_progressive_keep", action="store_true",
                   help="Skip progressive-keep sweep")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    cfg    = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(cfg.checkpoint.load_checkpoint)
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint : {ckpt_path}")

    ckpt       = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    # ------------------------------------------------------------------
    # Tokenizer + audio processor
    # ------------------------------------------------------------------
    tokenizer = build_tokenizer(cfg.model.model_name)
    if cfg.model.audio_encoder_type.lower() == "whisper":
        audio_processor = WhisperProcessor.from_pretrained(cfg.model.audio_encoder_id)
    else:
        audio_processor = Wav2Vec2Processor.from_pretrained(cfg.model.audio_encoder_id)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    import inspect
    sig        = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    model_kw   = {k: v for k, v in cfg.model.__dict__.items() if k in valid_keys}
    model      = ESLGradingModelByCandidatesWithAudio(**model_kw).to(device)
    load_checkpoint_shape_safe(model, state_dict)
    model.eval()
    print("Model      : loaded OK")

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    split_path_map = {
        "train": cfg.data.train_path,
        "val":   cfg.data.val_path,
        "test":  cfg.data.test_path,
    }
    csv_path = split_path_map[args.split]
    loader   = build_loader(csv_path, cfg, tokenizer, audio_processor,
                            limit=args.limit)
    print(f"Split      : {args.split}  ({csv_path})")

    # ------------------------------------------------------------------
    # Chunk / part dimensions
    # ------------------------------------------------------------------
    num_chunks_per_part = (
        cfg.audio.eval_num_chunks
        if cfg.audio.eval_num_chunks is not None
        else cfg.audio.num_chunks
    )
    num_parts    = cfg.model.num_parts
    total_chunks = num_parts * num_chunks_per_part
    edge_thr     = cfg.training.edge_threshold

    print(f"Layout     : {num_parts} parts x {num_chunks_per_part} chunks"
          f" = {total_chunks} total chunks")
    print(f"Edge thr   : {edge_thr}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # BASELINE — full audio, no masking
    # ==================================================================
    print("\n[Baseline] Full-audio inference ...")
    baseline_preds, y_true, candidate_ids = run_inference(
        model, loader, device, cfg
    )
    baseline_metrics = compute_metrics(y_true, baseline_preds, edge_thr)
    print(f"  MAE  = {baseline_metrics['mae']:.4f}")
    print(f"  RMSE = {baseline_metrics['rmse']:.4f}")
    print(f"  QWK  = {baseline_metrics['qwk']:.4f}")
    print(f"  n    = {baseline_metrics['n']}")

    all_results = {
        "split":      args.split,
        "checkpoint": str(ckpt_path),
        "baseline":   baseline_metrics,
        "layout": {
            "num_parts":            num_parts,
            "chunks_per_part":      num_chunks_per_part,
            "total_chunks":         total_chunks,
        },
    }

    # ==================================================================
    # ABLATION 1 — Chunk-Zero
    # ==================================================================
    if not args.skip_chunk_zero:
        delta_matrix = ablation_chunk_zero(
            model, loader, device, cfg, total_chunks, baseline_preds
        )

        # Per-sample CSV
        col_names  = [f"chunk_{c:02d}_delta" for c in range(total_chunks)]
        df_deltas  = pd.DataFrame(delta_matrix, columns=col_names)
        df_deltas.insert(0, "Candidate_ID",   candidate_ids)
        df_deltas.insert(1, "true_score",     y_true)
        df_deltas.insert(2, "baseline_pred",  np.round(baseline_preds * 2) / 2)
        df_deltas.to_csv(out_dir / "chunk_zero_deltas.csv", index=False)

        # Summary CSV
        abs_mean   = np.abs(delta_matrix).mean(axis=0)
        abs_std    = np.abs(delta_matrix).std(axis=0)
        signed_mean = delta_matrix.mean(axis=0)

        part_labels    = [f"part_{c // num_chunks_per_part + 1}" for c in range(total_chunks)]
        chunk_in_part  = [c % num_chunks_per_part                for c in range(total_chunks)]

        df_summary = pd.DataFrame({
            "chunk_idx":         range(total_chunks),
            "part":              part_labels,
            "chunk_in_part":     chunk_in_part,
            "abs_delta_mean":    abs_mean,
            "abs_delta_std":     abs_std,
            "signed_delta_mean": signed_mean,
            "importance_rank":   pd.Series(abs_mean)
                                   .rank(ascending=False)
                                   .astype(int)
                                   .values,
        })
        df_summary.to_csv(out_dir / "chunk_zero_summary.csv", index=False)

        top5 = df_summary.nsmallest(5, "importance_rank")[
            ["chunk_idx", "part", "chunk_in_part", "abs_delta_mean", "importance_rank"]
        ]
        print("\n[Chunk-Zero] Top-5 most important chunks (highest mean |delta|):")
        print(top5.to_string(index=False))

        all_results["chunk_zero"] = {
            "most_important_chunk_idx": int(abs_mean.argmax()),
            "mean_abs_delta_per_chunk": abs_mean.tolist(),
            "top5_chunks":              top5.to_dict(orient="records"),
        }

    # ==================================================================
    # ABLATION 2 — Part-Zero
    # ==================================================================
    if not args.skip_part_zero:
        part_delta_matrix = ablation_part_zero(
            model, loader, device, cfg,
            num_parts, num_chunks_per_part, baseline_preds
        )

        col_names      = [f"part_{p+1}_delta" for p in range(num_parts)]
        df_part_deltas = pd.DataFrame(part_delta_matrix, columns=col_names)
        df_part_deltas.insert(0, "Candidate_ID",  candidate_ids)
        df_part_deltas.insert(1, "true_score",    y_true)
        df_part_deltas.insert(2, "baseline_pred", np.round(baseline_preds * 2) / 2)
        df_part_deltas.to_csv(out_dir / "part_zero_deltas.csv", index=False)

        abs_part_mean  = np.abs(part_delta_matrix).mean(axis=0)
        abs_part_std   = np.abs(part_delta_matrix).std(axis=0)
        signed_part    = part_delta_matrix.mean(axis=0)

        df_part_summary = pd.DataFrame({
            "part":              [f"Part {p+1}" for p in range(num_parts)],
            "abs_delta_mean":    abs_part_mean,
            "abs_delta_std":     abs_part_std,
            "signed_delta_mean": signed_part,
            "importance_rank":   pd.Series(abs_part_mean)
                                   .rank(ascending=False)
                                   .astype(int)
                                   .values,
        })
        df_part_summary.to_csv(out_dir / "part_zero_summary.csv", index=False)

        print("\n[Part-Zero] Part importance summary:")
        print(df_part_summary.to_string(index=False))

        all_results["part_zero"] = {
            "most_important_part":       int(abs_part_mean.argmax() + 1),
            "abs_delta_mean_per_part":   abs_part_mean.tolist(),
            "signed_delta_mean_per_part": signed_part.tolist(),
        }

    # ==================================================================
    # ABLATION 3 — Progressive Keep
    # ==================================================================
    if not args.skip_progressive_keep:
        prog_records = ablation_progressive_keep(
            model, loader, device, cfg, total_chunks, y_true, edge_thr
        )
        df_prog = pd.DataFrame(prog_records)
        # Reorder columns so n_chunks_kept is first
        cols = ["n_chunks_kept"] + [c for c in df_prog.columns if c != "n_chunks_kept"]
        df_prog = df_prog[cols]
        df_prog.to_csv(out_dir / "progressive_metrics.csv", index=False)

        # Find elbow: first N where MAE <= 105% of full-audio MAE
        full_mae   = baseline_metrics["mae"]
        thr_mae    = full_mae * 1.05
        elbow_rows = df_prog[df_prog["mae"] <= thr_mae]
        elbow_n    = (
            int(elbow_rows["n_chunks_kept"].min())
            if not elbow_rows.empty
            else total_chunks
        )

        print(f"\n[Progressive-Keep] Elbow point "
              f"(MAE <= 105% of baseline {full_mae:.4f}): "
              f"N = {elbow_n} / {total_chunks} chunks")
        print(df_prog[["n_chunks_kept", "mae", "qwk"]].to_string(index=False))

        all_results["progressive_keep"] = {
            "elbow_n_chunks": elbow_n,
            "total_chunks":   total_chunks,
            "records":        prog_records,
        }

    # ==================================================================
    # Save consolidated JSON summary
    # ==================================================================
    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All ablation results saved to: {out_dir}")
    print(f"  chunk_zero_deltas.csv      per-sample delta per chunk")
    print(f"  chunk_zero_summary.csv     mean |delta| + rank per chunk")
    print(f"  part_zero_deltas.csv       per-sample delta per part")
    print(f"  part_zero_summary.csv      mean |delta| + rank per part")
    print(f"  progressive_metrics.csv    MAE/QWK vs. #chunks kept")
    print(f"  ablation_summary.json      merged JSON summary")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()