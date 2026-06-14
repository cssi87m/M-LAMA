"""
Inference script for current model (supports question-aware encoder, gated fusion, whisper/non-hier, etc.)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Wav2Vec2Processor, WhisperProcessor

from config import Config
from dataloader import ESLDatasetByCandidatesWithAudio, get_collate_fn_bycandidates_with_audio
from model import ESLGradingModelByCandidatesWithAudio
from utils import clean_dataframe_bycandidates


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for current ESL model")
    default_config = Path(__file__).with_name("config.yaml")
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="Model/preds", help="Directory to save preds/metrics")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of rows for quick test")
    return parser.parse_args()


def load_checkpoint_shape_safe(model: torch.nn.Module, state_dict: dict):
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
        print(f"⚠ Skipped {len(skipped)} keys (shape/name mismatch). Showing up to 5: {skipped[:5]}")
    return None


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    tok.padding_side = "right"
    return tok


def build_loader(csv_path: str, cfg: Config, tokenizer, audio_processor, limit: int | None = None):
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)
    #df = clean_dataframe_bycandidates(df, remove_low_content=True, filter_scores=True, criteria=cfg.data.criteria)

    # Use eval_num_chunks if specified, otherwise use num_chunks
    num_chunks = cfg.audio.eval_num_chunks if cfg.audio.eval_num_chunks is not None else cfg.audio.num_chunks

    dataset = ESLDatasetByCandidatesWithAudio(
        df,
        criteria=cfg.data.criteria,
        audio_processor=audio_processor,
        encoder_type=cfg.model.audio_encoder_type,
        num_chunks=num_chunks,  # Use eval_num_chunks for evaluation
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
    # Use eval_batch_size if specified, otherwise use batch_size
    batch_size = cfg.training.eval_batch_size if cfg.training.eval_batch_size is not None else cfg.training.batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Use eval_batch_size for evaluation
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=True,
    )
    return loader


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    bins_true = np.round(y_true * 2) / 2.0
    bins_pred = np.round(y_pred * 2) / 2.0
    ints_true = np.round(bins_true * 2).astype(int)
    ints_pred = np.round(bins_pred * 2).astype(int)
    return cohen_kappa_score(ints_true, ints_pred, weights="quadratic")


def eval_split(name: str, loader: DataLoader, model, device: str, cfg: Config, edge_thr: float):
    all_preds, all_true, all_ids = [], [], []
    model.eval()
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{name} evaluating"):
            audio = batch["audio"].to(device)
            true_scores = batch["score"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                if cfg.model.use_question_encoder:
                    q_ids = batch["question_input_ids"].to(device)
                    q_mask = batch["question_attention_mask"].to(device)
                    r_ids = batch["response_input_ids"].to(device)
                    r_mask = batch["response_attention_mask"].to(device)
                    outputs = model(
                        question_input_ids=q_ids,
                        question_attention_mask=q_mask,
                        response_input_ids=r_ids,
                        response_attention_mask=r_mask,
                        audio=audio,
                    )
                else:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    outputs = model(input_ids, attention_mask, audio)

            pred_scores = outputs["expected_score"].detach().cpu().numpy()
            all_preds.append(pred_scores)
            all_true.append(true_scores.cpu().numpy())
            all_ids.extend(batch["candidate_id"])

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    # IMPORTANT: Round predictions to nearest 0.5 (VSTEP scoring: 3.0, 3.5, 4.0, ..., 9.0)
    y_pred_rounded = np.round(y_pred * 2) / 2
    y_pred_rounded = np.clip(y_pred_rounded, 0, 10)

    # Compute metrics using ROUNDED predictions
    err = y_pred_rounded - y_true

    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    qwk = float(quadratic_weighted_kappa(y_true, y_pred_rounded))
    
    edge_mask = (y_true <= edge_thr) | (y_true >= (10 - edge_thr))
    mid_mask = ~edge_mask
    edge_mae = float(np.mean(np.abs(err[edge_mask]))) if edge_mask.any() else float("nan")
    mid_mae = float(np.mean(np.abs(err[mid_mask]))) if mid_mask.any() else float("nan")
    edge_pred_ratio = float(((y_pred_rounded <= edge_thr) | (y_pred_rounded >= (10 - edge_thr))).mean())
    edge_true_ratio = float(edge_mask.mean())

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "qwk": qwk,
        "edge_mae": edge_mae,
        "mid_mae": mid_mae,
        "edge_pred_ratio": edge_pred_ratio,
        "edge_true_ratio": edge_true_ratio,
        "n_edge": int(edge_mask.sum()),
        "n_mid": int(mid_mask.sum()),
        "n": int(len(y_true)),
    }

    print(f"\n{name} metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    df_out = pd.DataFrame({
        "Candidate_ID": all_ids,
        "true_score": y_true,
        "pred_score_raw": y_pred,
        "pred_score_rounded": y_pred_rounded,
        "error": err,
        "abs_error": np.abs(err),
    })
    return df_out, metrics


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.checkpoint.load_checkpoint)
    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    tokenizer = build_tokenizer(cfg.model.model_name)
    if cfg.model.audio_encoder_type.lower() == "whisper":
        audio_processor = WhisperProcessor.from_pretrained(cfg.model.audio_encoder_id)
    else:
        audio_processor = Wav2Vec2Processor.from_pretrained(cfg.model.audio_encoder_id)

    # Build model with ALL config parameters (including LoRA configs)
    import inspect
    sig = inspect.signature(ESLGradingModelByCandidatesWithAudio.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    model_kwargs = {k: v for k, v in cfg.model.__dict__.items() if k in valid_params}
    model = ESLGradingModelByCandidatesWithAudio(**model_kwargs).to(device)

    load_checkpoint_shape_safe(model, state_dict)
    model.to(device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #train_loader = build_loader(cfg.data.train_path, cfg, tokenizer, audio_processor, limit=args.limit)
    #val_loader = build_loader(cfg.data.val_path, cfg, tokenizer, audio_processor, limit=args.limit)
    test_loader = build_loader(cfg.data.test_path, cfg, tokenizer, audio_processor, limit=args.limit)

    edge_thr = cfg.training.edge_threshold
    #train_df, train_metrokics = eval_split("TRAIN", train_loader, model, device, cfg, edge_thr)
    #val_df, val_metrics = eval_split("VAL", val_loader, model, device, cfg, edge_thr)
    test_df, test_metrics = eval_split("TEST", test_loader, model, device, cfg, edge_thr)

    #train_df.to_csv(out_dir / "train_predictions.csv", index=False)
    #val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    summary = {
        #"train": train_metrics,
        #"val": val_metrics,
        "test": test_metrics,
        "checkpoint": str(ckpt_path),
        "edge_threshold": edge_thr,
        "config": cfg.to_dict(),
    }
    with open(out_dir / "metrics_summary_val.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved preds to {out_dir}")
    print("val_predictions.csv, metrics_summary_val.json")

if __name__ == "__main__":
    main()
