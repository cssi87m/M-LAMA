"""
Inference script for ESL Speaking Grading Model

Loads best checkpoint, runs inference on val and test splits,
and saves per-sample predictions plus aggregate metrics (MAE, MSE, RMSE, QWK,
and per-range stats for edge/mid).
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlama.modeling import find_checkpoint, model_kwargs_from_config
from mlama.reproducibility import configure_tokenizers, set_seed

try:
    from .audio_encoders import AudioEncoderFactory
    from .config import Config
    from .dataloader import ESLDatasetByCandidatesWithAudio, get_collate_fn_bycandidates_with_audio
    from .model import ESLGradingModelByCandidatesWithAudio
    from .utils import get_class_counts_from_dataframe, get_effective_number_weights
except ImportError:
    from audio_encoders import AudioEncoderFactory
    from config import Config
    from dataloader import ESLDatasetByCandidatesWithAudio, get_collate_fn_bycandidates_with_audio
    from model import ESLGradingModelByCandidatesWithAudio
    from utils import get_class_counts_from_dataframe, get_effective_number_weights


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Inference on val/test splits")
    default_config = Path(__file__).parent / "config" / "config.yaml"
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to config YAML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (.pth). If None, will use config.checkpoint.load_checkpoint or latest best in save_dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Model/preds",
        help="Directory to save prediction CSVs",
    )
    return parser.parse_args(argv)


def load_model_and_processors(cfg: Config, ckpt_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "right"

    audio_processor = AudioEncoderFactory.get_processor(
        encoder_type=cfg.model.audio_encoder_type,
        model_id=cfg.model.audio_encoder_id
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    model = ESLGradingModelByCandidatesWithAudio(
        **model_kwargs_from_config(ESLGradingModelByCandidatesWithAudio, cfg.model)
    )

    print(f"\nLoading checkpoint with strict=False...")
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

    if missing_keys:
        print(f"  Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
    if unexpected_keys:
        print(f"  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")

    # Check if QuestionAwareEncoder keys are missing (expected if loading old checkpoint)
    qa_missing = [k for k in missing_keys if 'question_aware_encoder' in k]
    if qa_missing and cfg.model.use_question_encoder:
        print(f"  ⚠️  QuestionAwareEncoder keys missing (expected for old checkpoints): {len(qa_missing)}")

    model.to(device)
    model.eval()
    return model, tokenizer, audio_processor


def build_dataloader(csv_path: str, cfg: Config, tokenizer, audio_processor):
    df = pd.read_csv(csv_path)
    # df = clean_dataframe_bycandidates(
    #     df,
    #     remove_low_content=False,
    #     filter_scores=True,
    #     criteria=cfg.data.criteria,
    # )
    # Use eval_num_chunks if specified, otherwise use num_chunks
    num_chunks = cfg.audio.eval_num_chunks if cfg.audio.eval_num_chunks is not None else cfg.audio.num_chunks

    # STEP 3: Pass separate_question_response flag
    dataset = ESLDatasetByCandidatesWithAudio(
        df,
        criteria=cfg.data.criteria,
        audio_processor=audio_processor,
        encoder_type=cfg.model.audio_encoder_type,
        num_chunks=num_chunks,  # Use eval_num_chunks for evaluation
        chunk_length_sec=cfg.audio.chunk_length_sec,
        separate_question_response=cfg.model.use_question_encoder,
    )
    # STEP 3: Pass separate_tokenize flag
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
    return loader, df


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    QWK expects discrete labels; round to nearest 0.5 bin (0-10) then map to ints.
    """
    bins = np.round(y_pred * 2) / 2.0
    bins_true = np.round(y_true * 2) / 2.0
    ints = np.round(bins * 2).astype(int)
    ints_true = np.round(bins_true * 2).astype(int)
    return cohen_kappa_score(ints_true, ints, weights="quadratic")


def eval_split(
    name: str,
    loader: DataLoader,
    model: ESLGradingModelByCandidatesWithAudio,
    device: str,
    cfg: Config,
    class_weights: torch.Tensor,
    edge_threshold: float,
):
    all_preds = []
    all_true = []
    all_ids = []
    with torch.no_grad():
        for batch in loader:
            # Move audio and scores to device
            audio = batch["audio"].to(device)
            true_scores = batch["score"].to(device)

            # STEP 3: Conditional input based on use_question_encoder
            if cfg.model.use_question_encoder:
                question_input_ids = batch["question_input_ids"].to(device)
                question_attention_mask = batch["question_attention_mask"].to(device)
                response_input_ids = batch["response_input_ids"].to(device)
                response_attention_mask = batch["response_attention_mask"].to(device)

                outputs = model(
                    question_input_ids=question_input_ids,
                    question_attention_mask=question_attention_mask,
                    response_input_ids=response_input_ids,
                    response_attention_mask=response_attention_mask,
                    audio=audio
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
    mae = np.mean(np.abs(y_pred_rounded - y_true))
    mse = np.mean((y_pred_rounded - y_true) ** 2)
    rmse = math.sqrt(mse)
    qwk = quadratic_weighted_kappa(y_true, y_pred_rounded)

    # Per-range stats - using ROUNDED predictions
    edge_mask = (y_true <= edge_threshold) | (y_true >= (10 - edge_threshold))
    mid_mask = ~edge_mask

    def masked_mae(mask):
        if mask.sum() == 0:
            return float("nan")
        return np.mean(np.abs(y_pred_rounded[mask] - y_true[mask]))

    edge_mae = masked_mae(edge_mask)
    mid_mae = masked_mae(mid_mask)

    edge_pred_ratio = float(((y_pred_rounded <= edge_threshold) | (y_pred_rounded >= (10 - edge_threshold))).mean())
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
        "n": len(y_true),
    }

    df_out = pd.DataFrame(
        {
            "Candidate_ID": all_ids,
            "true_score": y_true,
            "pred_score_raw": y_pred,
            "pred_score_rounded": y_pred_rounded,
            "error": y_pred_rounded - y_true,
            "abs_error": np.abs(y_pred_rounded - y_true),
        }
    )

    print(f"\n{name} results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return df_out, metrics


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = Config.from_yaml(args.config)
    runtime = set_seed()
    configure_tokenizers(parallelism=False)
    device = runtime.device
    ckpt_path = find_checkpoint(args.checkpoint, cfg.checkpoint.load_checkpoint, cfg.checkpoint.save_dir)
    print(f"Using checkpoint: {ckpt_path}")

    model, tokenizer, audio_processor = load_model_and_processors(cfg, ckpt_path, device)

    # Compute class weights for weighted metrics (if needed)
    train_df = pd.read_csv(cfg.data.train_path)
    class_bins = [i * 0.5 for i in range(21)]
    class_counts = get_class_counts_from_dataframe(train_df, class_bins, criteria=cfg.data.criteria)
    loss_weights = get_effective_number_weights(class_counts, beta=cfg.data.class_weight_beta).to(device)

    # Dataloaders
    val_loader, _ = build_dataloader(cfg.data.val_path, cfg, tokenizer, audio_processor)
    test_loader, _ = build_dataloader(cfg.data.test_path, cfg, tokenizer, audio_processor)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Edge threshold from config
    edge_thr = cfg.training.edge_threshold

    val_df, val_metrics = eval_split(
        "VAL", val_loader, model, device, cfg, loss_weights, edge_thr
    )
    test_df, test_metrics = eval_split(
        "TEST", test_loader, model, device, cfg, loss_weights, edge_thr
    )

    val_path = out_dir / "val_predictions.csv"
    test_path = out_dir / "test_predictions.csv"
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved predictions:")
    print(f"  VAL -> {val_path}")
    print(f"  TEST -> {test_path}")

    # Save metrics summary
    summary = {
        "val": val_metrics,
        "test": test_metrics,
        "checkpoint": ckpt_path,
        "edge_threshold": edge_thr,
    }
    summary_path = out_dir / "metrics_summary.json"
    import json

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary -> {summary_path}")


if __name__ == "__main__":
    main()
