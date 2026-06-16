"""Command-line dispatcher for the M-LAMA experiment package."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M-LAMA experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the full text+audio M-LAMA model")
    train.add_argument("--config", required=True, help="YAML config path")
    train.add_argument("--checkpoint", default=None, help="Optional checkpoint to initialize from")
    train.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on val/test splits")
    evaluate.add_argument("--config", required=True, help="YAML config path")
    evaluate.add_argument("--checkpoint", default=None, help="Checkpoint path")
    evaluate.add_argument("--output_dir", default="runs/preds", help="Prediction output directory")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        from Model.train import main as train_main

        train_main(
            argv=[
                "--config",
                args.config,
                *(["--checkpoint", args.checkpoint] if args.checkpoint else []),
                *(["--no_wandb"] if args.no_wandb else []),
            ]
        )
    elif args.command == "evaluate":
        from Model.test import main as eval_main

        eval_main(
            argv=[
                "--config",
                args.config,
                *(["--checkpoint", args.checkpoint] if args.checkpoint else []),
                "--output_dir",
                args.output_dir,
            ]
        )


if __name__ == "__main__":
    main()
