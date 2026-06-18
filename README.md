# M-LAMA

M-LAMA is the experiment code for **"M-LAMA: Multimodal Automated Scoring of Long-form Spoken English"**. The model scores long-form spoken English sessions with text, audio, and question context. It supports the five scoring criteria used in the paper: `grammar`, `fluency`, `pronunciation`, `vocabulary`, and `content`.

The private paper dataset is not included in this repository. To reproduce the reported experiments, prepare the same CSV splits and audio files, then point the YAML configs to those local paths.

## What Is In This Repo

```text
.
├── Model/                         # Main M-LAMA single-stage/final-stage experiments
│   ├── config/                    # Criterion-specific YAML configs
│   ├── ablation_Q_aware/          # Question-aware encoder ablation
│   ├── ablation_longaudio_weight/ # Temporal audio chunk importance ablation
│   └── ablation_weight_text_audio/# Text/audio contribution ablation
├── Model_finetune_3_stages/       # Three-stage training for one criterion
├── Exp2_noaudio/                  # Text-only experiment: contribution of audio module
├── Exp2_notext/                   # Audio-only experiment: contribution of text module
├── mlama/                         # Shared package utilities and CLI dispatcher
├── scripts/examples/              # Portable example shell commands
├── pyproject.toml                 # Editable package metadata
└── requirements.txt               # Dependency list
```

## Model Summary

M-LAMA combines:

- A text encoder for ASR transcript and question context.
- An audio encoder for long-form speech chunks.
- Part-aware audio modeling across speaking parts.
- Question-aware text encoding for task fulfillment.
- Cross-modal audio/text interaction and gated fusion.
- Score prediction over 21 bins from 0.0 to 10.0 in 0.5 increments.

The main implementation lives in `Model/model.py`, with training in `Model/train.py`, evaluation in `Model/test.py`, data loading in `Model/dataloader.py`, and losses in `Model/losses.py`.

## Setup

Create an environment with Python 3.10+ and install the package in editable mode:

```bash
cd /path/to/M-LAMA
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you prefer requirements-only installation:

```bash
pip install -r requirements.txt
```

Download NLTK stopwords if your environment does not already have them:

```bash
python3 -m nltk.downloader stopwords
```

Optional W&B logging uses your shell environment:

```bash
export WANDB_API_KEY="..."
```

Do not commit API keys or private dataset paths.

## Data Format

Each split CSV should contain one row per candidate/session. The main pipeline expects these columns:

| Column | Description |
| --- | --- |
| `Candidate_ID` | Unique candidate/session id |
| `text` | Python-literal list of transcripts for the speaking parts |
| `question_type` | Python-literal list of part/question type ids |
| `Question` | Optional Python-literal list of question strings |
| `absolute_paths` | Python-literal list of audio file paths |
| `final`, `grammar`, `fluency`, `pronunciation`, `vocabulary`, `content` | Scores, depending on the criterion |

The default configs contain historical absolute paths from the original experiment machine. Before running, edit:

```yaml
data:
  train_path: /path/to/train.csv
  val_path: /path/to/val.csv
  test_path: /path/to/test.csv
checkpoint:
  load_checkpoint: /path/to/optional/init_or_eval_checkpoint.pth
  save_dir: runs/checkpoints/<experiment>
logging:
  log_dir: runs/logs/<experiment>
```

## Main Training

Train one criterion:

```bash
python3 -m Model.train --config Model/config/config_fluency.yaml
```

Equivalent package CLI:

```bash
python3 -m mlama.cli train --config Model/config/config_fluency.yaml
```

Initialize from a checkpoint:

```bash
python3 -m Model.train \
  --config Model/config/config_fluency.yaml \
  --checkpoint /path/to/model_best.pth
```

Train all five criterion-specific models:

```bash
bash scripts/examples/01_main_train_5_criteria.sh
```

The training script saves the final model to `checkpoint.save_dir` and keeps the best validation checkpoint according to the trainer logic.

## Evaluation

Evaluate a checkpoint on validation and test splits:

```bash
python3 -m Model.test \
  --config Model/config/config_fluency.yaml \
  --checkpoint /path/to/model_best.pth \
  --output_dir runs/preds/fluency
```

If `--checkpoint` is omitted, evaluation uses `checkpoint.load_checkpoint` from the config, or the latest `model_best_mae_*.pth` in `checkpoint.save_dir`.

Evaluate all five criteria:

```bash
bash scripts/examples/02_main_evaluate_5_criteria.sh
```

Outputs include:

- `val_predictions.csv`
- `test_predictions.csv`
- `metrics_summary.json`

## Example Shell Scripts

All old machine-specific shell scripts were removed. The only shell examples now live in `scripts/examples/`.

| Script | Experiment |
| --- | --- |
| `01_main_train_5_criteria.sh` | Main M-LAMA training for all criteria |
| `02_main_evaluate_5_criteria.sh` | Main M-LAMA evaluation for all criteria |
| `03_ablation_question_aware.sh` | Question-aware module ablation |
| `04_ablation_text_audio_fusion.sh` | Text/audio fusion contribution ablation |
| `05_ablation_audio_chunks.sh` | Long-audio temporal chunk ablation |
| `06_exp2_text_only_noaudio.sh` | Exp2 text-only/no-audio experiment |
| `07_exp2_audio_only_notext.sh` | Exp2 audio-only/no-text experiment |
| `08_three_stage_preprocess.sh` | Three-stage preprocessing example |
| `09_three_stage_train.sh` | Three-stage training example |

Examples are repo-relative and configurable through environment variables. For example:

```bash
CRITERIA="fluency grammar" bash scripts/examples/01_main_train_5_criteria.sh
CHECKPOINT=/path/to/model_best.pth bash scripts/examples/03_ablation_question_aware.sh
MODE=train bash scripts/examples/06_exp2_text_only_noaudio.sh
```

Metrics include MAE, MSE, RMSE, QWK, edge-score MAE, mid-score MAE, and score-range counts.

## Ablations

### Question-Aware Encoder

Tests the effect of bypassing the question-aware module at inference time:

```bash
python3 -m Model.ablation_Q_aware.ablation_qaware \
  --config Model/config/config_fluency.yaml \
  --ckpt /path/to/model_best.pth \
  --split test
```

### Text/Audio Fusion Contribution

Measures learned fusion weights and modality scaling effects:

```bash
python3 -m Model.ablation_weight_text_audio.ablation_fusion \
  --config Model/config/config_fluency.yaml \
  --ckpt /path/to/model_best.pth \
  --split test
```

Print static fusion weights:

```bash
python3 -m Model.ablation_weight_text_audio.print_weights \
  --config Model/config/config_fluency.yaml \
  --ckpt /path/to/model_best.pth
```

### Long-Audio Chunk Importance

Measures which temporal chunks and speaking parts most affect prediction:

```bash
python3 -m Model.ablation_longaudio_weight.ablation_chunk_importance \
  --config Model/config/config_fluency.yaml \
  --checkpoint /path/to/model_best.pth \
  --output_dir runs/ablations/chunks/fluency \
  --split test
```

## Text-Only And Audio-Only Experiments

These reproduce the Exp2 modality-contribution studies:

```bash
cd Exp2_noaudio
python3 train.py --config config/config_fluency.yaml
python3 test.py --config config/config_fluency.yaml
```

```bash
cd Exp2_notext
python3 train.py --config config/config_fluency.yaml
python3 test.py --config config/config_fluency.yaml
```

Use the matching configs for the other criteria.

## Three-Stage Fine-Tuning

`Model_finetune_3_stages/` implements the progressive training strategy for one criterion:

1. Stage 1: contrastive text/audio alignment.
2. Stage 2: ordinal/range-aware classification.
3. Stage 3: fine-grained regression with combined loss.

Configure paths and hyperparameters in:

```text
Model_finetune_3_stages/config.py
```

Then run:

```bash
bash scripts/examples/08_three_stage_preprocess.sh
bash scripts/examples/09_three_stage_train.sh
```

See `Model_finetune_3_stages/README.md` for stage-specific notes.

## Reproducibility Notes

- The main package seeds Python, NumPy, PyTorch, and CUDA through `mlama.reproducibility.set_seed`.
- Default seed is `42`.
- Tokenizer parallelism is disabled by default for stable logs.
- Configs define model, audio chunking, data paths, losses, checkpoint paths, and logging.
- Predictions are rounded to the nearest 0.5 point before reported evaluation metrics.
- The private dataset and trained checkpoints are required to reproduce the exact paper numbers.

## Development Checks

Run syntax checks without loading large models:

```bash
python3 -m py_compile \
  mlama/*.py \
  Model/train.py Model/test.py Model/trainer.py Model/dataloader.py Model/utils.py Model/text_processing.py
```

Check the package CLI:

```bash
python3 -m mlama.cli --help
```

## Contact

For inquiries regarding data access or technical details related to this paper, please contact:

- minh.dxq225449@sis.hust.edu.vn
- son.dn225997@sis.hust.edu.vn
- anhbtm@soict.edu.vn
- lenp@soict.edu.vn

## Citation

If you use this code, cite the associated paper:

```bibtex
@inproceedings{mlama2026,
  title     = {M-LAMA: Multimodal Automated Scoring of Long-form Spoken English},
  author    = {Dao-Xuan-Quang, Minh and Dinh-Nguyen, Son and Bui, Thi-Mai-Anh and Nguyen, Phi-Le},
  booktitle = {Interspeech},
  year      = {2026}
}
```
