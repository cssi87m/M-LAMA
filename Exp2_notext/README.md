# Exp2_notext: Audio-Focused ESL Grading Model

## Overview
This experiment tests the **audio encoder's capability** by using a **fixed text input** instead of actual transcripts. The text encoder processes only a placeholder text: `"This is work of candidate: "`, forcing the model to rely primarily on audio features for scoring.

## Key Differences from Main Model
- **Text Input**: Fixed placeholder text for all samples (no actual transcript)
- **Focus**: Audio encoder learning and audio feature extraction
- **Architecture**: Same as main model (text + audio fusion)
- **Purpose**: Measure how much information can be extracted from audio alone

## Setup

### Files
- `train.py`: Training script with proper tokenizer setup
- `trainer.py`: Training loop (copied from Model/)
- `config.py`: Configuration management (copied from Model/)
- `dataloader.py`: Modified to use fixed text placeholder
- `model.py`: Same model architecture as main experiment
- `losses.py`: Combined loss functions
- `utils.py`: Utility functions
- `audio_encoders.py`: Audio encoder factory

### Configuration
Edit `config/config.yaml` to set:
- Model architecture parameters
- Training hyperparameters
- Data paths
- Audio processing settings (num_chunks, chunk_length_sec)

## Usage

### Training
```bash
# Basic training
./train.sh

# With specific criteria
python train.py --config config/config_grammar.yaml

# Resume from checkpoint
python train.py --checkpoint checkpoints/model_best_mae_exp2_notext.pth

# Disable wandb logging
./train.sh --no_wandb
```

### Evaluation
```bash
# Test best model
python test.py --config config/config.yaml

# Generate predictions
python test.py --save_predictions --output result/
```

## Expected Behavior
- **Text encoder** processes the same fixed text for all samples
- **Audio encoder** must learn to extract all relevant information from audio
- **Fusion layer** combines fixed text embeddings with audio features
- **Performance** expected to be lower than full model but should still achieve reasonable results

## Results
Results are saved to:
- `checkpoints/`: Best model checkpoints
- `result/`: Predictions and metrics
- Wandb logs (if enabled)

## Comparison
Compare with:
- `Exp2_noaudio/`: Uses full text, no audio (text-only baseline)
- `Model/`: Uses both text transcripts and audio (full model)
