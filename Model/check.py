import torch
from pprint import pprint
from pathlib import Path

ckpt_path = Path('/home/user06/Interspeech_2026/Model/Model/checkpoints_unseen_new/model_best_mae_V2_final_score_512dfuse_focal_ranking_nonhir_pretrain_from_olddata_unseen_new_whisper.pth')
ckpt = torch.load(ckpt_path, map_location='cuda')
print('Keys in checkpoint:', list(ckpt.keys()))

cfg = ckpt.get('config')
if cfg:
    print('Config from checkpoint:')
    pprint(cfg)

state_dict = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
print('Sample keys from model_state_dict:')
print(list(state_dict.keys())[:20])

print('Sample audio_encoder keys:')
print([k for k in state_dict.keys() if k.startswith('audio_encoder.')][:20])
print('Sample text encoder keys:')
print([k for k in state_dict.keys() if k.startswith('encoder.')][:20])
