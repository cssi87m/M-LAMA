#!/usr/bin/env bash

#SBATCH --job-name=interspeech
#SBATCH --output=interspeech.log
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00             # đặt tối đa được phép ở cụm bạn
#SBATCH --requeue
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --nodelist=dgx02
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
set -euo pipefail

echo "[$(date)] Node: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# --- (tuỳ cụm) nạp module CUDA để có nvcc ---
if command -v module &>/dev/null; then
  module load cuda || true
fi

# Bẫy tín hiệu để thoát gọn
cleanup() {
  echo "[$(date)] Caught signal, exiting."
  exit 0
}
trap cleanup SIGINT SIGTERM

# Tạo chương trình CUDA C cực nhẹ để giữ context (ít RAM hơn PyTorch)
cat > infer.cu <<'CU'
#include <cuda_runtime.h>
#include <unistd.h>

__global__ void noop() {}

int main() {
  // Khởi tạo CUDA context mà không cần cấp nhiều bộ nhớ
  cudaFree(0);
  // Gọi một kernel trống để đảm bảo context active
  noop<<<1,1>>>();
  cudaDeviceSynchronize();

  // Giữ tiến trình sống mãi tới khi bị kill/scancel
  while (true) {
    sleep(3600); // ngủ 1 giờ rồi lặp
  }
  return 0;
}
CU

# Biên dịch; nếu không có nvcc thì fallback sang PyTorch (nếu có)
if command -v nvcc &>/dev/null && nvcc --version &>/dev/null; then
  echo "[$(date)] Compiling infer.cu with nvcc..."
  nvcc -O2 infer.cu -o infer
  echo "[$(date)] Running infer..."
  ./infer
else
  echo "[$(date)] nvcc not found; falling back to PyTorch keepalive..."
  python3 - <<'PY'
import time, sys
try:
    import torch
except Exception as e:
    print("PyTorch not available; cannot hold GPU without CUDA context.", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA not available; cannot hold GPU.", file=sys.stderr)
    sys.exit(1)

torch.cuda.init()                 # tạo CUDA context
_ = torch.empty(1, device='cuda') # cấp phát siêu nhỏ để giữ allocator
torch.cuda.synchronize()
print("GPU context established via PyTorch. Holding until killed...")
while True:
    time.sleep(3600)
PY
fi
