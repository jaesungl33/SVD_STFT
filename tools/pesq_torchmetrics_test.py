import sys
import json
from pathlib import Path
import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None

try:
    from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as pesq_tm
    import torch
except Exception as e:
    print(json.dumps({
        "error": "torchmetrics PESQ not available",
        "hint": "pip install 'numpy<2' 'scipy<2' librosa==0.10.1 'torchmetrics[audio]==0.11.1' pesq==0.0.4",
        "exception": str(e)
    }, indent=2))
    sys.exit(1)

SR16 = 16000
SR32 = 32000

# Pick one file
cands = list(Path('100sample_wav/music_wav').glob('*.wav'))
if not cands:
    print(json.dumps({"error": "no wavs found in 100sample_wav/music_wav"}))
    sys.exit(1)

src = cands[0]

y, sr = sf.read(str(src))
if y.ndim > 1:
    y = y.mean(axis=1)
if sr != SR16:
    if librosa is None:
        print(json.dumps({"error": "librosa required for resampling"}))
        sys.exit(1)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=SR16)
else:
    y16 = y.astype(np.float32)

# make a slightly degraded version: 16k->32k->16k
if librosa is None:
    print(json.dumps({"error": "librosa required for resampling"}))
    sys.exit(1)

y32 = librosa.resample(y16, orig_sr=SR16, target_sr=SR32)
y16_back = librosa.resample(y32, orig_sr=SR32, target_sr=SR16)

n = min(len(y16), len(y16_back))
y16 = y16[:n].astype(np.float32)
y16_back = y16_back[:n].astype(np.float32)

# Torch tensors
ref_t = torch.tensor(y16, dtype=torch.float32)
deg_clean_t = torch.tensor(y16, dtype=torch.float32)
deg_resamp_t = torch.tensor(y16_back, dtype=torch.float32)

# PESQ (wb, 16 kHz)
try:
    pesq_clean = float(pesq_tm(deg_clean_t, ref_t, fs=SR16, mode='wb'))
    pesq_resamp = float(pesq_tm(deg_resamp_t, ref_t, fs=SR16, mode='wb'))
except Exception as e:
    print(json.dumps({
        "error": "PESQ backend error",
        "hint": "Ensure pesq is installed and compiled against current numpy.",
        "exception": str(e)
    }, indent=2))
    sys.exit(1)

print(json.dumps({
    "file": src.name,
    "pesq_clean_wb_16k": pesq_clean,
    "pesq_resample_16k_32k_16k": pesq_resamp
}, indent=2))
