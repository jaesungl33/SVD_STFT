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

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# local modules
sys.path.append('src')
from stft.svd_stft import compute_stft
from stft.stft_transform import reconstruct_audio
from final_best_svd_stft import (
    final_best_embed_svd_stft as embed_fn,
)

SR16 = 16000
N_FFT = 1024
HOP = 256
WINDOW = 'hann'

inputs = [
    Path('100sample_wav/music_wav'),
    Path('100sample_wav/speech_wav'),
]

files = []
for d in inputs:
    if d.exists():
        files.extend(sorted(d.glob('*.wav')))

if not files:
    print(json.dumps({"error": "no wav files found in expected folders"}, indent=2))
    sys.exit(1)

# fixed payload
rng = np.random.default_rng(123)
payload = rng.integers(0, 2, size=16).tolist()

rows = []

for p in files:
    try:
        y, sr = sf.read(str(p))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR16:
            if librosa is None:
                raise RuntimeError('librosa required for resampling')
            y16 = librosa.resample(y, orig_sr=sr, target_sr=SR16)
        else:
            y16 = y.astype(np.float32)

        # Embed
        S = compute_stft(y16, SR16, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
        S_wm, ref = embed_fn(S, payload, alpha=0.15, block_size=(8, 8), key=42, redundancy=5, use_error_correction=True)
        y_wm = reconstruct_audio(S_wm, N_FFT, WINDOW)

        # Align
        n = min(len(y16), len(y_wm))
        y16 = y16[:n].astype(np.float32)
        y_wm = y_wm[:n].astype(np.float32)

        # PESQ
        ref_t = torch.tensor(y16, dtype=torch.float32)
        deg_t = torch.tensor(y_wm, dtype=torch.float32)
        pesq_val = float(pesq_tm(deg_t, ref_t, fs=SR16, mode='wb'))

        rows.append({
            'file': p.name,
            'pesq_watermarked_wb_16k': pesq_val
        })
    except Exception as e:
        rows.append({
            'file': p.name,
            'error': str(e)
        })

out_dir = Path('audioseal_eval')
out_dir.mkdir(exist_ok=True)

# Save CSV
import pandas as pd
pd.DataFrame(rows).to_csv(out_dir / 'pesq_torchmetrics_watermarked.csv', index=False)

# Summary
valid = [r for r in rows if 'error' not in r]
summary = {
    'num_files': len(files),
    'num_valid': len(valid),
    'pesq_watermarked_mean': float(np.mean([r['pesq_watermarked_wb_16k'] for r in valid])) if valid else None,
}

with open(out_dir / 'pesq_torchmetrics_watermarked_summary.json', 'w') as f:
    json.dump({'rows': rows, 'summary': summary}, f, indent=2)

print(json.dumps(summary, indent=2))
