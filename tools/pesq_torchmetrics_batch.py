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

        if librosa is None:
            raise RuntimeError('librosa required for resampling')
        y32 = librosa.resample(y16, orig_sr=SR16, target_sr=SR32)
        y16_back = librosa.resample(y32, orig_sr=SR32, target_sr=SR16)

        n = min(len(y16), len(y16_back))
        y16 = y16[:n].astype(np.float32)
        y16_back = y16_back[:n].astype(np.float32)

        ref_t = torch.tensor(y16, dtype=torch.float32)
        deg_clean_t = torch.tensor(y16, dtype=torch.float32)
        deg_resamp_t = torch.tensor(y16_back, dtype=torch.float32)

        pesq_clean = float(pesq_tm(deg_clean_t, ref_t, fs=SR16, mode='wb'))
        pesq_resamp = float(pesq_tm(deg_resamp_t, ref_t, fs=SR16, mode='wb'))

        rows.append({
            'file': p.name,
            'pesq_clean_wb_16k': pesq_clean,
            'pesq_resample_16k_32k_16k': pesq_resamp
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
pd.DataFrame(rows).to_csv(out_dir / 'pesq_torchmetrics_16k_and_resample.csv', index=False)

# Summary
valid = [r for r in rows if 'error' not in r]
summary = {
    'num_files': len(files),
    'num_valid': len(valid),
    'pesq_clean_mean': float(np.mean([r['pesq_clean_wb_16k'] for r in valid])) if valid else None,
    'pesq_resample_mean': float(np.mean([r['pesq_resample_16k_32k_16k'] for r in valid])) if valid else None,
    'pesq_delta_mean': float(np.mean([r['pesq_clean_wb_16k'] - r['pesq_resample_16k_32k_16k'] for r in valid])) if valid else None,
}

with open(out_dir / 'pesq_torchmetrics_summary.json', 'w') as f:
    json.dump({'rows': rows, 'summary': summary}, f, indent=2)

print(json.dumps(summary, indent=2))
