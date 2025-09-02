import sys
from pathlib import Path
import json
import numpy as np
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None

SR16 = 16000
SR32 = 32000

out_dir = Path('pesq_quick_outputs')
out_dir.mkdir(exist_ok=True)

# Find one file
candidates = list(Path('100sample_wav/music_wav').glob('*.wav'))
if not candidates:
    print(json.dumps({"error": "no wavs in 100sample_wav/music_wav"}))
    sys.exit(1)

src = candidates[0]

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

# 16k -> 32k -> 16k
if librosa is None:
    print(json.dumps({"error": "librosa required for resampling"}))
    sys.exit(1)

y32 = librosa.resample(y16, orig_sr=SR16, target_sr=SR32)
y16_back = librosa.resample(y32, orig_sr=SR32, target_sr=SR16)

# align lengths
n = min(len(y16), len(y16_back))
y16 = y16[:n].astype(np.float32)
y16_back = y16_back[:n].astype(np.float32)

# Save
ref_path = out_dir / 'ref_16k.wav'
deg_path = out_dir / 'deg_16k_resamp32k_back.wav'
sf.write(str(ref_path), y16, SR16)
sf.write(str(deg_path), y16_back, SR16)

# SI-SNR helper
def si_snr(ref, est, eps=1e-8):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    t = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    n = est - t
    return float(10 * np.log10((np.dot(t, t) + eps) / (np.dot(n, n) + eps)))

sisnr = si_snr(y16, y16_back)

print(json.dumps({
    "file": src.name,
    "ref": str(ref_path),
    "deg": str(deg_path),
    "si_snr_db": sisnr
}, indent=2))
