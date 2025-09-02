import os, glob, math, numpy as np
import torch, torchaudio
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Import SVD_STFT functions
from stft.svd_stft import compute_stft
from stft.stft_transform import reconstruct_audio
from final_best_svd_stft import (
    final_best_embed_svd_stft as embed_fn,
    final_best_extract_svd_stft as extract_fn,
)

SR = 16_000
INTERMEDIATE = 32_000  # AudioSeal's resample rate
N_FFT = 1024
HOP = 256
WINDOW = 'hann'

# Fixed payload for consistency
rng = np.random.default_rng(123)
PAYLOAD = rng.integers(0, 2, size=16).tolist()

def embed_svd_stft(wave, sr, payload=None):
    """Embed watermark using SVD_STFT"""
    if payload is None:
        payload = PAYLOAD
    
    # Convert torch tensor to numpy
    if isinstance(wave, torch.Tensor):
        wave_np = wave.numpy().flatten()
    else:
        wave_np = wave.flatten()
    
    # Compute STFT
    S = compute_stft(wave_np, sr, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    
    # Embed watermark
    S_wm, ref = embed_fn(S, payload, alpha=0.15, block_size=(8, 8), key=42, redundancy=5, use_error_correction=True)
    
    # Reconstruct audio
    wave_wm = reconstruct_audio(S_wm, N_FFT, WINDOW)
    
    # Convert back to torch tensor
    wave_wm_tensor = torch.tensor(wave_wm, dtype=torch.float32)
    
    # Ensure same shape as input
    if isinstance(wave, torch.Tensor):
        if wave.ndim == 2:  # (channels, samples)
            wave_wm_tensor = wave_wm_tensor.unsqueeze(0)
        if wave_wm_tensor.shape[-1] < wave.shape[-1]:
            wave_wm_tensor = torch.nn.functional.pad(wave_wm_tensor, (0, wave.shape[-1] - wave_wm_tensor.shape[-1]))
        else:
            wave_wm_tensor = wave_wm_tensor[..., :wave.shape[-1]]
    
    return wave_wm_tensor

def detect_svd_stft(wave, sr):
    """Detect watermark and return (bits, ber)"""
    # Convert torch tensor to numpy
    if isinstance(wave, torch.Tensor):
        wave_np = wave.numpy().flatten()
    else:
        wave_np = wave.flatten()
    
    # Compute STFT
    S = compute_stft(wave_np, sr, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    
    # Reference data (we need this for extraction)
    # For detection, we'll use a simple approach: try to extract and measure BER
    try:
        # Create minimal reference data
        ref_data = {
            'alpha': 0.15,
            'block_size': (8, 8),
            'key': 42,
            'redundancy': 5,
            'use_error_correction': True,
            'payload_length': len(PAYLOAD)
        }
        
        extracted_bits = extract_fn(S, ref_data, alpha=0.15, block_size=(8, 8), key=42, num_bits=len(PAYLOAD), threshold_method='hybrid')
        
        if len(extracted_bits) == len(PAYLOAD):
            # Calculate BER
            ber = sum(1 for a, b in zip(extracted_bits, PAYLOAD) if a != b) / len(PAYLOAD)
            return extracted_bits, ber
        else:
            # If extraction failed, return high BER
            return [], 1.0
            
    except Exception as e:
        # If extraction fails, return high BER
        return [], 1.0

def resample_roundtrip(wave, sr=SR, mid=INTERMEDIATE):
    # 16k -> 32k -> 16k with high-quality sinc (torchaudio uses windowed-sinc)
    to_mid = torchaudio.transforms.Resample(orig_freq=sr, new_freq=mid)
    to_orig = torchaudio.transforms.Resample(orig_freq=mid, new_freq=sr)
    with torch.no_grad():
        w_mid = to_mid(wave)
        w_back = to_orig(w_mid)
    # Trim/pad to exactly original length (shape preserved per paper)
    L = wave.shape[-1]
    if w_back.shape[-1] < L:
        w_back = torch.nn.functional.pad(w_back, (0, L - w_back.shape[-1]))
    else:
        w_back = w_back[..., :L]
    return w_back

def detect_score(wave, sr):
    out = detect_svd_stft(wave, sr)
    if isinstance(out, tuple):  # (bits, ber) style
        bits, ber = out
        return 1.0 - float(ber)  # higher is better
    return float(out)

# --- Load a small dataset of wavs at SR ---
def load_wavs(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.wav")))
    waves = []
    for p in paths:
        w, sr = torchaudio.load(p)
        if sr != SR:
            w = torchaudio.functional.resample(w, sr, SR)
        # mono
        if w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
        waves.append(w)
    return waves

# --- Build balanced attacked sets (resample only) ---
def build_attacked_sets(waves):
    wm_attacked = []
    clean_attacked = []
    for w in waves:
        w = w.clone()
        w_wm = embed_svd_stft(w, SR, payload=None)
        wm_attacked.append(resample_roundtrip(w_wm, SR, INTERMEDIATE))
        clean_attacked.append(resample_roundtrip(w, SR, INTERMEDIATE))
    return wm_attacked, clean_attacked

# --- Evaluate like AudioSeal: best-threshold accuracy and ROC AUC ---
def evaluate_resample_only(wm_attacked, clean_attacked):
    scores = [detect_score(w, SR) for w in wm_attacked] + [detect_score(w, SR) for w in clean_attacked]
    labels = [1]*len(wm_attacked) + [0]*len(clean_attacked)

    # threshold sweep
    cand = sorted(set(scores))
    if len(cand) > 400:
        # downsample thresholds for speed
        idxs = np.linspace(0, len(cand)-1, 400).astype(int)
        cand = [cand[i] for i in idxs]

    best = {"acc": -1, "thr": None, "tpr": None, "fpr": None}
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    for thr in cand:
        preds = (scores_arr >= thr).astype(int)
        tp = ((preds == 1) & (labels_arr == 1)).sum()
        tn = ((preds == 0) & (labels_arr == 0)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        if acc > best["acc"]:
            best = {"acc": float(acc), "thr": float(thr), "tpr": float(tpr), "fpr": float(fpr)}

    # ROC AUC
    try:
        auc = float(roc_auc_score(labels_arr, scores_arr))
    except Exception:
        auc = float("nan")

    return best, auc

if __name__ == "__main__":
    # Use the 100sample_wav folders
    music_folder = "100sample_wav/music_wav"
    speech_folder = "100sample_wav/speech_wav"
    
    waves = []
    if os.path.exists(music_folder):
        waves.extend(load_wavs(music_folder))
    if os.path.exists(speech_folder):
        waves.extend(load_wavs(speech_folder))
    
    if not waves:
        print("No wav files found in expected folders")
        sys.exit(1)
    
    print(f"Loaded {len(waves)} audio files")
    
    wm_attacked, clean_attacked = build_attacked_sets(waves)
    best, auc = evaluate_resample_only(wm_attacked, clean_attacked)
    print(f"[Resample 32k round-trip] Acc={best['acc']:.3f} (TPR={best['tpr']:.3f}/FPR={best['fpr']:.3f}) at threshold={best['thr']:.4f}, AUC={auc:.3f}")
    
    # Save detailed results
    import json
    results = {
        "attack": "resample_32k_roundtrip",
        "num_files": len(waves),
        "best_accuracy": best["acc"],
        "best_threshold": best["thr"],
        "true_positive_rate": best["tpr"],
        "false_positive_rate": best["fpr"],
        "roc_auc": auc
    }
    
    os.makedirs("audioseal_eval", exist_ok=True)
    with open("audioseal_eval/resample_attack_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to audioseal_eval/resample_attack_results.json")
