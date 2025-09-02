import sys
import json
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import roc_auc_score

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Import base functions
from stft.svd_stft import compute_stft
from stft.stft_transform import reconstruct_audio

SR = 16_000
INTERMEDIATE = 32_000
N_FFT = 1024
HOP = 256
WINDOW = 'hann'

# Fixed payload
rng = np.random.default_rng(123)
PAYLOAD = rng.integers(0, 2, size=16).tolist()

def improved_embed_svd_stft(wave, sr, payload=None):
    """Improved SVD_STFT embedding with better resampling robustness"""
    if payload is None:
        payload = PAYLOAD
    
    # Convert torch tensor to numpy
    if isinstance(wave, torch.Tensor):
        wave_np = wave.numpy().flatten()
    else:
        wave_np = wave.flatten()
    
    # Compute STFT
    S = compute_stft(wave_np, sr, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    
    # Enhanced embedding with frequency-domain spreading
    S_wm, ref = enhanced_embed_with_spreading(S, payload)
    
    # Reconstruct audio
    wave_wm = reconstruct_audio(S_wm, N_FFT, WINDOW)
    
    # Convert back to torch tensor
    wave_wm_tensor = torch.tensor(wave_wm, dtype=torch.float32)
    
    # Ensure same shape as input
    if isinstance(wave, torch.Tensor):
        if wave.ndim == 2:
            wave_wm_tensor = wave_wm_tensor.unsqueeze(0)
        if wave_wm_tensor.shape[-1] < wave.shape[-1]:
            wave_wm_tensor = torch.nn.functional.pad(wave_wm_tensor, (0, wave.shape[-1] - wave_wm_tensor.shape[-1]))
        else:
            wave_wm_tensor = wave_wm_tensor[..., :wave.shape[-1]]
    
    return wave_wm_tensor

def enhanced_embed_with_spreading(S_complex, bits):
    """Enhanced embedding with frequency-domain spreading and synchronization"""
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Add synchronization markers
    sync_pattern = [1, 0, 1, 0, 1, 0]  # 6-bit sync pattern
    full_payload = sync_pattern + bits
    
    # Frequency-domain spreading
    rows, cols = S_mag.shape
    block_size = (8, 8)
    
    # Use mid-frequency bands for better robustness
    f_min = int(rows * 0.2)  # 20% of frequency range
    f_max = int(rows * 0.8)  # 80% of frequency range
    
    # Create frequency-domain spreading sequence
    spreading_seq = np.random.default_rng(42).normal(0, 1, len(full_payload))
    
    # Adaptive embedding strength based on frequency
    alpha_base = 0.1
    alpha_adaptive = np.zeros(len(full_payload))
    
    for i in range(len(full_payload)):
        # Stronger embedding in mid-frequencies
        freq_ratio = (f_min + i * (f_max - f_min) // len(full_payload)) / rows
        alpha_adaptive[i] = alpha_base * (1 + 0.5 * np.sin(2 * np.pi * freq_ratio))
    
    # Embed with spreading
    S_mag_mod = S_mag.copy()
    
    for i, (bit, spread_val, alpha) in enumerate(zip(full_payload, spreading_seq, alpha_adaptive)):
        # Select frequency band
        f_center = f_min + i * (f_max - f_min) // len(full_payload)
        f_start = max(0, f_center - 4)
        f_end = min(rows, f_center + 4)
        
        # Embed in frequency band
        for f in range(f_start, f_end):
            for t in range(0, cols, 8):
                if t + 8 <= cols:
                    block = S_mag_mod[f:f+8, t:t+8]
                    if block.shape == (8, 8):
                        # SVD embedding with spreading
                        U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
                        
                        # Enhanced modulation
                        modulation = alpha * (2 * bit - 1) * (1 + 0.1 * spread_val)
                        Sigma[0] = Sigma[0] * (1 + modulation)
                        
                        # Reconstruct block
                        block_mod = U @ np.diag(Sigma) @ Vh
                        S_mag_mod[f:f+8, t:t+8] = block_mod
    
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    
    # Store reference data
    ref_data = {
        'sync_pattern': sync_pattern,
        'payload_length': len(bits),
        'alpha_adaptive': alpha_adaptive.tolist(),
        'spreading_seq': spreading_seq.tolist(),
        'f_min': f_min,
        'f_max': f_max
    }
    
    return S_complex_mod, ref_data

def calculate_nc(original_bits, extracted_bits):
    """Calculate Normalized Correlation between original and extracted bits"""
    if len(original_bits) != len(extracted_bits):
        return 0.0
    
    # Convert to numpy arrays
    orig = np.array(original_bits, dtype=float)
    extr = np.array(extracted_bits, dtype=float)
    
    # Calculate correlation
    numerator = np.sum((orig - np.mean(orig)) * (extr - np.mean(extr)))
    denominator = np.sqrt(np.sum((orig - np.mean(orig))**2) * np.sum((extr - np.mean(extr))**2))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator

def improved_detect_svd_stft(wave, sr):
    """Improved detection with synchronization and spreading compensation"""
    # Convert torch tensor to numpy
    if isinstance(wave, torch.Tensor):
        wave_np = wave.numpy().flatten()
    else:
        wave_np = wave.flatten()
    
    # Compute STFT
    S = compute_stft(wave_np, sr, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    
    try:
        # Enhanced extraction with synchronization
        extracted_bits = enhanced_extract_with_sync(S)
        
        if len(extracted_bits) >= len(PAYLOAD):
            # Calculate BER for payload bits (skip sync pattern)
            payload_bits = extracted_bits[6:6+len(PAYLOAD)]  # Skip 6-bit sync
            if len(payload_bits) == len(PAYLOAD):
                ber = sum(1 for a, b in zip(payload_bits, PAYLOAD) if a != b) / len(PAYLOAD)
                nc = calculate_nc(PAYLOAD, payload_bits)
                return payload_bits, ber, nc
            else:
                return [], 1.0, 0.0
        else:
            return [], 1.0, 0.0
            
    except Exception as e:
        return [], 1.0, 0.0

def enhanced_extract_with_sync(S_complex):
    """Enhanced extraction with synchronization search"""
    S_mag = np.abs(S_complex)
    rows, cols = S_mag.shape
    
    # Search for sync pattern
    sync_pattern = [1, 0, 1, 0, 1, 0]
    f_min = int(rows * 0.2)
    f_max = int(rows * 0.8)
    
    # Multiple detection attempts with different parameters
    detection_attempts = []
    
    for attempt in range(5):
        # Vary detection parameters
        alpha_base = 0.1 * (0.8 + 0.4 * attempt / 4)
        
        extracted_bits = []
        
        for i in range(len(sync_pattern) + len(PAYLOAD)):
            # Select frequency band
            f_center = f_min + i * (f_max - f_min) // (len(sync_pattern) + len(PAYLOAD))
            f_start = max(0, f_center - 4)
            f_end = min(rows, f_center + 4)
            
            # Collect detection scores from frequency band
            scores = []
            
            for f in range(f_start, f_end):
                for t in range(0, cols, 8):
                    if t + 8 <= cols:
                        block = S_mag[f:f+8, t:t+8]
                        if block.shape == (8, 8):
                            U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
                            
                            # Multiple detection methods
                            # Method 1: Threshold-based
                            threshold = np.mean(Sigma) * alpha_base
                            bit1 = 1 if Sigma[0] > threshold else 0
                            
                            # Method 2: Relative change
                            bit2 = 1 if Sigma[0] > np.median(Sigma) else 0
                            
                            # Method 3: Pattern matching
                            bit3 = 1 if Sigma[0] > np.percentile(Sigma, 75) else 0
                            
                            # Combine methods
                            bit = 1 if (bit1 + bit2 + bit3) >= 2 else 0
                            scores.append(bit)
            
            if scores:
                # Majority voting
                extracted_bits.append(1 if np.mean(scores) > 0.5 else 0)
            else:
                extracted_bits.append(0)
        
        detection_attempts.append(extracted_bits)
    
    # Find best detection attempt by sync pattern matching
    best_attempt = None
    best_sync_score = -1
    
    for attempt in detection_attempts:
        if len(attempt) >= len(sync_pattern):
            sync_bits = attempt[:len(sync_pattern)]
            sync_score = sum(1 for a, b in zip(sync_bits, sync_pattern) if a == b)
            if sync_score > best_sync_score:
                best_sync_score = sync_score
                best_attempt = attempt
    
    return best_attempt if best_attempt else []

def resample_roundtrip(wave, sr=SR, mid=INTERMEDIATE):
    """16k -> 32k -> 16k resampling"""
    to_mid = torchaudio.transforms.Resample(orig_freq=sr, new_freq=mid)
    to_orig = torchaudio.transforms.Resample(orig_freq=mid, new_freq=sr)
    with torch.no_grad():
        w_mid = to_mid(wave)
        w_back = to_orig(w_mid)
    L = wave.shape[-1]
    if w_back.shape[-1] < L:
        w_back = torch.nn.functional.pad(w_back, (0, L - w_back.shape[-1]))
    else:
        w_back = w_back[..., :L]
    return w_back

def detect_score(wave, sr):
    out = improved_detect_svd_stft(wave, sr)
    if isinstance(out, tuple) and len(out) == 3:
        bits, ber, nc = out
        # Return NC as the detection score (higher is better)
        return float(nc)
    return 0.0

def load_wavs(folder):
    import glob
    import os
    paths = sorted(glob.glob(os.path.join(folder, "*.wav")))
    waves = []
    for p in paths:
        w, sr = torchaudio.load(p)
        if sr != SR:
            w = torchaudio.functional.resample(w, sr, SR)
        if w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
        waves.append(w)
    return waves

def build_attacked_sets(waves):
    wm_attacked = []
    clean_attacked = []
    for w in waves:
        w = w.clone()
        w_wm = improved_embed_svd_stft(w, SR, payload=None)
        wm_attacked.append(resample_roundtrip(w_wm, SR, INTERMEDIATE))
        clean_attacked.append(resample_roundtrip(w, SR, INTERMEDIATE))
    return wm_attacked, clean_attacked

def evaluate_resample_only(wm_attacked, clean_attacked):
    # Get detailed results for analysis
    wm_results = []
    clean_results = []
    
    for w in wm_attacked:
        result = improved_detect_svd_stft(w, SR)
        if len(result) == 3:
            bits, ber, nc = result
            wm_results.append({'ber': ber, 'nc': nc})
        else:
            wm_results.append({'ber': 1.0, 'nc': 0.0})
    
    for w in clean_attacked:
        result = improved_detect_svd_stft(w, SR)
        if len(result) == 3:
            bits, ber, nc = result
            clean_results.append({'ber': ber, 'nc': nc})
        else:
            clean_results.append({'ber': 1.0, 'nc': 0.0})
    
    # Use NC for detection scores
    scores = [r['nc'] for r in wm_results] + [r['nc'] for r in clean_results]
    labels = [1]*len(wm_attacked) + [0]*len(clean_attacked)

    cand = sorted(set(scores))
    if len(cand) > 400:
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

    try:
        auc = float(roc_auc_score(labels_arr, scores_arr))
    except Exception:
        auc = float("nan")

    # Calculate average BER and NC
    wm_ber_avg = np.mean([r['ber'] for r in wm_results])
    wm_nc_avg = np.mean([r['nc'] for r in wm_results])
    clean_ber_avg = np.mean([r['ber'] for r in clean_results])
    clean_nc_avg = np.mean([r['nc'] for r in clean_results])

    return best, auc, {
        'wm_ber_avg': wm_ber_avg,
        'wm_nc_avg': wm_nc_avg,
        'clean_ber_avg': clean_ber_avg,
        'clean_nc_avg': clean_nc_avg
    }

if __name__ == "__main__":
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
    print("Testing improved SVD_STFT with frequency-domain spreading and synchronization...")
    
    wm_attacked, clean_attacked = build_attacked_sets(waves)
    best, auc, avg_metrics = evaluate_resample_only(wm_attacked, clean_attacked)
    print(f"[Improved Resample 32k round-trip] Acc={best['acc']:.3f} (TPR={best['tpr']:.3f}/FPR={best['fpr']:.3f}) at threshold={best['thr']:.4f}, AUC={auc:.3f}")
    print(f"Average BER (Watermarked): {avg_metrics['wm_ber_avg']:.3f}")
    print(f"Average NC (Watermarked): {avg_metrics['wm_nc_avg']:.3f}")
    print(f"Average BER (Clean): {avg_metrics['clean_ber_avg']:.3f}")
    print(f"Average NC (Clean): {avg_metrics['clean_nc_avg']:.3f}")
    
    # Save results
    import json
    results = {
        "attack": "improved_resample_32k_roundtrip",
        "num_files": len(waves),
        "best_accuracy": best["acc"],
        "best_threshold": best["thr"],
        "true_positive_rate": best["tpr"],
        "false_positive_rate": best["fpr"],
        "roc_auc": auc,
        "improvements": [
            "frequency_domain_spreading",
            "synchronization_markers", 
            "adaptive_embedding_strength",
            "multiple_detection_methods",
            "majority_voting"
        ],
        "average_metrics": avg_metrics
    }
    
    os.makedirs("audioseal_eval", exist_ok=True)
    with open("audioseal_eval/improved_resample_attack_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to audioseal_eval/improved_resample_attack_results.json")
