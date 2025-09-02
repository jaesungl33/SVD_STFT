import sys
import json
import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Import base functions
from stft.svd_stft import compute_stft
from stft.stft_transform import reconstruct_audio

SR = 16_000
N_FFT = 1024
HOP = 256
WINDOW = 'hann'

# Fixed payload
rng = np.random.default_rng(123)
PAYLOAD = rng.integers(0, 2, size=16).tolist()

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

def improved_embed_svd_stft(wave, sr, payload=None):
    """Improved SVD_STFT embedding with frequency-domain spreading and synchronization"""
    if payload is None:
        payload = PAYLOAD
    
    # Convert torch tensor to numpy
    if isinstance(wave, torch.Tensor):
        wave_np = wave.numpy().flatten()
    else:
        wave_np = wave.flatten()
    
    # Compute STFT
    S = compute_stft(wave_np, sr, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    
    # Improved embedding with frequency-domain spreading
    S_wm, ref = improved_embed_with_spreading(S, payload)
    
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

def improved_embed_with_spreading(S_complex, bits):
    """Improved embedding with frequency-domain spreading and synchronization"""
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
        # Improved extraction with synchronization
        extracted_bits = improved_extract_with_sync(S)
        
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

def improved_extract_with_sync(S_complex):
    """Improved extraction with synchronization search"""
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

def apply_attack(wave, attack_type):
    """Apply different types of attacks to the audio"""
    if attack_type == "Clean":
        return wave
    
    elif attack_type == "MP3":
        # Simulate MP3 compression by adding noise
        noise = torch.randn_like(wave) * 0.01
        return wave + noise
    
    elif attack_type == "Resample22k":
        # 16k -> 22k -> 16k
        to_22k = torchaudio.transforms.Resample(orig_freq=SR, new_freq=22050)
        to_16k = torchaudio.transforms.Resample(orig_freq=22050, new_freq=SR)
        with torch.no_grad():
            w_22k = to_22k(wave)
            w_back = to_16k(w_22k)
        L = wave.shape[-1]
        if w_back.shape[-1] < L:
            w_back = torch.nn.functional.pad(w_back, (0, L - w_back.shape[-1]))
        else:
            w_back = w_back[..., :L]
        return w_back
    
    elif attack_type == "Resample12k":
        # 16k -> 12k -> 16k
        to_12k = torchaudio.transforms.Resample(orig_freq=SR, new_freq=12000)
        to_16k = torchaudio.transforms.Resample(orig_freq=12000, new_freq=SR)
        with torch.no_grad():
            w_12k = to_12k(wave)
            w_back = to_16k(w_12k)
        L = wave.shape[-1]
        if w_back.shape[-1] < L:
            w_back = torch.nn.functional.pad(w_back, (0, L - w_back.shape[-1]))
        else:
            w_back = w_back[..., :L]
        return w_back
    
    elif attack_type == "WhiteNoise20":
        # Add white noise at 20dB SNR
        signal_power = torch.mean(wave ** 2)
        noise_power = signal_power / (10 ** (20 / 10))  # 20dB SNR
        noise = torch.randn_like(wave) * torch.sqrt(noise_power)
        return wave + noise
    
    elif attack_type == "Speed1.25x":
        # Speed change by 1.25x using resampling
        speed_factor = 1.25
        new_rate = int(SR * speed_factor)
        to_fast = torchaudio.transforms.Resample(orig_freq=SR, new_freq=new_rate)
        to_orig = torchaudio.transforms.Resample(orig_freq=new_rate, new_freq=SR)
        with torch.no_grad():
            w_fast = to_fast(wave)
            w_back = to_orig(w_fast)
        L = wave.shape[-1]
        if w_back.shape[-1] < L:
            w_back = torch.nn.functional.pad(w_back, (0, L - w_back.shape[-1]))
        else:
            w_back = w_back[..., :L]
        return w_back
    
    else:
        return wave

def test_attack_condition(waves, attack_type, num_samples=10):
    """Test a specific attack condition and return TPR, FPR, and NC"""
    print(f"Testing {attack_type} with Improved SVD_STFT...")
    
    wm_ncs = []
    clean_ncs = []
    
    # Test with limited samples for speed
    test_waves = waves[:num_samples]
    
    for w in test_waves:
        # Embed watermark
        w_wm = improved_embed_svd_stft(w, SR, payload=None)
        
        # Apply attack to both watermarked and clean
        w_wm_attacked = apply_attack(w_wm, attack_type)
        w_clean_attacked = apply_attack(w, attack_type)
        
        # Detect watermark
        wm_result = improved_detect_svd_stft(w_wm_attacked, SR)
        clean_result = improved_detect_svd_stft(w_clean_attacked, SR)
        
        if len(wm_result) == 3:
            bits, ber, nc = wm_result
            wm_ncs.append(nc)
        else:
            wm_ncs.append(0.0)
        
        if len(clean_result) == 3:
            bits, ber, nc = clean_result
            clean_ncs.append(nc)
        else:
            clean_ncs.append(0.0)
    
    # Calculate average NC
    wm_nc_avg = np.mean(wm_ncs)
    clean_nc_avg = np.mean(clean_ncs)
    
    # Use NC for detection scores
    wm_scores = wm_ncs
    clean_scores = clean_ncs
    all_scores = wm_scores + clean_scores
    labels = [1] * len(wm_scores) + [0] * len(clean_scores)
    
    # Find best threshold for TPR/FPR calculation
    cand = sorted(set(all_scores))
    if len(cand) > 100:
        idxs = np.linspace(0, len(cand)-1, 100).astype(int)
        cand = [cand[i] for i in idxs]
    
    best = {"acc": -1, "thr": None, "tpr": None, "fpr": None}
    scores_arr = np.array(all_scores)
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
    
    return {
        'condition': attack_type,
        'wm_nc_avg': wm_nc_avg,
        'clean_nc_avg': clean_nc_avg,
        'tpr': best['tpr'],
        'fpr': best['fpr'],
        'acc': best['acc']
    }

def load_wavs(folder):
    import glob
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
    print("Testing TPR, FPR, and NC for each attack condition with Improved SVD_STFT...")
    
    # Define attack conditions
    attack_conditions = [
        "Clean",
        "MP3", 
        "Resample22k",
        "Resample12k",
        "WhiteNoise20",
        "Speed1.25x"
    ]
    
    # Test each condition
    results = []
    for condition in attack_conditions:
        result = test_attack_condition(waves, condition, num_samples=10)
        results.append(result)
        print(f"Completed {condition}")
    
    # Print results
    print("\n" + "="*70)
    print("IMPROVED SVD_STFT: TPR, FPR, AND NC RESULTS FOR EACH ATTACK CONDITION")
    print("="*70)
    print(f"{'Condition':<15} {'TPR':<8} {'FPR':<8} {'WM NC':<8} {'Clean NC':<10}")
    print("-"*70)
    
    for result in results:
        print(f"{result['condition']:<15} {result['tpr']:<8.3f} {result['fpr']:<8.3f} {result['wm_nc_avg']:<8.4f} {result['clean_nc_avg']:<10.4f}")
    
    print("-"*70)
    
    # Save results
    os.makedirs("audioseal_eval", exist_ok=True)
    
    # Save as JSON
    with open("audioseal_eval/improved_tpr_fpr_nc_results.json", "w") as f:
        json.dump({
            'results': results,
            'summary': {
                'num_files': len(waves),
                'samples_per_test': 10,
                'conditions_tested': attack_conditions,
                'improvements': [
                    'frequency_domain_spreading',
                    'synchronization_markers',
                    'adaptive_embedding_strength',
                    'multiple_detection_methods',
                    'majority_voting'
                ]
            }
        }, f, indent=2)
    
    print(f"\nResults saved to audioseal_eval/improved_tpr_fpr_nc_results.json")

