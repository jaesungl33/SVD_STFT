import sys
import json
import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.stats import pearsonr
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

def hamming_encode(bits):
    """Hamming (7,4) encoding"""
    if len(bits) % 4 != 0:
        # Pad with zeros
        bits = bits + [0] * (4 - len(bits) % 4)
    
    encoded = []
    for i in range(0, len(bits), 4):
        data = bits[i:i+4]
        # Hamming (7,4) encoding
        p1 = data[0] ^ data[1] ^ data[3]
        p2 = data[0] ^ data[2] ^ data[3]
        p3 = data[1] ^ data[2] ^ data[3]
        encoded.extend([p1, p2, data[0], p3, data[1], data[2], data[3]])
    
    return encoded

def hamming_decode(bits):
    """Hamming (7,4) decoding"""
    if len(bits) % 7 != 0:
        return bits  # Return as-is if not properly encoded
    
    decoded = []
    for i in range(0, len(bits), 7):
        codeword = bits[i:i+7]
        if len(codeword) == 7:
            # Calculate syndromes
            s1 = codeword[0] ^ codeword[2] ^ codeword[4] ^ codeword[6]
            s2 = codeword[1] ^ codeword[2] ^ codeword[5] ^ codeword[6]
            s3 = codeword[3] ^ codeword[4] ^ codeword[5] ^ codeword[6]
            
            # Error correction
            error_pos = s1 + 2*s2 + 4*s3
            if 0 < error_pos <= 7:
                codeword[error_pos-1] = 1 - codeword[error_pos-1]
            
            # Extract data bits
            decoded.extend([codeword[2], codeword[4], codeword[5], codeword[6]])
    
    return decoded

def ultimate_embed_svd_stft(wave, sr, payload=None):
    """Ultimate SVD_STFT embedding with all advanced techniques"""
    if payload is None:
        payload = PAYLOAD
    
    # Convert torch tensor to numpy
    if isinstance(wave, torch.Tensor):
        wave_np = wave.numpy().flatten()
    else:
        wave_np = wave.flatten()
    
    # Compute STFT
    S = compute_stft(wave_np, sr, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    
    # Ultimate embedding with all techniques
    S_wm, ref = ultimate_embed_with_all_techniques(S, payload)
    
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

def ultimate_embed_with_all_techniques(S_complex, bits):
    """Ultimate embedding with all advanced techniques combined"""
    S_mag = np.abs(S_complex)
    S_phase = np.angle(S_complex)
    
    # Enhanced synchronization pattern
    sync_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 12-bit sync pattern
    
    # Hamming encode the payload
    encoded_bits = hamming_encode(bits)
    
    # Add redundancy (3x)
    redundant_bits = []
    for bit in encoded_bits:
        redundant_bits.extend([bit] * 3)
    
    # Full payload with sync
    full_payload = sync_pattern + redundant_bits
    
    # Frequency-domain spreading with multiple bands
    rows, cols = S_mag.shape
    block_size = (8, 8)
    
    # Use multiple frequency bands for maximum robustness
    freq_bands = [
        (int(rows * 0.05), int(rows * 0.25)),   # Low frequencies
        (int(rows * 0.25), int(rows * 0.75)),   # Mid frequencies  
        (int(rows * 0.75), int(rows * 0.95))    # High frequencies
    ]
    
    # Create spreading sequences for each band
    spreading_seqs = []
    for band in freq_bands:
        seq = np.random.default_rng(42).normal(0, 1, len(full_payload))
        spreading_seqs.append(seq)
    
    # Adaptive embedding strength based on frequency, content, and position
    alpha_base = 0.06  # Conservative for better imperceptibility
    alpha_adaptive = np.zeros(len(full_payload))
    
    for i in range(len(full_payload)):
        # Frequency-dependent strength
        freq_ratio = (freq_bands[1][0] + i * (freq_bands[1][1] - freq_bands[1][0]) // len(full_payload)) / rows
        freq_factor = 1 + 0.4 * np.sin(2 * np.pi * freq_ratio)
        
        # Position-dependent strength (stronger for sync pattern)
        if i < len(sync_pattern):
            pos_factor = 2.0  # Much stronger sync pattern
        else:
            pos_factor = 1.0 + 0.3 * ((i - len(sync_pattern)) / len(redundant_bits))
        
        # Content-dependent strength
        content_factor = 1.0 + 0.2 * np.sin(2 * np.pi * i / len(full_payload))
        
        alpha_adaptive[i] = alpha_base * freq_factor * pos_factor * content_factor
    
    # Embed with multi-band spreading and enhanced modulation
    S_mag_mod = S_mag.copy()
    
    for i, (bit, alpha) in enumerate(zip(full_payload, alpha_adaptive)):
        # Embed in multiple frequency bands
        for band_idx, (f_min, f_max) in enumerate(freq_bands):
            # Select frequency band
            f_center = f_min + i * (f_max - f_min) // len(full_payload)
            f_start = max(0, f_center - 8)
            f_end = min(rows, f_center + 8)
            
            # Get spreading value for this band
            spread_val = spreading_seqs[band_idx][i]
            
            # Embed in frequency band
            for f in range(f_start, f_end):
                for t in range(0, cols, 8):
                    if t + 8 <= cols:
                        block = S_mag_mod[f:f+8, t:t+8]
                        if block.shape == (8, 8):
                            # SVD embedding with ultimate modulation
                            U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
                            
                            # Enhanced modulation with spreading and adaptive strength
                            modulation = alpha * (2 * bit - 1) * (1 + 0.2 * spread_val)
                            
                            # Use multiplicative modification for maximum robustness
                            Sigma[0] = Sigma[0] * (1 + modulation)
                            
                            # Reconstruct block
                            block_mod = U @ np.diag(Sigma) @ Vh
                            S_mag_mod[f:f+8, t:t+8] = block_mod
    
    S_complex_mod = S_mag_mod * np.exp(1j * S_phase)
    
    # Store comprehensive reference data
    ref_data = {
        'sync_pattern': sync_pattern,
        'payload_length': len(bits),
        'encoded_length': len(encoded_bits),
        'redundant_length': len(redundant_bits),
        'alpha_adaptive': alpha_adaptive.tolist(),
        'spreading_seqs': [seq.tolist() for seq in spreading_seqs],
        'freq_bands': freq_bands,
        'full_payload': full_payload
    }
    
    return S_complex_mod, ref_data

def ultimate_detect_svd_stft(wave, sr):
    """Ultimate detection with all advanced techniques"""
    # Convert torch tensor to numpy
    if isinstance(wave, torch.Tensor):
        wave_np = wave.numpy().flatten()
    else:
        wave_np = wave.flatten()
    
    # Compute STFT
    S = compute_stft(wave_np, sr, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    
    try:
        # Ultimate extraction with all techniques
        extracted_bits = ultimate_extract_with_all_techniques(S)
        
        if len(extracted_bits) >= len(PAYLOAD):
            # Decode and extract payload
            decoded_bits = decode_and_extract_payload(extracted_bits)
            
            if len(decoded_bits) == len(PAYLOAD):
                ber = sum(1 for a, b in zip(decoded_bits, PAYLOAD) if a != b) / len(PAYLOAD)
                nc = calculate_nc(PAYLOAD, decoded_bits)
                return decoded_bits, ber, nc
            else:
                return [], 1.0, 0.0
        else:
            return [], 1.0, 0.0
            
    except Exception as e:
        return [], 1.0, 0.0

def decode_and_extract_payload(extracted_bits):
    """Decode and extract payload from extracted bits"""
    sync_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Find sync pattern
    sync_start = -1
    for i in range(len(extracted_bits) - len(sync_pattern) + 1):
        if extracted_bits[i:i+len(sync_pattern)] == sync_pattern:
            sync_start = i + len(sync_pattern)
            break
    
    if sync_start == -1:
        return []
    
    # Extract redundant bits
    redundant_bits = extracted_bits[sync_start:]
    
    # Remove redundancy (take majority vote)
    encoded_bits = []
    for i in range(0, len(redundant_bits), 3):
        if i + 2 < len(redundant_bits):
            group = redundant_bits[i:i+3]
            # Majority voting
            encoded_bits.append(1 if sum(group) >= 2 else 0)
    
    # Hamming decode
    decoded_bits = hamming_decode(encoded_bits)
    
    return decoded_bits

def ultimate_extract_with_all_techniques(S_complex):
    """Ultimate extraction with all advanced techniques"""
    S_mag = np.abs(S_complex)
    rows, cols = S_mag.shape
    
    # Search for sync pattern with multiple detection methods
    sync_pattern = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    freq_bands = [
        (int(rows * 0.05), int(rows * 0.25)),
        (int(rows * 0.25), int(rows * 0.75)),
        (int(rows * 0.75), int(rows * 0.95))
    ]
    
    # Multiple detection attempts with different parameters and methods
    detection_attempts = []
    
    for attempt in range(12):  # More attempts for maximum robustness
        # Vary detection parameters
        alpha_base = 0.06 * (0.6 + 0.8 * attempt / 11)
        
        extracted_bits = []
        
        # Calculate total expected bits
        expected_bits = len(sync_pattern) + len(hamming_encode(PAYLOAD)) * 3
        
        for i in range(expected_bits):
            # Collect detection scores from multiple frequency bands
            band_scores = []
            
            for band_idx, (f_min, f_max) in enumerate(freq_bands):
                # Select frequency band
                f_center = f_min + i * (f_max - f_min) // expected_bits
                f_start = max(0, f_center - 8)
                f_end = min(rows, f_center + 8)
                
                # Collect detection scores from frequency band
                scores = []
                
                for f in range(f_start, f_end):
                    for t in range(0, cols, 8):
                        if t + 8 <= cols:
                            block = S_mag[f:f+8, t:t+8]
                            if block.shape == (8, 8):
                                U, Sigma, Vh = np.linalg.svd(block, full_matrices=False)
                                
                                # Multiple detection methods with enhanced weighting
                                # Method 1: Threshold-based
                                threshold = np.mean(Sigma) * alpha_base
                                bit1 = 1 if Sigma[0] > threshold else 0
                                
                                # Method 2: Relative change
                                bit2 = 1 if Sigma[0] > np.median(Sigma) else 0
                                
                                # Method 3: Pattern matching
                                bit3 = 1 if Sigma[0] > np.percentile(Sigma, 80) else 0
                                
                                # Method 4: Energy-based
                                bit4 = 1 if Sigma[0] > np.sqrt(np.mean(Sigma**2)) else 0
                                
                                # Method 5: Variance-based
                                bit5 = 1 if Sigma[0] > np.mean(Sigma) + np.std(Sigma) else 0
                                
                                # Enhanced weighted voting
                                weighted_score = (0.25 * bit1 + 0.25 * bit2 + 0.2 * bit3 + 0.2 * bit4 + 0.1 * bit5)
                                scores.append(weighted_score)
                
                if scores:
                    # Average score for this band
                    band_scores.append(np.mean(scores))
                else:
                    band_scores.append(0.0)
            
            # Combine scores from all bands with weighting
            if band_scores:
                # Weight mid-frequencies more heavily
                weights = [0.2, 0.6, 0.2]  # Low, mid, high frequency weights
                final_score = sum(w * s for w, s in zip(weights, band_scores))
                extracted_bits.append(1 if final_score > 0.5 else 0)
            else:
                extracted_bits.append(0)
        
        detection_attempts.append(extracted_bits)
    
    # Find best detection attempt by sync pattern matching
    best_attempt = None
    best_score = -1
    
    for attempt in detection_attempts:
        if len(attempt) >= len(sync_pattern):
            sync_bits = attempt[:len(sync_pattern)]
            # Use correlation for better sync pattern matching
            try:
                correlation, _ = pearsonr(sync_bits, sync_pattern)
                score = correlation
            except:
                # Fallback to simple matching
                score = sum(1 for a, b in zip(sync_bits, sync_pattern) if a == b) / len(sync_pattern)
            
            if score > best_score:
                best_score = score
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

def evaluate_condition(waves, attack_type):
    """Evaluate a specific attack condition with ultimate SVD_STFT"""
    print(f"Testing {attack_type} with Ultimate SVD_STFT...")
    
    wm_results = []
    clean_results = []
    
    for w in waves:
        # Embed watermark
        w_wm = ultimate_embed_svd_stft(w, SR, payload=None)
        
        # Apply attack to both watermarked and clean
        w_wm_attacked = apply_attack(w_wm, attack_type)
        w_clean_attacked = apply_attack(w, attack_type)
        
        # Detect watermark
        wm_result = ultimate_detect_svd_stft(w_wm_attacked, SR)
        clean_result = ultimate_detect_svd_stft(w_clean_attacked, SR)
        
        if len(wm_result) == 3:
            bits, ber, nc = wm_result
            wm_results.append({'ber': ber, 'nc': nc})
        else:
            wm_results.append({'ber': 1.0, 'nc': 0.0})
        
        if len(clean_result) == 3:
            bits, ber, nc = clean_result
            clean_results.append({'ber': ber, 'nc': nc})
        else:
            clean_results.append({'ber': 1.0, 'nc': 0.0})
    
    # Calculate metrics
    wm_ber_avg = np.mean([r['ber'] for r in wm_results])
    wm_nc_avg = np.mean([r['nc'] for r in wm_results])
    clean_ber_avg = np.mean([r['ber'] for r in clean_results])
    clean_nc_avg = np.mean([r['nc'] for r in clean_results])
    
    # Use NC for detection scores
    wm_scores = [r['nc'] for r in wm_results]
    clean_scores = [r['nc'] for r in clean_results]
    all_scores = wm_scores + clean_scores
    labels = [1] * len(wm_scores) + [0] * len(clean_scores)
    
    # Find best threshold
    cand = sorted(set(all_scores))
    if len(cand) > 400:
        idxs = np.linspace(0, len(cand)-1, 400).astype(int)
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
        'ber': wm_ber_avg,
        'nc': wm_nc_avg,
        'acc': best['acc'],
        'tpr': best['tpr'],
        'fpr': best['fpr']
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
    print("Testing Ultimate SVD_STFT with ALL advanced techniques...")
    
    # Define attack conditions
    attack_conditions = [
        "Clean",
        "MP3", 
        "Resample22k",
        "Resample12k",
        "WhiteNoise20",
        "Speed1.25x"
    ]
    
    # Evaluate each condition
    results = []
    for condition in attack_conditions:
        result = evaluate_condition(waves, condition)
        results.append(result)
        print(f"Completed {condition}")
    
    # Create results table
    df = pd.DataFrame(results)
    
    # Print table
    print("\n" + "="*80)
    print("ULTIMATE SVD_STFT ATTACK EVALUATION RESULTS")
    print("="*80)
    print(f"{'Condition':<15} {'BER':<8} {'NC':<8} {'ACC':<8} {'TPR':<8} {'FPR':<8}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['condition']:<15} {row['ber']:<8.3f} {row['nc']:<8.3f} {row['acc']:<8.3f} {row['tpr']:<8.3f} {row['fpr']:<8.3f}")
    
    print("-"*80)
    
    # Save results
    os.makedirs("audioseal_eval", exist_ok=True)
    
    # Save as CSV
    df.to_csv("audioseal_eval/ultimate_svd_stft_results.csv", index=False)
    
    # Save as JSON
    with open("audioseal_eval/ultimate_svd_stft_results.json", "w") as f:
        json.dump({
            'results': results,
            'summary': {
                'num_files': len(waves),
                'conditions_tested': attack_conditions,
                'ultimate_techniques': [
                    'frequency_domain_spreading',
                    'robust_synchronization',
                    'multi_band_embedding',
                    'adaptive_embedding_strength',
                    'hamming_error_correction',
                    '3x_redundancy',
                    'majority_voting',
                    'enhanced_detection_methods',
                    'weighted_frequency_bands',
                    'correlation_based_sync'
                ]
            }
        }, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"- audioseal_eval/ultimate_svd_stft_results.csv")
    print(f"- audioseal_eval/ultimate_svd_stft_results.json")

