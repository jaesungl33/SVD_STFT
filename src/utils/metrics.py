# src/utils/metrics.py
import numpy as np
from typing import List, Dict, Tuple
import warnings

def compute_sisnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Compute Scale-invariant Signal-to-Noise Ratio (SiSNR) in dB.
    
    Args:
        original: Original audio signal
        watermarked: Watermarked audio signal
        
    Returns:
        SiSNR value in dB
    """
    if len(original) != len(watermarked):
        min_len = min(len(original), len(watermarked))
        original = original[:min_len]
        watermarked = watermarked[:min_len]
    
    # Center the signals
    original_centered = original - np.mean(original)
    watermarked_centered = watermarked - np.mean(watermarked)
    
    # Compute the optimal scaling factor
    numerator = np.sum(original_centered * watermarked_centered)
    denominator = np.sum(watermarked_centered ** 2)
    
    if denominator == 0:
        return float('inf')
    
    alpha = numerator / denominator
    
    # Compute the scaled and shifted version
    watermarked_scaled = alpha * watermarked_centered
    
    # Compute SiSNR
    signal_power = np.sum(original_centered ** 2)
    noise_power = np.sum((original_centered - watermarked_scaled) ** 2)
    
    # Handle the case where noise_power is very small (essentially perfect reconstruction)
    if noise_power < 1e-12:  # Very small threshold
        # Return a high but finite value instead of infinity
        return 120.0  # 120 dB is considered excellent quality
    
    sisnr = 10 * np.log10(signal_power / noise_power)
    
    # Clamp to reasonable range
    sisnr = max(-100.0, min(120.0, sisnr))
    
    return float(sisnr)


def compute_ber(extracted_bits: List[int], true_bits: List[int]) -> float:
    """Compute Bit Error Rate (BER).
    
    Args:
        extracted_bits: Extracted watermark bits
        true_bits: Original watermark bits
        
    Returns:
        BER value between 0 and 1
    """
    if len(extracted_bits) != len(true_bits):
        min_len = min(len(extracted_bits), len(true_bits))
        extracted_bits = extracted_bits[:min_len]
        true_bits = true_bits[:min_len]
    
    if len(true_bits) == 0:
        return 1.0
    
    errors = sum(1 for b1, b2 in zip(extracted_bits, true_bits) if b1 != b2)
    ber = errors / len(true_bits)
    return float(ber)


def compute_nc(extracted_bits: List[int], true_bits: List[int]) -> float:
    """Compute Normalized Correlation (NC).
    
    Args:
        extracted_bits: Extracted watermark bits
        true_bits: Original watermark bits
        
    Returns:
        NC value between -1 and 1
    """
    if len(extracted_bits) != len(true_bits):
        min_len = min(len(extracted_bits), len(true_bits))
        extracted_bits = extracted_bits[:min_len]
        true_bits = true_bits[:min_len]
    
    if len(true_bits) == 0:
        return 0.0
    
    # Convert to numpy arrays
    extracted = np.array(extracted_bits, dtype=float)
    true = np.array(true_bits, dtype=float)
    
    # Compute correlation
    correlation = np.corrcoef(extracted, true)[0, 1]
    
    # Handle NaN values
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


def compute_tpr_fpr(extracted_bits: List[int], true_bits: List[int]) -> Tuple[float, float]:
    """Compute True Positive Rate (TPR) and False Positive Rate (FPR).
    
    Args:
        extracted_bits: Extracted watermark bits
        true_bits: Original watermark bits
        
    Returns:
        Tuple of (TPR, FPR) values between 0 and 1
    """
    if len(extracted_bits) != len(true_bits):
        min_len = min(len(extracted_bits), len(true_bits))
        extracted_bits = extracted_bits[:min_len]
        true_bits = true_bits[:min_len]
    
    if len(true_bits) == 0:
        return 0.0, 0.0
    
    # Convert to numpy arrays
    extracted = np.array(extracted_bits)
    true = np.array(true_bits)
    
    # True Positives: correctly identified 1s
    tp = np.sum((extracted == 1) & (true == 1))
    
    # False Positives: incorrectly identified 1s (extracted 1 when true was 0)
    fp = np.sum((extracted == 1) & (true == 0))
    
    # True Negatives: correctly identified 0s
    tn = np.sum((extracted == 0) & (true == 0))
    
    # False Negatives: incorrectly identified 0s (extracted 0 when true was 1)
    fn = np.sum((extracted == 0) & (true == 1))
    
    # Compute TPR (Sensitivity/Recall) and FPR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return float(tpr), float(fpr)


def compute_pesq(original: np.ndarray, watermarked: np.ndarray, sr: int) -> float:
    """Compute Perceptual Audio Quality Score (PAQS) - alternative to PESQ.
    
    Args:
        original: Original audio signal
        watermarked: Watermarked audio signal
        sr: Sample rate
        
    Returns:
        PAQS score (higher is better, similar to PESQ scale)
    """
    try:
        # Use spectral centroid distance as a perceptual quality metric
        import librosa
        
        if len(original) != len(watermarked):
            min_len = min(len(original), len(watermarked))
            original = original[:min_len]
            watermarked = watermarked[:min_len]
        
        # Compute spectral features
        spec_orig = np.abs(librosa.stft(original, n_fft=2048, hop_length=512))
        spec_wm = np.abs(librosa.stft(watermarked, n_fft=2048, hop_length=512))
        
        # Spectral centroid
        cent_orig = librosa.feature.spectral_centroid(S=spec_orig, sr=sr).flatten()
        cent_wm = librosa.feature.spectral_centroid(S=spec_wm, sr=sr).flatten()
        
        # Spectral rolloff
        roll_orig = librosa.feature.spectral_rolloff(S=spec_orig, sr=sr).flatten()
        roll_wm = librosa.feature.spectral_rolloff(S=spec_wm, sr=sr).flatten()
        
        # Spectral bandwidth
        bw_orig = librosa.feature.spectral_bandwidth(S=spec_orig, sr=sr).flatten()
        bw_wm = librosa.feature.spectral_bandwidth(S=spec_wm, sr=sr).flatten()
        
        # Compute perceptual distance
        cent_dist = np.mean(np.abs(cent_orig - cent_wm)) / (np.mean(cent_orig) + 1e-10)
        roll_dist = np.mean(np.abs(roll_orig - roll_wm)) / (np.mean(roll_orig) + 1e-10)
        bw_dist = np.mean(np.abs(bw_orig - bw_wm)) / (np.mean(bw_orig) + 1e-10)
        
        # Convert to quality score (higher is better, similar to PESQ scale)
        # PESQ typically ranges from -0.5 to 4.5, we'll map to 1.0 to 5.0
        perceptual_distance = (cent_dist + roll_dist + bw_dist) / 3
        paqs_score = 5.0 - 4.0 * perceptual_distance
        
        # Clamp to reasonable range
        paqs_score = max(1.0, min(5.0, paqs_score))
        
        return float(paqs_score)
        
    except Exception as e:
        warnings.warn(f"PAQS computation failed: {e}. Using PSNR fallback.")
        return compute_psnr(original, watermarked) / 20.0 + 1.0  # Scale PSNR to 1-5 range


def compute_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) as fallback for PESQ.
    
    Args:
        original: Original audio signal
        watermarked: Watermarked audio signal
        
    Returns:
        PSNR value in dB
    """
    if len(original) != len(watermarked):
        min_len = min(len(original), len(watermarked))
        original = original[:min_len]
        watermarked = watermarked[:min_len]
    
    mse = np.mean((original - watermarked) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # Normalize to [0, 1] range for PSNR calculation
    max_val = np.max(np.abs(original))
    if max_val > 0:
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = 0.0
    
    return float(psnr)


def compute_mse(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE).
    
    Args:
        original: Original audio signal
        watermarked: Watermarked audio signal
        
    Returns:
        MSE value
    """
    if len(original) != len(watermarked):
        min_len = min(len(original), len(watermarked))
        original = original[:min_len]
        watermarked = watermarked[:min_len]
    
    mse = np.mean((original - watermarked) ** 2)
    return float(mse)


def compute_ssim(original: np.ndarray, watermarked: np.ndarray, window_size: int = 11) -> float:
    """Compute Structural Similarity Index (SSIM).
    
    Args:
        original: Original audio signal
        watermarked: Watermarked audio signal
        window_size: Window size for SSIM computation
        
    Returns:
        SSIM value between -1 and 1
    """
    try:
        from skimage.metrics import structural_similarity
        
        if len(original) != len(watermarked):
            min_len = min(len(original), len(watermarked))
            original = original[:min_len]
            watermarked = watermarked[:min_len]
        
        # Reshape to 2D for SSIM computation
        original_2d = original.reshape(-1, 1)
        watermarked_2d = watermarked.reshape(-1, 1)
        
        ssim_score = structural_similarity(
            original_2d, watermarked_2d, 
            win_size=min(window_size, min(original_2d.shape)),
            data_range=original_2d.max() - original_2d.min()
        )
        
        return float(ssim_score)
        
    except ImportError:
        warnings.warn("scikit-image not available. SSIM computation skipped.")
        return 0.0
    except Exception as e:
        warnings.warn(f"SSIM computation failed: {e}")
        return 0.0


def evaluate_performance(orig: np.ndarray, mod: np.ndarray, extracted_bits: List[int], 
                        true_bits: List[int], sr: int) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        orig: Original audio signal
        mod: Watermarked audio signal
        extracted_bits: Extracted watermark bits
        true_bits: Original watermark bits
        sr: Sample rate
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['SiSNR'] = compute_sisnr(orig, mod)
    metrics['BER'] = compute_ber(extracted_bits, true_bits)
    metrics['NC'] = compute_nc(extracted_bits, true_bits)
    metrics['PESQ'] = compute_pesq(orig, mod, sr)
    metrics['PSNR'] = compute_psnr(orig, mod)
    metrics['MSE'] = compute_mse(orig, mod)
    metrics['SSIM'] = compute_ssim(orig, mod)
    
    # Watermark detection metrics
    tpr, fpr = compute_tpr_fpr(extracted_bits, true_bits)
    metrics['TPR'] = tpr
    metrics['FPR'] = fpr
    
    return metrics


def compute_pesq_drop(original_pesq: float, watermarked_pesq: float) -> float:
    """Compute PESQ drop (difference between original and watermarked PESQ).
    
    Args:
        original_pesq: PESQ score of original audio
        watermarked_pesq: PESQ score of watermarked audio
        
    Returns:
        PESQ drop value
    """
    return original_pesq - watermarked_pesq


def format_metrics_for_csv(metrics: Dict[str, float]) -> Dict[str, str]:
    """Format metrics for CSV export with proper precision.
    
    Args:
        metrics: Dictionary of metric values
        
    Returns:
        Dictionary with formatted string values
    """
    formatted = {}
    
    for key, value in metrics.items():
        if key in ['SiSNR', 'PSNR']:
            formatted[key] = f"{value:.2f}"
        elif key in ['BER', 'MSE']:
            formatted[key] = f"{value:.6f}"
        elif key in ['NC', 'SSIM', 'TPR', 'FPR']:
            formatted[key] = f"{value:.4f}"
        elif key == 'PESQ':
            formatted[key] = f"{value:.3f}"
        else:
            formatted[key] = f"{value:.4f}"
    
    return formatted 