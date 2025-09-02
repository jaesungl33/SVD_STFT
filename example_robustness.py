#!/usr/bin/env python3
"""
Example script demonstrating the robustness evaluation system.
This script shows how to use the system with optimized parameters and interpret results.
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils.robustness import RobustnessEvaluator
from stft.svd_stft import embed_svd_stft, extract_svd_stft, compute_stft
from stft.stft_transform import reconstruct_audio


def create_example_audio(duration: float = 10.0, sample_rate: int = 16000) -> np.ndarray:
    """Create a more complex test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a complex signal with multiple frequencies and some variation
    audio = (0.4 * np.sin(2 * np.pi * 440 * t) +      # A4 note
             0.3 * np.sin(2 * np.pi * 880 * t) +      # A5 note
             0.2 * np.sin(2 * np.pi * 1760 * t) +     # A6 note
             0.1 * np.sin(2 * np.pi * 220 * t) +      # A3 note
             0.05 * np.sin(2 * np.pi * 110 * t))      # A2 note
    
    # Add some amplitude modulation
    audio *= (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
    
    return audio


def demonstrate_robustness():
    """Demonstrate robustness evaluation with optimized parameters."""
    print("=" * 60)
    print("SVD_STFT ROBUSTNESS EVALUATION DEMONSTRATION")
    print("=" * 60)
    
    # Create test audio
    sample_rate = 16000
    audio = create_example_audio(duration=8.0, sample_rate=sample_rate)
    print(f"Created test audio: {len(audio)/sample_rate:.1f} seconds")
    
    # Create watermark bits
    watermark_bits = np.random.randint(0, 2, 32).tolist()
    print(f"Generated watermark: {watermark_bits[:10]}... (32 bits total)")
    
    # Try different parameters to find better ones
    print("\nTesting different watermark parameters...")
    
    # Parameter combinations to test
    param_combinations = [
        {'alpha': 0.05, 'block_size': (8, 8)},
        {'alpha': 0.1, 'block_size': (8, 8)},
        {'alpha': 0.2, 'block_size': (8, 8)},
        {'alpha': 0.1, 'block_size': (16, 16)},
        {'alpha': 0.2, 'block_size': (16, 16)},
    ]
    
    best_ber = 1.0
    best_params = None
    best_watermarked_audio = None
    best_extract_params = None
    
    for params in param_combinations:
        try:
            # Embed watermark
            S = compute_stft(audio, sample_rate, n_fft=256, hop_length=64, window='hann')
            S_watermarked, sigma_ref = embed_svd_stft(
                S, watermark_bits, 
                alpha=params['alpha'], 
                block_size=params['block_size'], 
                key=42
            )
            watermarked_audio = reconstruct_audio(S_watermarked, 256, 'hann')
            
            # Ensure same length
            min_len = min(len(audio), len(watermarked_audio))
            watermarked_audio = watermarked_audio[:min_len]
            
            # Test clean extraction
            S_test = compute_stft(watermarked_audio, sample_rate, n_fft=256, hop_length=64, window='hann')
            extracted_bits = extract_svd_stft(
                S_test, 
                params['alpha'],
                params['block_size'],
                42,
                len(watermark_bits),
                sigma_ref=sigma_ref
            )
            
            # Calculate BER
            ber = sum(1 for i in range(len(watermark_bits)) if watermark_bits[i] != extracted_bits[i]) / len(watermark_bits)
            
            print(f"  α={params['alpha']}, block={params['block_size']}: BER={ber:.4f}")
            
            if ber < best_ber:
                best_ber = ber
                best_params = params
                best_watermarked_audio = watermarked_audio
                best_extract_params = {
                    'alpha': params['alpha'],
                    'block_size': params['block_size'],
                    'key': 42,
                    'num_bits': len(watermark_bits),
                    'sigma_ref': sigma_ref
                }
                
        except Exception as e:
            print(f"  α={params['alpha']}, block={params['block_size']}: Error - {e}")
            continue
    
    if best_params is None:
        print("No successful parameter combination found!")
        return
    
    print(f"\nBest parameters: α={best_params['alpha']}, block={best_params['block_size']}")
    print(f"Best clean BER: {best_ber:.4f}")
    
    # Initialize evaluator
    evaluator = RobustnessEvaluator(sample_rate)
    
    def extract_function(audio_signal, **params):
        """Extract watermark from audio."""
        S = compute_stft(audio_signal, sample_rate, n_fft=256, hop_length=64, window='hann')
        return extract_svd_stft(
            S, 
            params['alpha'],
            params['block_size'],
            params['key'],
            params['num_bits'],
            sigma_ref=params['sigma_ref']
        )
    
    # Test robustness with best parameters
    print("\nTesting robustness against attacks...")
    results = evaluator.evaluate_robustness(
        original_audio=audio,
        watermarked_audio=best_watermarked_audio,
        extract_function=extract_function,
        extract_params=best_extract_params,
        watermark_bits=watermark_bits,
        test_config="evaluation"
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("=" * 60)
    
    successful_attacks = 0
    total_attacks = len(results)
    
    for attack_name, result in results.items():
        ber = result.get('ber', 1.0)
        snr = result.get('snr', -np.inf)
        success = result.get('success', False)
        
        if success:
            successful_attacks += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{attack_name:<20} BER: {ber:<8.4f} SNR: {snr:<8.2f} dB {status}")
    
    success_rate = (successful_attacks / total_attacks) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({successful_attacks}/{total_attacks})")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Group attacks by type
    filter_attacks = ['bandpass_filter', 'highpass_filter', 'lowpass_filter']
    noise_attacks = ['pink_noise', 'white_noise']
    codec_attacks = ['aac', 'mp3', 'encodec']
    other_attacks = ['speed_change', 'resample', 'boost_audio', 'duck_audio', 'echo', 'smooth']
    
    attack_groups = {
        'Filter Attacks': filter_attacks,
        'Noise Attacks': noise_attacks,
        'Codec Attacks': codec_attacks,
        'Other Attacks': other_attacks
    }
    
    for group_name, attacks in attack_groups.items():
        group_success = 0
        group_total = 0
        
        for attack in attacks:
            if attack in results:
                if results[attack].get('success', False):
                    group_success += 1
                group_total += 1
        
        if group_total > 0:
            group_rate = (group_success / group_total) * 100
            print(f"{group_name:<20}: {group_rate:>6.1f}% ({group_success}/{group_total})")
    
    # Save example files
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    sf.write(output_dir / "original.wav", audio, sample_rate)
    sf.write(output_dir / "watermarked.wav", best_watermarked_audio, sample_rate)
    
    print(f"\nExample audio files saved to: {output_dir}")
    
    # Generate detailed report
    print("\n" + "=" * 60)
    report = evaluator.generate_report(results)
    print(report)
    
    # Save results
    import json
    with open(output_dir / "example_results.json", 'w') as f:
        json.dump({
            'parameters': best_params,
            'clean_ber': best_ber,
            'results': results,
            'success_rate': success_rate
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_dir / 'example_results.json'}")


if __name__ == "__main__":
    demonstrate_robustness()
