# Robustness Evaluation System

This document describes the robustness evaluation system for SVD_STFT watermarking, which implements audio augmentations similar to AudioSeal for comprehensive testing of watermark robustness.

## Overview

The robustness evaluation system tests watermark resilience against various audio processing attacks and modifications. It implements 14 different attack types with both training and evaluation configurations, as specified in the AudioSeal reference.

## Attack Types

### 1. Bandpass Filter
- **Training**: 300Hz - 8000Hz
- **Evaluation**: 500Hz - 5000Hz
- **Description**: Combines highpass and lowpass filtering to allow a specific frequency band to pass through.

### 2. Highpass Filter
- **Training**: 500Hz cutoff
- **Evaluation**: 1500Hz cutoff
- **Description**: Uses a highpass filter to cut frequencies below a certain threshold.

### 3. Lowpass Filter
- **Training**: 5000Hz cutoff
- **Evaluation**: 500Hz cutoff
- **Description**: Applies a lowpass filter to cut frequencies above a cutoff frequency.

### 4. Speed Change
- **Training**: Random between 0.9 and 1.1
- **Evaluation**: Fixed at 1.25
- **Description**: Changes the speed of the audio by a factor close to 1.

### 5. Resample
- **Training**: 32kHz
- **Evaluation**: 32kHz
- **Description**: Upsamples to intermediate sample rate and then downsamples back to original rate.

### 6. Boost Audio
- **Training**: Factor 1.2
- **Evaluation**: Factor 10.0
- **Description**: Amplifies the audio by multiplying by a factor.

### 7. Duck Audio
- **Training**: Factor 0.8
- **Evaluation**: Factor 0.1
- **Description**: Reduces the volume of the audio by a multiplying factor.

### 8. Echo
- **Training**: Random delay 0.1-0.5s, random volume 0.1-0.5
- **Evaluation**: Fixed delay 0.5s, fixed volume 0.5
- **Description**: Applies an echo effect with delay and less loud copy of the original.

### 9. Pink Noise
- **Training**: Standard deviation 0.01
- **Evaluation**: Standard deviation 0.1
- **Description**: Adds pink noise for background noise effect.

### 10. White Noise
- **Training**: Standard deviation 0.001
- **Evaluation**: Standard deviation 0.05
- **Description**: Adds Gaussian noise to the waveform.

### 11. Smooth
- **Training**: Window size random between 2 and 10
- **Evaluation**: Window size fixed at 40
- **Description**: Smooths the audio signal using a moving average filter.

### 12. AAC Encoding
- **Training**: 128kbps
- **Evaluation**: 64kbps
- **Description**: Encodes the audio in AAC format and decodes back.

### 13. MP3 Encoding
- **Training**: 128kbps
- **Evaluation**: 32kbps
- **Description**: Encodes the audio in MP3 format and decodes back.

### 14. EnCodec
- **Training**: Default settings
- **Evaluation**: Default settings
- **Description**: Resamples at 24kHz, encodes with EnCodec (nq=16), and resamples back to 16kHz.

## Usage

### Quick Test

Run a simple test with synthetic audio:

```bash
python test_robustness.py
```

This will:
- Create a synthetic audio signal
- Embed a watermark
- Test against all attacks
- Display results
- Save test audio files

### Comprehensive Evaluation

For comprehensive evaluation on real audio files:

```bash
python robustness_evaluation.py --input_dir /path/to/audio/files --output_dir results
```

#### Command Line Options

- `--input_dir`: Directory containing audio files to test (required)
- `--output_dir`: Output directory for results (default: "robustness_results")
- `--sample_rate`: Target sample rate (default: 16000)
- `--num_bits`: Number of watermark bits (default: 64)
- `--alpha`: Watermark strength parameter (default: 0.1)
- `--block_size`: Block size as 'rows,cols' (default: "8,8")
- `--key`: Random seed for watermark generation (default: 42)
- `--test_config`: Test configuration - "training" or "evaluation" (default: "evaluation")
- `--max_files`: Maximum number of files to test (default: None)

#### Example Commands

```bash
# Test with evaluation configuration
python robustness_evaluation.py --input_dir 100sample_wav --output_dir eval_results

# Test with training configuration
python robustness_evaluation.py --input_dir 100sample_wav --output_dir train_results --test_config training

# Test with custom parameters
python robustness_evaluation.py --input_dir 100sample_wav --alpha 0.2 --block_size "16,16" --num_bits 128

# Test limited number of files
python robustness_evaluation.py --input_dir 100sample_wav --max_files 10
```

## Output Files

The evaluation generates several output files:

### 1. `detailed_results.json`
Contains detailed results for each file and attack, including:
- File path
- Watermark bits
- Extraction parameters
- Results for each attack (BER, SNR, success)

### 2. `aggregated_results.json`
Aggregated statistics across all files:
- Mean, standard deviation, and median BER
- Mean and standard deviation SNR
- Success rate percentage
- Number of tests

### 3. `robustness_summary.csv`
CSV format summary for easy analysis in spreadsheet software.

### 4. `robustness_report.txt`
Human-readable report with:
- Overall success rate
- Detailed results table
- Summary statistics

### 5. `robustness_visualization.png`
Visualization with four plots:
- Mean BER by attack
- Success rate by attack
- Mean SNR by attack
- BER vs Success Rate scatter plot

## Metrics

### Bit Error Rate (BER)
- **Definition**: Ratio of incorrectly extracted bits to total bits
- **Range**: 0.0 (perfect) to 1.0 (complete failure)
- **Success Threshold**: BER < 0.1 (10% error rate)

### Signal-to-Noise Ratio (SNR)
- **Definition**: Ratio of signal power to noise power in dB
- **Range**: -∞ to +∞ dB
- **Higher is better**: Indicates less distortion

### Success Rate
- **Definition**: Percentage of successful extractions (BER < 0.1)
- **Range**: 0% to 100%
- **Higher is better**: Indicates better robustness

## Dependencies

### Required
- `numpy`: Numerical computations
- `scipy`: Signal processing
- `librosa`: Audio processing
- `soundfile`: Audio I/O
- `matplotlib`: Visualization
- `pandas`: Data analysis
- `seaborn`: Enhanced visualization
- `tqdm`: Progress bars

### Optional
- `torch`: For EnCodec encoding (if available)
- `encodec`: For EnCodec encoding (if available)
- `ffmpeg`: For AAC/MP3 encoding (if available)

## Implementation Details

### AudioAugmenter Class
Handles individual audio augmentations:
- Filter operations (bandpass, highpass, lowpass)
- Time-domain modifications (speed, echo)
- Noise addition (white, pink)
- Codec compression (AAC, MP3, EnCodec)
- Volume modifications (boost, duck)

### RobustnessEvaluator Class
Orchestrates the evaluation process:
- Manages attack parameters for training/evaluation
- Calculates metrics (BER, SNR)
- Generates reports and visualizations
- Handles error cases gracefully

### SVDSTFTRobustnessTester Class
Integrates with SVD_STFT watermarking:
- Embeds watermarks using existing SVD_STFT implementation
- Extracts watermarks from attacked audio
- Manages test parameters and results

## Customization

### Adding New Attacks
To add a new attack type:

1. Add the attack method to `AudioAugmenter` class
2. Add attack parameters to the `attack_params` dictionary in `evaluate_robustness`
3. Add the attack to the `attacks` dictionary

### Modifying Parameters
Attack parameters can be modified in the `evaluate_robustness` method by changing the values in the `attack_params` dictionaries.

### Custom Metrics
Additional metrics can be added by:
1. Implementing calculation methods in `RobustnessEvaluator`
2. Adding them to the results dictionary
3. Updating the report generation

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: AAC/MP3 encoding will fall back to original audio
2. **EnCodec not available**: EnCodec encoding will fall back to original audio
3. **Memory issues**: Use `--max_files` to limit the number of files processed
4. **Audio format issues**: Ensure audio files are in supported formats (WAV, MP3, FLAC, etc.)

### Performance Tips

1. Use `--max_files` for quick testing
2. Process files in batches for large datasets
3. Use SSD storage for better I/O performance
4. Consider using multiprocessing for large-scale evaluations

## References

This implementation follows the AudioSeal robustness evaluation methodology as described in:
- AudioSeal: A Universal Audio Watermarking System
- Section D.2: Robustness Augmentations
