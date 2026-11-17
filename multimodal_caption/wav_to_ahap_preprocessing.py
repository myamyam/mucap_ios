"""
파일 전체 gain을 최대 +6dB까지 적용해서 작은 소리 보완

Example:
  python -m multimodal_caption.wav_to_ahap_preprocessing --in tests/ttock.wav
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

if __package__:
    from .hybrid_test import PRESETS, load_audio_mono
    from .wav_to_ahap import convert_wav as render_ahap
else:
    from hybrid_test import PRESETS, load_audio_mono
    from wav_to_ahap import convert_wav as render_ahap


def preprocess_audio(y, sr, *, quiet_threshold_dbfs=-26.0, target_dbfs=-20.0,
                     max_boost_db=6.0, peak_limit=0.95):
    """Choose one gain using whole-recording RMS dBFS, including silent samples.

    RMS = sqrt(mean(y**2)); dBFS = 20*log10(RMS).
    Exact silence is reported as None dBFS (minus infinity) for valid JSON.
    """
    y = np.asarray(y, dtype=np.float64)
    controls = [quiet_threshold_dbfs, target_dbfs, max_boost_db, peak_limit, sr]
    
    if not all(np.isfinite(controls)) or sr <= 0:
        raise ValueError('Preprocessing controls must be finite; sr must be positive')
    if not quiet_threshold_dbfs <= target_dbfs <= 0:
        raise ValueError('Require quiet_threshold_dbfs <= target_dbfs <= 0')
    if not 0 <= max_boost_db <= 60 or not 0 < peak_limit <= 1:
        raise ValueError('Require max_boost_db in [0, 60] and peak_limit in (0, 1]')
    if y.ndim != 1 or not len(y) or not np.all(np.isfinite(y)):
        raise ValueError('Audio must be nonempty finite mono samples')
    
    rms = float(np.sqrt(np.mean(y ** 2)))
    level = float(20 * np.log10(rms)) if rms > 0 else None
    requested_db = (min(max_boost_db, target_dbfs - level)
                    if level is not None and level < quiet_threshold_dbfs else 0.0)
    peak = float(np.max(np.abs(y)))
    desired = 10 ** (requested_db / 20)
    multiplier = max(1.0, min(desired, peak_limit / max(peak, 1e-12)))
    applied_db = float(20 * np.log10(multiplier))
    return (y * multiplier).astype(np.float32), {
        'rms_before': rms, 'dbfs_before': level,
        'rms_after': rms * multiplier,
        'dbfs_after': level + applied_db if level is not None else None,
        'requested_boost_db': requested_db,
        'applied_boost_db': applied_db,
        'gain_multiplier': multiplier,
        'peak_before': peak, 'peak_after': peak * multiplier,
        'peak_limited': bool(multiplier + 1e-9 < desired),
        'quiet_threshold_dbfs': quiet_threshold_dbfs,
        'target_dbfs': target_dbfs, 'max_boost_db': max_boost_db,
        'peak_limit': peak_limit,
    }


def convert_wav(in_path, out_path=None, *, preset='ambience', reference_rms=0.25,
                quiet_threshold_dbfs=-26.0, target_dbfs=-20.0,
                max_boost_db=6.0, peak_limit=0.95, **haptic_options):
    source = Path(in_path)
    if source.suffix.lower() != '.wav' or not source.is_file():
        raise ValueError(f'Input must be an existing WAV: {source}')
    destination = Path(out_path) if out_path else source.with_name(source.stem + '_preprocessed.ahap')
    if destination.suffix.lower() != '.ahap' or destination.resolve() == source.resolve():
        raise ValueError('Output must be a separate .ahap file')
    if preset not in PRESETS:
        raise ValueError(f'Unknown preset: {preset}')
    y, sr = load_audio_mono(str(source), PRESETS[preset].sr)
    processed, report = preprocess_audio(
        y, sr, quiet_threshold_dbfs=quiet_threshold_dbfs, target_dbfs=target_dbfs,
        max_boost_db=max_boost_db, peak_limit=peak_limit)
    # FLOAT preserves quiet samples without PCM16 requantization. Both hybrid
    # and onset analysis in the original converter receive the same signal.
    with tempfile.TemporaryDirectory(prefix='ahap-preprocessing-') as directory:
        temporary_wav = Path(directory) / 'processed.wav'
        sf.write(temporary_wav, processed, sr, subtype='FLOAT')
        output, metrics = render_ahap(temporary_wav, destination, preset=preset,
                                     reference_rms=reference_rms, **haptic_options)
    return output, dict(metrics, reference_rms=reference_rms, preprocessing=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in', dest='in_path', required=True)
    parser.add_argument('--out', dest='out_path')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='ambience')
    parser.add_argument('--quality', choices=['low', 'mid', 'high'], default='mid')
    for name, default in [('reference-rms', 0.25), ('quiet-threshold-dbfs', -26.0),
                          ('target-dbfs', -20.0), ('max-boost-db', 6.0),
                          ('peak-limit', 0.95), ('gain', 1.0), ('strength', 1.0),
                          ('control-hz', 100.0), ('curve-tolerance', 0.01)]:
        parser.add_argument('--' + name, type=float, default=default)
    args = parser.parse_args()
    try:
        path, metrics = convert_wav(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Written: {path}\n' + json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
