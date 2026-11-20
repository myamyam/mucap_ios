import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from multimodal_caption.wav_to_ahap_preprocessing import convert_wav as baseline
from multimodal_caption.wav_to_ahap_2 import convert_wav, select_transients, background_envelope, compact_curves
from multimodal_caption.hybrid_test import load_audio_mono

ROOT = Path(__file__).resolve().parent


class TransientSelectionTests(unittest.TestCase):
    def test_background_suppression_keeps_events_and_bounds_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['ttock', 'kung', 'ttiroriro', 'ppippippi']:
                src = ROOT / (name+'.wav')
                new, metrics = convert_wav(src, Path(tmp)/'new.ahap')
                self.assertGreater(metrics['continuous_events'], 0)
                points = sum(len(p['ParameterCurve']['ParameterCurveControlPoints'])
                             for p in json.loads(new.read_text())['Pattern'] if 'ParameterCurve' in p)
                self.assertLessEqual(points, 192)
                if name == 'ttock':
                    self.assertEqual(metrics['transient_events'], 3)
                    self.assertLess(metrics['background_suppression']['mean_multiplier'], .8)
                if name == 'kung':
                    self.assertEqual(metrics['transient_events'], 4)

    def test_tonal_signal_and_bass_tail_protection(self):
        sr = 48000
        times = np.arange(0, 1, .01)
        tone = .1*np.sin(2*np.pi*440*np.arange(sr)/sr)
        env, info = background_envelope(tone, sr, times, [])
        np.testing.assert_allclose(env, 1)
        y, sr = load_audio_mono(str(ROOT/'kung.wav'), 48000)
        times = np.arange(0, len(y)/sr, .01)
        impacts, _, _ = select_transients(y, sr, 256)
        protected, _ = background_envelope(y, sr, times, impacts)
        unprotected, _ = background_envelope(y, sr, times, [])
        self.assertTrue(np.all(protected >= unprotected - 1e-10))
        self.assertGreater(np.max(protected - unprotected), .01)

    def test_two_point_curves_and_budget_fallback(self):
        times = np.linspace(0, 1, 300)
        curves, info = compact_curves(times, times, times, .01, [])
        self.assertEqual(info['points'], 4)
        self.assertEqual(len(curves), 2)
        values = np.random.default_rng(12).uniform(0, 1, len(times))
        _, info = compact_curves(times, values, values, .001, times.tolist())
        self.assertLessEqual(info['points'], 192)
        self.assertTrue(info['forced_points_relaxed'])

    def test_reference_is_denominator_not_normalization_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = ROOT / 'ttock.wav'
            _, first = convert_wav(src, Path(tmp)/'a.ahap', reference_rms=.25)
            _, second = convert_wav(src, Path(tmp)/'b.ahap', reference_rms=.5)
            self.assertFalse(first['normalization']['haptic_rms_normalized'])
            self.assertEqual(first['normalization']['mode'], 'peak')
            self.assertAlmostEqual(first['normalization']['audio']['peak_after'], 10**(-1/20), places=6)
            self.assertAlmostEqual(first['normalization']['haptic_rms'], second['normalization']['haptic_rms'])
            self.assertNotAlmostEqual(first['normalization']['haptic_rms'], .25, places=3)
            self.assertAlmostEqual(first['peak_intensity'], 2 * second['peak_intensity'], places=5)

    def test_variable_count_and_gain_invariance(self):
        sr = 48000
        for count in [0, 1, 2, 4]:
            rng = np.random.default_rng(42)
            y = rng.normal(0, .0001, 3*sr)
            for t in np.linspace(.4, 2.4, count):
                start = int(t*sr)
                k = np.arange(4800)
                y[start:start+len(k)] += .3*np.exp(-k/(sr*.012))*rng.normal(size=len(k))
            times, _, _ = select_transients(y, sr, 256)
            quiet_times, _, _ = select_transients(y*.01, sr, 256)
            self.assertEqual(len(times), count)
            np.testing.assert_allclose(times, quiet_times)

    def test_silence_creates_no_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)/'silence.wav'
            sf.write(src, np.zeros(48000), 48000)
            out, _ = convert_wav(src)
            self.assertEqual(json.loads(out.read_text())['Pattern'], [])


if __name__ == '__main__':
    unittest.main()
