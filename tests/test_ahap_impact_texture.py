import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from multimodal_caption.ahap_test import convert_wav, detect_impacts


ROOT = Path(__file__).resolve().parent


class ImpactTextureTests(unittest.TestCase):
    def test_knocks_stay_separate_and_bass_keeps_decay(self):
        summaries = {}
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('ttock', 'kung'):
                out, _ = convert_wav(ROOT / (name + '.wav'), Path(tmp) / (name + '.ahap'))
                pattern = json.loads(out.read_text())['Pattern']
                events = [p['Event'] for p in pattern if 'Event' in p]
                hits = [e for e in events if e['EventType'] == 'HapticTransient']
                tails = [e for e in events if e['EventType'] == 'HapticContinuous']
                curves = [p['ParameterCurve'] for p in pattern if 'ParameterCurve' in p
                          and p['ParameterCurve']['ParameterID'] == 'HapticIntensityControl']
                effective, sharpness = [], []
                for hit in hits:
                    params = {p['ParameterID']: p['ParameterValue'] for p in hit['EventParameters']}
                    eligible = [c for c in curves if c['Time'] <= hit['Time']]
                    multiplier = 1.0
                    if eligible:
                        curve = eligible[-1]
                        pts = curve['ParameterCurveControlPoints']
                        multiplier = np.interp(hit['Time'] - curve['Time'],
                                               [p['Time'] for p in pts],
                                               [p['ParameterValue'] for p in pts])
                    effective.append(params['HapticIntensity'] * multiplier)
                    sharpness.append(params['HapticSharpness'])
                summaries[name] = (hits, tails, effective, sharpness)
        knocks, thumps = summaries['ttock'], summaries['kung']
        self.assertEqual(len(knocks[0]), 3)
        self.assertEqual(len(knocks[1]), 0)
        self.assertEqual(len(thumps[0]), 4)
        self.assertGreater(len(thumps[1]), 0)
        self.assertGreater(min(thumps[2]), max(knocks[2]))
        self.assertLess(max(thumps[3]), min(knocks[3]))
        for tail in thumps[1]:
            preceding = [h['Time'] for h in thumps[0] if h['Time'] < tail['Time']]
            self.assertTrue(preceding)
            self.assertLess(tail['Time'] - preceding[-1], 0.1)

    def test_silence_and_noise_do_not_create_hits(self):
        sr = 48000
        noise = np.random.default_rng(42).normal(0, 0.0001, sr)
        self.assertEqual(len(detect_impacts(noise, sr, 256)[0]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / 'silence.wav'
            sf.write(wav, np.zeros(sr), sr)
            out, _ = convert_wav(wav)
            self.assertEqual(json.loads(out.read_text())['Pattern'], [])


if __name__ == '__main__':
    unittest.main()
