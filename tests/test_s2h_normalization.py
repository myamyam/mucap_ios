import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from multimodal_caption.normalization_s2h import peak_normalize, rms_normalize, SOUND2HAP_RMS
from multimodal_caption.wav_to_ahap_preprocessing import convert_wav
from multimodal_caption.hybrid_test import PRESETS


class NormalizationTests(unittest.TestCase):
    def test_reference_formulas_and_amplitude_invariance(self):
        x = np.array([0.01, -0.2, 0.05, 0.1])
        peak, _ = peak_normalize(x)
        np.testing.assert_allclose(peak, x / .2 * 10 ** (-1/20), rtol=1e-6)
        same, _ = peak_normalize(x * .1)
        np.testing.assert_allclose(peak, same)
        normalized, info = rms_normalize(x)
        np.testing.assert_allclose(normalized, x / (np.sqrt(np.mean(x*x)) * np.sqrt(2)), rtol=1e-6)
        self.assertAlmostEqual(info['rms_after'], SOUND2HAP_RMS, places=6)
        for fn in (peak_normalize, rms_normalize):
            result, _ = fn(np.zeros(100))
            self.assertTrue(np.all(result == 0))
            with self.assertRaises(ValueError):
                fn(np.array([np.nan]))

    def test_integration_silence_dc_and_rms_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, x in [('silence', np.zeros(48000)), ('dc', np.full(48000,.1)),
                            ('tone', .02*np.sin(2*np.pi*220*np.arange(48000)/48000))]:
                source = Path(tmp)/(name+'.wav')
                sf.write(source, x, 48000, subtype='FLOAT')
                original = source.read_bytes()
                out, info = convert_wav(source)
                self.assertEqual(source.read_bytes(), original)
                data = json.loads(out.read_text())
                if name != 'tone':
                    self.assertEqual(data['Pattern'], [])
                else:
                    self.assertAlmostEqual(info['normalization']['haptic']['rms_after'], .25, places=6)
                    _, legacy = convert_wav(source, normalization='none')
                    self.assertEqual(legacy['normalization']['mode'], 'none')
                    _, exact = convert_wav(source, haptic_target_rms=SOUND2HAP_RMS)
                    self.assertAlmostEqual(exact['normalization']['haptic']['rms_after'], SOUND2HAP_RMS, places=6)
            self.assertTrue(PRESETS['ambience'].match_rms)
            self.assertEqual(PRESETS['ambience'].out_peak, .9)


if __name__ == '__main__':
    unittest.main()
