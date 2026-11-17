import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from multimodal_caption.wav_to_ahap_preprocessing import preprocess_audio, convert_wav


class PreprocessingTests(unittest.TestCase):
    def test_quiet_gain_preserves_waveform_and_measures_whole_file(self):
        sr = 48000
        wave = .02 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
        output, info = preprocess_audio(wave, sr)
        self.assertAlmostEqual(info['applied_boost_db'], 6)
        np.testing.assert_allclose(output, wave * info['gain_multiplier'], atol=1e-8)
        _, padded = preprocess_audio(np.r_[np.zeros(sr), wave, np.zeros(sr)], sr)
        self.assertAlmostEqual(padded['rms_before'], info['rms_before'] / np.sqrt(3))
        self.assertAlmostEqual(padded['dbfs_before'], info['dbfs_before'] - 10*np.log10(3))

    def test_loud_silence_and_peak_ceiling(self):
        sr = 48000
        for wave in [np.zeros(sr),
                     .3 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)]:
            output, info = preprocess_audio(wave, sr)
            self.assertEqual(info['gain_multiplier'], 1)
            np.testing.assert_allclose(output, wave, atol=1e-7)
        wave = .01 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
        wave[2000] = .8
        output, info = preprocess_audio(wave, sr)
        self.assertTrue(info['peak_limited'])
        self.assertLessEqual(float(np.max(np.abs(output))), .950001)

    def test_conversion_does_not_modify_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'quiet.wav'
            sr = 48000
            sf.write(source, .02 * np.sin(2*np.pi*200*np.arange(sr)/sr), sr)
            before = source.read_bytes()
            out, info = convert_wav(source)
            self.assertEqual(before, source.read_bytes())
            self.assertEqual(info['reference_rms'], .25)
            self.assertGreater(info['preprocessing']['applied_boost_db'], 0)
            self.assertEqual(json.loads(out.read_text())['Version'], 1.0)


if __name__ == '__main__':
    unittest.main()
