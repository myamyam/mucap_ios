import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from multimodal_caption.wav_to_ahap import convert_wav as original_convert
from multimodal_caption.wav_to_ahap_preprocessing import convert_wav


class PreprocessingTests(unittest.TestCase):
    def test_none_matches_original_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'quiet.wav'
            sr = 48000
            sf.write(source, .02 * np.sin(2*np.pi*200*np.arange(sr)/sr), sr)
            before = source.read_bytes()
            original, _ = original_convert(source, root / 'original.ahap')
            processed, metrics = convert_wav(source, normalization='none')
            self.assertEqual(json.loads(original.read_text()), json.loads(processed.read_text()))
            self.assertEqual(metrics['normalization']['mode'], 'none')
            out, info = convert_wav(source)
            self.assertEqual(before, source.read_bytes())
            self.assertAlmostEqual(info['normalization']['haptic']['rms_after'], .25, places=6)
            self.assertEqual(json.loads(out.read_text())['Version'], 1.0)


if __name__ == '__main__':
    unittest.main()
