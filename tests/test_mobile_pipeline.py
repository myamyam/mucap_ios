import io
import base64
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import types
import wave
import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from multimodal_caption.mobile_pipeline import Handler, ANALYSIS_LOCK

from multimodal_caption.mobile_pipeline import validate_wav, wav_haptic_events


def wav_data(seconds=1, silent=False):
    rate = 8000
    samples = [0 if silent else int(12000 * math.sin(2*math.pi*100*i/rate))
               for i in range(round(seconds*rate))]
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack('<' + 'h'*len(samples), *samples))
    return stream.getvalue()


class MobilePipelineTests(unittest.TestCase):
    def test_transcription_skips_presentation_stage(self):
        from multimodal_caption.mobile_pipeline import analyze
        caption = MagicMock()
        caption.AudioFeatureExtraction.return_value.extract_features.return_value = {}
        caption.parse_caption.return_value = [(0.0, '둥')]
        package = types.ModuleType('mucap')
        package.generate_caption = caption
        with patch.dict('sys.modules', {'mucap': package}), \
             patch('multimodal_caption.mobile_pipeline.configure_environment'), \
             patch('multimodal_caption.mobile_pipeline.build_outputs') as outputs:
            result = analyze(wav_data(), transcription_only=True)
            self.assertEqual(result['mode'], 'transcription')
            self.assertEqual(result['captions'], [{'time': 0.0, 'end': 1.0, 'text': '둥'}])
            self.assertNotIn('events', result)
            outputs.assert_not_called()
            caption.generate_haptic_output.assert_not_called()

    def test_enrich_keeps_transcript_without_llm(self):
        from multimodal_caption.mobile_pipeline import decode_outputs_request, enrich
        payload = dict(audio=base64.b64encode(wav_data()).decode(), model='test',
                       captions=[dict(time=0, end=1, text='둥')])
        parsed = decode_outputs_request(json.dumps(payload).encode())
        caption = MagicMock()
        package = types.ModuleType('mucap')
        package.generate_caption = caption
        with patch.dict('sys.modules', {'mucap': package}), \
             patch('multimodal_caption.mobile_pipeline.configure_environment'), \
             patch('multimodal_caption.mobile_pipeline.build_outputs', return_value=(payload['captions'], [])) as outputs:
            result = enrich(parsed)
            self.assertEqual(result['captions'], payload['captions'])
            outputs.assert_called_once()
            caption.generate_valid_caption.assert_not_called()
            caption.llm_api.assert_not_called()
        payload['captions'][0]['end'] = 2
        with self.assertRaises(ValueError):
            decode_outputs_request(json.dumps(payload).encode())

    def events(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'haptic.wav'
            path.write_bytes(data)
            return wav_haptic_events(path)

    def test_health_responds_while_analysis_busy(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}'
        try:
            with ANALYSIS_LOCK:
                with urllib.request.urlopen(url + '/health', timeout=2) as response:
                    self.assertEqual(json.load(response)['status'], 'ok')
                request = urllib.request.Request(url + '/analyze', data=wav_data())
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request, timeout=2)
                self.assertEqual(error.exception.code, 503)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_empty_recording_reports_zero_frames(self):
        with self.assertRaisesRegex(ValueError, "frames=0"):
            validate_wav(wav_data(0))

    def test_duration_and_truncated_input(self):
        self.assertEqual(validate_wav(wav_data()), 1)
        for data in [b'invalid', wav_data()[:-20], wav_data(15.1)]:
            with self.assertRaises(ValueError):
                validate_wav(data)

    def test_silence_has_no_vibration(self):
        self.assertEqual(self.events(wav_data(silent=True)), [])

    def test_events_fit_phone_limits(self):
        events = self.events(wav_data(15))
        self.assertGreater(len(events), 1)
        self.assertLessEqual(len(events), 256)
        for event in events:
            self.assertLessEqual(event['time'] + event['duration'], 15)
            for key in ['intensity', 'sharpness']:
                self.assertTrue(0 <= event[key] <= 1)

    def test_quiet_signal_is_not_amplified_to_full_strength(self):
        data = bytearray(wav_data())
        # PCM data begins at byte 44 for this generated WAV.
        for i in range(44, len(data), 2):
            value = struct.unpack_from('<h', data, i)[0]
            struct.pack_into('<h', data, i, round(value * .001))
        self.assertLess(max(e['intensity'] for e in self.events(bytes(data))), .01)


if __name__ == '__main__':
    unittest.main()
