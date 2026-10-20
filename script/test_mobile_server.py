"""Send a generated 3-second test sound; never reads or sends a user recording."""
import argparse
import base64
import io
import json
import math
import struct
import time
import urllib.request
import urllib.error
import wave

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8765/analyze')
parser.add_argument('--with-outputs', action='store_true')
args = parser.parse_args()
rate = 16000
samples = []
for i in range(3 * rate):
    t = i / rate
    note_time = t % .5
    frequency = [220, 330, 440, 330, 220, 440][min(5, int(t / .5))]
    envelope = math.exp(-note_time * 12) * min(1, note_time / .005)
    samples.append(int(14000 * envelope * math.sin(2 * math.pi * frequency * t)))
stream = io.BytesIO()
with wave.open(stream, 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    wav.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
start = time.monotonic()
request = urllib.request.Request(args.url, data=stream.getvalue(),
                                 headers={'Content-Type': 'audio/wav'})
try:
    with urllib.request.urlopen(request, timeout=600) as response:
        value = json.load(response)
        assert value['version'] == 1 and value['captions']
        if args.url.endswith('/transcribe'):
            assert value['mode'] == 'transcription' and 'events' not in value
        else:
            assert 0 < len(value['events']) <= 256
        assert all(0 <= e['intensity'] <= 1 and e['time']+e['duration'] <= 15
                   for e in value.get('events', []))
        print(json.dumps({'status': response.status, 'model': value['model'],
              'captions': len(value['captions']), 'haptic_events': len(value.get('events', [])),
              'elapsed_seconds': round(time.monotonic()-start, 2)}))
    if args.with_outputs:
        payload = dict(audio=base64.b64encode(stream.getvalue()).decode(),
                       model=value['model'], captions=value['captions'])
        request = urllib.request.Request(args.url.rsplit('/', 1)[0] + '/outputs',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            enriched = json.load(response)
            assert [(c['time'], c['end'], c['text']) for c in enriched['captions']] == [
                (c['time'], c['end'], c['text']) for c in value['captions']]
            assert enriched['events']
            print(json.dumps({'outputs_status': response.status, 'transcript_preserved': True,
                              'haptic_events': len(enriched['events'])}))
except urllib.error.HTTPError as exc:
    print('HTTP', exc.code, exc.read(2048).decode(errors='replace'))
    raise SystemExit(1)
