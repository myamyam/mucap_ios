"""
MuCap adapter for Flutter/Core Haptics. Run from the repository root.

python -m multimodal_caption.mobile_pipeline --port 8765
"""
from __future__ import annotations

import argparse
import array
import base64
import binascii
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time as clock_time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 6 * 1024 * 1024
ANALYSIS_LOCK = threading.Lock()


def configure_environment():
    from dotenv import load_dotenv
    load_dotenv(ROOT / "mucap" / ".env", override=False)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "mucap/.matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Set OPENAI_API_KEY in mucap/.env before starting the server")


def validate_wav(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            if wav.getsampwidth() != 2 or wav.getnchannels() not in (1, 2):
                raise ValueError("16-bit mono/stereo PCM WAV required")
            duration = frames / rate
            if not 0 < duration <= 15:
                detail = (f"duration={duration:.6f}s, frames={frames}, "
                          f"sample_rate={rate}Hz, channels={wav.getnchannels()}, "
                          f"file_bytes={len(data)}")
                if frames == 0:
                    raise ValueError(f"녹음 파일에 오디오 프레임이 없습니다. ({detail})")
                raise ValueError(f"Audio must be between 0 and 15 seconds ({detail})")
            if len(wav.readframes(frames)) != frames * wav.getnchannels() * 2:
                raise ValueError("Truncated WAV")
            return duration
    except (wave.Error, EOFError, ZeroDivisionError) as exc:
        raise ValueError("Invalid PCM WAV") from exc


def wav_haptic_events(path: Path) -> list[dict]:
    """Approximate the rendered WAV envelope, not its carrier frequencies.

    80ms bins yield at most 188 continuous events for a 15-second clip.
    Silent bins stay silent. Sharpness follows local envelope rise.
    """
    data = path.read_bytes()
    validate_wav(data)
    
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        samples = array.array("h", wav.readframes(wav.getnframes()))
    
    if sys.byteorder != "little":
        samples.byteswap()
    
    step = max(1, round(rate * .08)) * channels
    levels = []
    
    for start in range(0, len(samples), step):
        chunk = samples[start:start + step]
        rms = math.sqrt(sum((v / 32768) ** 2 for v in chunk) / len(chunk))
        levels.append((start / channels / rate, len(chunk) / channels / rate, rms))
    
    peak = max((v for _, _, v in levels), default=0)
    events = []
    previous = 0.0
    
    for time, duration, rms in levels:
        if rms > max(1e-5, peak * .04):
            intensity = min(1.0, rms / max(.08, peak))
            events.append(dict(time=time, duration=min(duration, 15-time),
                               intensity=intensity,
                               sharpness=min(1.0, .2 + max(0, rms-previous) / max(.08, peak))))
        previous = rms
    
    return events


def caption_timeline(tokens, duration):
    return [
        dict(
            time=t, 
            end=tokens[i+1][0] if i+1 < len(tokens) else duration,
            text=text,
        )
        for i, (t, text) in enumerate(tokens)]


def build_outputs(caption, wav, directory, features, timeline):
    """Optional presentation stage, skipped entirely by /transcribe."""
    from mucap.render import audio_text_viz as visual
    
    table = visual.note_table(visual.parse_notes(features.get("notes", [])))
    if not table.empty:
        table = visual.apply_note_mapping(table, visual.CAPTION_MAPPING)
    
    mode = visual._visual_mode_from_context(str(wav), features)
    song_style = visual.song_style(features)
    styled = []
   
    for cue in timeline:
        row = visual._row_at_time(cue["time"], table) if not table.empty else None
        if row is not None:
            style = visual.timeline_token_style(cue["text"], row, cue["time"], cue["end"], table, mode)
        else:
            # No reliable pitch: display readable neutral text rather than invented pink.
            style = dict(color="hsl(0 0% 35%)", font_size=22.1, offset_y=0,
                         characters=[dict(text=ch, color="hsl(0 0% 35%)", opacity=1, offset_y=0)
                                     for ch in cue["text"]])
        
        style['font_weight'] = int(song_style.get('font-weight', '400'))
        style['font_family'] = song_style.get('font-family', '')
        style['letter_spacing_em'] = float(song_style.get('letter-spacing', '0em').removesuffix('em'))
        style['has_pitch'] = row is not None
        styled.append(dict(cue, pitch=visual._pitch_norm(row, table) if row is not None else .5,
                           size=float(row["size_px"]) if row is not None else 26.0,
                           visual=style))
    
    haptic = caption.generate_haptic_output(str(wav), directory, features)
    return styled, wav_haptic_events(haptic)


def analyze(data: bytes, *, transcription_only=False) -> dict:
    configure_environment()
    started = clock_time.monotonic()
    duration = validate_wav(data)
    
    # The nested MuCap checkout is the import root used by its original CLI.
    if str(ROOT / "mucap") not in sys.path:
        sys.path.insert(0, str(ROOT / "mucap"))
    from mucap import generate_caption as caption

    with tempfile.TemporaryDirectory(prefix="mucap-mobile-") as directory:
        wav = Path(directory) / "recording.wav"
        wav.write_bytes(data)
        features = caption.AudioFeatureExtraction().extract_features(str(wav))
        features["envelop"] = features.get("envelope")
        
        guide = (ROOT / "mucap/mucap/prompt/design_guide_env.txt").read_text(encoding="utf-8")
        
        messages = caption.build_prompt_rhythm(
            caption.encode_audio(caption.wav_to_mp3(str(wav))), guide, features,
            use_rag=os.environ.get("MUCAP_USE_RAG") == "1", wav_path=str(wav))
        
        model = os.environ.get("MUCAP_MODEL", "gpt-audio-1.5")
        
        text = caption.generate_valid_caption(
            model, messages, str(wav),
            clear_attack_mode=caption.is_clear_attack_behavior(str(wav), features),
            percussive_generic_mode=caption.is_percussive_generic_behavior(str(wav), features))
        
        tokens = sorted((t, s) for t, s in caption.parse_caption(text) if 0 <= t < duration)
        
        if not tokens:
            raise RuntimeError("MuCap returned no timed captions")
        
        timeline = caption_timeline(tokens, duration)
        result = dict(version=1, mode="transcription", duration=duration, model=model,
                      captions=timeline)
        
        if not transcription_only:
            styled, events = build_outputs(caption, wav, directory, features, timeline)
            result.update(mode="multimodal", captions=styled, events=events)
        
        print(f"Analysis completed in {clock_time.monotonic()-started:.2f}s", flush=True)
        return result


def transcribe(data: bytes) -> dict:
    return analyze(data, transcription_only=True)


def decode_outputs_request(data):
    """Validate a retained transcript and its audio before optional rendering."""
    try:
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        audio = base64.b64decode(payload["audio"], validate=True)
        
        if len(audio) > MAX_BYTES:
            raise ValueError("WAV upload must be at most 6 MB")
        
        duration = validate_wav(audio)
        cues = payload["captions"]
        model = payload["model"]
        
        if not isinstance(model, str) or len(model) > 128:
            raise ValueError("Invalid model label")
        
        if not isinstance(cues, list) or not 1 <= len(cues) <= 2048:
            raise ValueError("Invalid caption count")
        
        timeline = []
        for cue in cues:
            t, end, text = cue["time"], cue["end"], cue["text"]
            if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (t, end))
                or not 0 <= t <= end <= duration
                or (timeline and t < timeline[-1]["time"])
                or not isinstance(text, str) or not 0 < len(text) <= 2048):
                raise ValueError("Invalid caption time or text")
            timeline.append(dict(time=t, end=end, text=text))
        
        return audio, duration, model, timeline
    
    except (KeyError, TypeError, binascii.Error, UnicodeError) as exc:
        raise ValueError("Invalid outputs request") from exc


def enrich(parsed) -> dict:
    """Add visual/haptic outputs without calling the caption model again."""
    audio, duration, model, timeline = parsed
    configure_environment()
    
    if str(ROOT / "mucap") not in sys.path:
        sys.path.insert(0, str(ROOT / "mucap"))
   
    from mucap import generate_caption as caption
    
    with tempfile.TemporaryDirectory(prefix="mucap-outputs-") as directory:
        wav = Path(directory) / "recording.wav"
        wav.write_bytes(audio)
        features = caption.AudioFeatureExtraction().extract_features(str(wav))
        styled, events = build_outputs(caption, wav, directory, features, timeline)
        
        return dict(version=1, mode="multimodal", duration=duration, model=model,
                    captions=styled, events=events)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ok", "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
                             "model": os.environ.get("MUCAP_MODEL", "gpt-audio-1.5")})
        else:
            self.reply(404, {"error": "Not found"})

    def reply(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        
        try:
            self.end_headers()
            self.wfile.write(data)
        
        except (BrokenPipeError, ConnectionResetError):
            pass  # The caller may have cancelled while analysis was running.

    def do_POST(self):
        if self.path not in ("/analyze", "/transcribe", "/outputs"):
            self.reply(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            limit = MAX_BYTES * 4 // 3 + 131072 if self.path == "/outputs" else MAX_BYTES
            
            if not 0 < size <= limit:
                raise ValueError("WAV upload must be at most 6 MB")
            self.connection.settimeout(30)
            data = self.rfile.read(size)
            
            if len(data) != size:
                raise ValueError("Incomplete upload")
            parsed = decode_outputs_request(data) if self.path == "/outputs" else None
            
            if parsed is None:
                validate_wav(data)
        
        except (ValueError, OSError) as exc:
            print(f"Rejected audio request: {exc}", flush=True)
            self.reply(400, {"error": str(exc)})
            return
        
        if not ANALYSIS_LOCK.acquire(blocking=False):
            self.reply(503, {"error": "Analysis is busy; retry after the current request finishes"})
            return
        try:
            if self.path == "/outputs":
                result = enrich(parsed)
            else:
                result = transcribe(data) if self.path == "/transcribe" else analyze(data)
        
        except Exception as exc:
            print(f"Analysis failed: {type(exc).__name__}", flush=True)
            # Do not expose provider errors, local paths, or credentials to clients.
            self.reply(500, {"error": "MuCap analysis failed; check server dependencies and API configuration"})
            return
        
        finally:
            ANALYSIS_LOCK.release()
        self.reply(200, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    configure_environment()
    print(f"MuCap listening on http://{args.host}:{args.port}; key loaded (hidden)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
