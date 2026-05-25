import argparse
import asyncio
import base64
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ObjectAudio:
    object_id: str
    instrument: str
    description: str
    audio_file: Path


@dataclass(frozen=True)
class Decision:
    is_sounding: bool
    confidence: float
    reason: str
    source: str
    raw_response: str = ""


def resolve_path(base: Path, value: str | Path | None, field_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required path field: {field_name}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def load_manifest(manifest_path: Path) -> tuple[str, Path, list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    data = load_json(manifest_path)
    clip_id = str(data.get("clip_id") or manifest_path.parent.name)
    output_dir = resolve_path(
        manifest_path.parent,
        data.get("output_dir") or "separated_audio",
        "output_dir",
    )
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Manifest must contain a non-empty objects list")
    return clip_id, output_dir, objects


def default_coords_csv(output_dir: Path) -> Path:
    base = output_dir.parent if output_dir.name == "separated_audio" else output_dir
    return base / "coords" / "object_coords.csv"


def default_output_csv(output_dir: Path) -> Path:
    base = output_dir.parent if output_dir.name == "separated_audio" else output_dir
    return base / "matching" / "time_object_audio_coords_full.csv"


def default_active_csv(full_csv: Path) -> Path:
    return full_csv.with_name("time_object_audio_coords_active_only.csv")


def object_id_from_entry(entry: dict[str, Any], index: int) -> str:
    return str(entry.get("object_id") or entry.get("id") or entry.get("name") or f"object_{index:02d}")


def load_separation_audio_files(output_dir: Path) -> dict[str, Path]:
    summary_path = output_dir / "separation_manifest.json"
    audio_files: dict[str, Path] = {}
    if not summary_path.exists():
        return audio_files

    data = load_json(summary_path)
    entries = data.get("objects")
    if not isinstance(entries, list):
        return audio_files

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        object_id = entry.get("object_id")
        audio_file = entry.get("audio_file")
        if object_id and audio_file:
            audio_files[str(object_id)] = Path(audio_file).resolve()
    return audio_files


def load_object_audio_map(manifest_objects: list[dict[str, Any]], output_dir: Path) -> dict[str, ObjectAudio]:
    summary_audio_files = load_separation_audio_files(output_dir)
    object_map: dict[str, ObjectAudio] = {}
    for index, entry in enumerate(manifest_objects, start=1):
        object_id = object_id_from_entry(entry, index)
        instrument = str(
            entry.get("instrument")
            or entry.get("class")
            or entry.get("label")
            or entry.get("description")
            or object_id
        )
        description = str(entry.get("description") or instrument)
        audio_file = summary_audio_files.get(object_id) or output_dir / object_id / f"{object_id}.wav"
        object_map[object_id] = ObjectAudio(
            object_id=object_id,
            instrument=instrument,
            description=description,
            audio_file=audio_file.resolve(),
        )
    return object_map


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No coordinate rows found: {path}")
    return rows, fieldnames


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(float(str(value)))


def row_time(row: dict[str, str], frame_interval: float | None) -> float:
    value = parse_float(row.get("time_sec"))
    if value is not None:
        return value
    if frame_interval is None:
        raise ValueError("Coordinate CSV has empty time_sec values. Pass --frame-interval.")
    frame_index = parse_int(row.get("frame_index"), 1)
    return max(0.0, (frame_index - 1) * frame_interval)


def import_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install soundfile to read wav files.") from exc
    return sf


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    sf = import_soundfile()
    if not path.exists():
        raise FileNotFoundError(f"Separated audio file does not exist: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return mono, int(sample_rate)


def audio_chunk(audio: np.ndarray, sample_rate: int, start_sec: float, duration_sec: float) -> np.ndarray:
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(len(audio), int(round((start_sec + duration_sec) * sample_rate)))
    if end <= start:
        return np.zeros(0, dtype=np.float32)
    return audio[start:end].astype(np.float32, copy=False)


def rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def pcm16_base64(audio: np.ndarray, source_rate: int, target_rate: int) -> str:
    resampled = resample_linear(audio, source_rate, target_rate)
    clipped = np.clip(resampled, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def pad_audio_min_duration(audio: np.ndarray, sample_rate: int, min_seconds: float) -> np.ndarray:
    min_samples = int(math.ceil(sample_rate * min_seconds))
    if audio.size >= min_samples:
        return audio
    return np.pad(audio, (0, min_samples - audio.size), mode="constant").astype(np.float32, copy=False)


def local_decision(chunk_rms: float, threshold: float) -> Decision:
    is_sounding = chunk_rms >= threshold
    return Decision(
        is_sounding=is_sounding,
        confidence=0.65 if is_sounding else 0.7,
        reason=f"local RMS {'above' if is_sounding else 'below'} threshold {threshold:g}",
        source="local_rms",
    )


def response_prompt(object_audio: ObjectAudio, row: dict[str, str], start_sec: float, end_sec: float, chunk_rms: float) -> str:
    bbox = (
        row.get("x_min", ""),
        row.get("y_min", ""),
        row.get("x_max", ""),
        row.get("y_max", ""),
    )
    center = (row.get("center_x", ""), row.get("center_y", ""))
    return (
        "You are labeling whether a separated instrument/object audio stem is actively sounding "
        "during one short video time window. The audio you receive is already the separated stem "
        "for the single visual object below.\n"
        f"object_id: {object_audio.object_id}\n"
        f"instrument: {object_audio.instrument}\n"
        f"description: {object_audio.description}\n"
        f"time_window_sec: {start_sec:.3f}-{end_sec:.3f}\n"
        f"frame_index: {row.get('frame_index', '')}\n"
        f"bbox_xyxy: {bbox}\n"
        f"center_xy: {center}\n"
        f"audio_rms: {chunk_rms:.8f}\n"
        "Return only JSON with this exact shape: "
        "{\"is_sounding\": true, \"confidence\": 0.0, \"reason\": \"short reason\"}. "
        "Use true only when the audio segment contains audible performance sound from this stem; "
        "use false for silence, leakage, room noise, or non-playing rests."
    )


def parse_decision(text: str, fallback: Decision, on_parse_error: str) -> Decision:
    candidates = [match.group(0) for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL)]
    if not candidates:
        if on_parse_error == "raise":
            raise ValueError(f"Realtime response did not contain JSON: {text!r}")
        if on_parse_error == "false":
            return Decision(False, 0.0, "Could not parse Realtime JSON response", "parse_error", text)
        return Decision(fallback.is_sounding, fallback.confidence, fallback.reason, "local_rms_parse_fallback", text)

    data: dict[str, Any] | None = None
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break

    if data is None:
        if on_parse_error == "raise":
            raise ValueError(f"Realtime response JSON candidates were invalid: {text!r}")
        if on_parse_error == "false":
            return Decision(False, 0.0, "Could not parse Realtime JSON response", "parse_error", text)
        return Decision(fallback.is_sounding, fallback.confidence, fallback.reason, "local_rms_parse_fallback", text)

    is_sounding = bool(data.get("is_sounding", False))
    confidence = parse_float(data.get("confidence"), 0.0) or 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason") or "")
    return Decision(is_sounding, confidence, reason, "gpt_realtime", text)


async def connect_websocket(url: str, headers: dict[str, str]):
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install websockets to use the Realtime API.") from exc

    try:
        return await websockets.connect(url, additional_headers=headers, max_size=None)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, max_size=None)


def text_from_done_event(event: dict[str, Any]) -> str:
    chunks: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"text", "transcript"} and isinstance(child, str):
                    chunks.append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(event.get("response", event))
    return "".join(chunks)


class RealtimeAudioJudge:
    def __init__(
        self,
        model: str,
        api_key: str,
        sample_rate: int,
        response_timeout: float,
        url: str | None = None,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.response_timeout = response_timeout
        self.url = url or f"wss://api.openai.com/v1/realtime?model={quote(model)}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
        }
        self.ws = None

    async def __aenter__(self) -> "RealtimeAudioJudge":
        self.ws = await connect_websocket(self.url, self.headers)
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "output_modalities": ["text"],
                    "instructions": (
                        "You judge short separated instrument audio clips. "
                        "Respond in English with compact JSON only. "
                        "Never include prose, markdown, or repeated text."
                    ),
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.sample_rate,
                            },
                            "turn_detection": None,
                        }
                    },
                },
            }
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.ws is not None:
            await self.ws.close()

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps(payload))

    async def judge(
        self,
        chunk: np.ndarray,
        source_rate: int,
        prompt: str,
        fallback: Decision,
        on_parse_error: str,
    ) -> Decision:
        chunk = pad_audio_min_duration(chunk, source_rate, 0.1)
        audio_payload = pcm16_base64(chunk, source_rate, self.sample_rate)
        await self._send(
            {
                "type": "response.create",
                "response": {
                    "conversation": "none",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                                {
                                    "type": "input_audio",
                                    "audio": audio_payload,
                                },
                            ],
                        }
                    ],
                    "output_modalities": ["text"],
                    "instructions": (
                        "Call emit_decision with the best judgment for this isolated audio segment. "
                        "Use false when uncertain or when the segment sounds like silence/leakage."
                    ),
                    "tools": [
                        {
                            "type": "function",
                            "name": "emit_decision",
                            "description": "Record whether the separated object audio is actively sounding.",
                            "parameters": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "is_sounding": {
                                        "type": "boolean",
                                        "description": "True only when the separated stem is audibly playing in this time window.",
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Short English reason.",
                                    },
                                },
                                "required": ["is_sounding", "confidence", "reason"],
                            },
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "name": "emit_decision",
                    },
                    "max_output_tokens": 200,
                },
            }
        )

        delta_parts: list[str] = []
        tool_arg_parts: list[str] = []
        final_tool_arguments = ""
        final_text = ""
        deadline = time.monotonic() + self.response_timeout
        assert self.ws is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for Realtime response")
            raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            event = json.loads(raw)
            event_type = str(event.get("type", ""))

            if event_type == "error":
                raise RuntimeError(json.dumps(event, ensure_ascii=False))
            if event_type == "response.function_call_arguments.delta" and isinstance(event.get("delta"), str):
                tool_arg_parts.append(event["delta"])
            if event_type == "response.function_call_arguments.done" and isinstance(event.get("arguments"), str):
                final_tool_arguments = event["arguments"]
            if event_type.endswith(".delta") and isinstance(event.get("delta"), str):
                delta_parts.append(event["delta"])
            if event_type == "response.output_text.done" and isinstance(event.get("text"), str):
                final_text = event["text"]
            if event_type == "response.output_item.done":
                final_text = text_from_done_event(event) or final_text
            if event_type == "response.done":
                final_text = text_from_done_event(event) or final_text
                break

        text = (final_tool_arguments or "".join(tool_arg_parts) or "".join(delta_parts) or final_text).strip()
        return parse_decision(text, fallback, on_parse_error)


def output_fields(input_fields: list[str]) -> list[str]:
    extras = [
        "audio_file",
        "audio_start_sec",
        "audio_end_sec",
        "audio_rms",
        "decision_source",
        "is_sounding",
        "confidence",
        "reason",
        "raw_realtime_response",
    ]
    return input_fields + [field for field in extras if field not in input_fields]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def write_outputs(rows: list[dict[str, str]], fieldnames: list[str], output_csv: Path, active_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    active_rows = [row for row in rows if str(row.get("is_sounding", "")).lower() == "true"]
    with active_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(active_rows)


async def run(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    clip_id, output_dir, manifest_objects = load_manifest(manifest_path)
    object_audio_map = load_object_audio_map(manifest_objects, output_dir)

    coords_csv = args.coords_csv or default_coords_csv(output_dir)
    output_csv = args.output_csv or default_output_csv(output_dir)
    active_csv = args.active_csv or default_active_csv(output_csv)

    coord_rows, coord_fields = read_csv_rows(coords_csv)
    if args.start_row > 0 or args.limit_rows:
        end_row = None if args.limit_rows is None else args.start_row + args.limit_rows
        coord_rows = coord_rows[args.start_row:end_row]

    audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
    work_items: list[tuple[dict[str, str], ObjectAudio, np.ndarray, int, float, float, float]] = []
    for row in coord_rows:
        object_id = str(row.get("object_id") or "").strip()
        if not object_id:
            raise ValueError("Coordinate CSV must include object_id for manifest mode.")
        if object_id not in object_audio_map:
            raise KeyError(f"object_id in coords CSV is not present in manifest: {object_id}")
        object_audio = object_audio_map[object_id]
        if object_audio.audio_file not in audio_cache:
            audio_cache[object_audio.audio_file] = load_audio(object_audio.audio_file)
        audio, sample_rate = audio_cache[object_audio.audio_file]
        start_sec = row_time(row, args.frame_interval)
        end_sec = start_sec + args.chunk_seconds
        chunk = audio_chunk(audio, sample_rate, start_sec, args.chunk_seconds)
        chunk_rms = rms(chunk)
        work_items.append((row, object_audio, chunk, sample_rate, start_sec, end_sec, chunk_rms))

    realtime_candidates = [
        item
        for item in work_items
        if not (
            args.skip_realtime_below_threshold
            and item[6] < args.rms_threshold
        )
    ]
    if args.max_calls is not None:
        realtime_candidates = realtime_candidates[: args.max_calls]

    print(f"Clip: {clip_id}")
    print(f"Coords CSV: {coords_csv}")
    print(f"Separated audio dir: {output_dir}")
    print(f"Rows to write: {len(work_items)}")
    print(f"Realtime calls planned: {len(realtime_candidates)}")
    print(f"Model: {args.model}")

    if args.dry_run:
        print("Dry run only; no API call and no CSV written.")
        return

    realtime_keys = {
        (id(row), object_audio.object_id, start_sec)
        for row, object_audio, _chunk, _sr, start_sec, _end_sec, _rms in realtime_candidates
    }
    api_key = os.environ.get(args.api_key_env)
    if args.local_only:
        api_key = None
    elif not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before using Realtime, or pass --local-only.")

    processed_rows: list[dict[str, str]] = []
    fieldnames = output_fields(coord_fields)

    async def process_with_judge(judge: RealtimeAudioJudge | None) -> None:
        calls_used = 0
        for row, object_audio, chunk, sample_rate, start_sec, end_sec, chunk_rms in work_items:
            fallback = local_decision(chunk_rms, args.rms_threshold)
            key = (id(row), object_audio.object_id, start_sec)
            use_realtime = (
                judge is not None
                and key in realtime_keys
                and chunk.size > 0
                and not math.isnan(chunk_rms)
            )
            if use_realtime:
                prompt = response_prompt(object_audio, row, start_sec, end_sec, chunk_rms)
                decision = await judge.judge(chunk, sample_rate, prompt, fallback, args.on_parse_error)
                calls_used += 1
                if args.sleep > 0:
                    await asyncio.sleep(args.sleep)
            else:
                decision = fallback

            output_row = dict(row)
            output_row.update(
                {
                    "audio_file": str(object_audio.audio_file),
                    "audio_start_sec": f"{start_sec:.3f}",
                    "audio_end_sec": f"{end_sec:.3f}",
                    "audio_rms": f"{chunk_rms:.8f}",
                    "decision_source": decision.source,
                    "is_sounding": bool_text(decision.is_sounding),
                    "confidence": f"{decision.confidence:.3f}",
                    "reason": decision.reason,
                    "raw_realtime_response": decision.raw_response,
                }
            )
            processed_rows.append(output_row)
        if judge is not None:
            print(f"Realtime calls completed: {calls_used}")

    if api_key and not args.local_only:
        async with RealtimeAudioJudge(
            model=args.model,
            api_key=api_key,
            sample_rate=args.realtime_sample_rate,
            response_timeout=args.response_timeout,
            url=args.url,
        ) as judge:
            await process_with_judge(judge)
    else:
        await process_with_judge(None)

    write_outputs(processed_rows, fieldnames, output_csv, active_csv)
    print(f"Saved full CSV: {output_csv}")
    print(f"Saved active-only CSV: {active_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use separated object wav files and per-frame coordinates to label which object "
            "is sounding at each timestamp. Realtime model is configurable with --model."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest generated by create_manifest.py.")
    parser.add_argument("--coords-csv", type=Path, default=None, help="CSV generated by frame_mask_coords.py.")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--active-csv", type=Path, default=None)
    parser.add_argument("--model", default="gpt-realtime", help="Realtime model name, e.g. gpt-realtime or gpt-realtime-mini.")
    parser.add_argument("--url", default=None, help="Override Realtime WebSocket URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--chunk-seconds", type=float, default=0.5, help="Audio duration to judge from each frame timestamp.")
    parser.add_argument("--frame-interval", type=float, default=None, help="Fallback when coords CSV has empty time_sec.")
    parser.add_argument("--realtime-sample-rate", type=int, default=24000, help="PCM16 sample rate sent to Realtime.")
    parser.add_argument("--rms-threshold", type=float, default=0.003, help="Local RMS threshold used for fallback/gating.")
    parser.add_argument(
        "--skip-realtime-below-threshold",
        action="store_true",
        help="Do not call Realtime for very quiet chunks; mark them with local RMS decision.",
    )
    parser.add_argument("--local-only", action="store_true", help="Write CSV using local RMS only; no Realtime API call.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned files/calls without using the API or writing CSV.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--max-calls", type=int, default=None, help="Cap Realtime calls for cost/debugging.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between Realtime calls.")
    parser.add_argument("--response-timeout", type=float, default=45.0)
    parser.add_argument(
        "--on-parse-error",
        choices=["local", "false", "raise"],
        default="local",
        help="What to do if the Realtime text is not parseable JSON.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
