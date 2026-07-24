"""
caption_skeleton_patch.py
=========================
mucap/generate_caption.py 의 LLM 프롬프트에 '골격(Skeleton)'과 '합의도'를
주입하는 패치. (목적: LLM 캡션을 골격·합의도로 개선)

generate_caption.py 는 audio-capable LLM이 음절을 생성하므로,
우리는 음절을 직접 만들지 않고 '프롬프트에 텍스트로 정보를 더한다'.

주입 지점 (generate_caption.py 의 실제 함수):
  ① interpret_music_features() 결과에 Skeleton 텍스트 추가
       → build_skeleton_text(): notes(onset/장단/쉼표)를 LLM이 읽을 서술형으로
  ② build_prompt_rhythm() 의 RAG 컨텍스트에 합의도 요약 추가
       → 검색된 유사 주석이 "몇 음절·얼마나 일치"였는지 신뢰도 신호

notes 는 interpret_music_features 가 받는 features dict 에 없으므로,
music_features.csv 에서 file_name 으로 따로 조회한다(get_skeleton_text_for_file).

설계 일관성:
  - 음절 직접 생성(heuristic/)과 달리, 여기서는 LLM 가이드용 텍스트만 만든다.
  - 골격은 RAG DB(json2ragdb, rag_skeleton_patch)에도 들어가야 검색·생성이 맞물린다.
"""

from __future__ import annotations
import ast
import math
import os
from os.path import join, dirname, basename

import pandas as pd

from mucap.data_io import feature_row_to_skeleton, parse_notes_field
from mucap.rag_skeleton_patch import build_skeleton_text, annotation_consensus_summary


# music_features.csv 캐시 (배치 처리 시 매번 읽지 않도록)
_FEATURE_DF = None
_FEATURE_CSV_PATH = None


def _audio_key(path_or_name: str) -> str:
    fname = basename(path_or_name)
    return fname[len("fixed_"):] if fname.startswith("fixed_") else fname


def _load_feature_df(feature_csv_path: str) -> pd.DataFrame:
    global _FEATURE_DF, _FEATURE_CSV_PATH
    if _FEATURE_DF is None or _FEATURE_CSV_PATH != feature_csv_path:
        _FEATURE_DF = pd.read_csv(feature_csv_path)
        _FEATURE_CSV_PATH = feature_csv_path
    return _FEATURE_DF


def _sample_cluster_indices(size: int, max_items: int = 12) -> list[int]:
    if size <= max_items:
        return list(range(size))

    head = list(range(min(4, size)))
    tail = list(range(max(0, size - 4), size))
    remaining = max_items - len(set(head + tail))
    middle = []
    if remaining > 0:
        center = size // 2
        start = center - remaining // 2
        start = max(len(head), start)
        start = min(start, max(len(head), size - len(tail) - remaining))
        middle = list(range(start, start + remaining))

    return sorted(set(head + middle + tail))


def _format_sampled_cluster_times(cluster: list[float]) -> str:
    indices = _sample_cluster_indices(len(cluster))
    pieces = []
    previous = None
    for index in indices:
        if previous is not None and index - previous > 1:
            pieces.append("...")
        pieces.append(f"{cluster[index]:.2f}")
        previous = index
    return ", ".join(pieces)


def _stored_audio_onsets(feature_row: dict | None) -> tuple[list[float], float | None]:
    """Read onset timestamps cached during the initial feature extraction."""
    if not feature_row:
        return [], None

    raw = feature_row.get("audio_onsets", [])
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = []
    if not isinstance(raw, (list, tuple)):
        raw = []

    onsets = []
    for value in raw:
        try:
            onset = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(onset) and onset >= 0:
            onsets.append(onset)

    try:
        duration = float(feature_row.get("duration"))
    except (TypeError, ValueError):
        duration = None
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        duration = None

    return sorted(onsets), duration


def _parse_top5_for_hint(value) -> list[tuple[str, float]]:
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    else:
        return []

    out = []
    for index, item in enumerate(parsed):
        try:
            if isinstance(item, dict):
                name = item.get("instrument") or item.get("name") or item.get("label")
                score = item.get("score") or item.get("probability") or item.get("confidence")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name, score = item[0], item[1]
            else:
                name, score = item, 1.0 / (index + 1)
            out.append((str(name).lower(), float(score)))
        except (TypeError, ValueError):
            continue
    return out


def _is_confident_string_context(row: dict) -> bool:
    top5 = _parse_top5_for_hint(row.get("top5_instruments", ""))
    if not top5:
        return False

    bowed_strings = {"violin", "viola", "cello", "doublebass"}
    piano_names = {"piano", "keyboard"}
    voice_names = {"voice", "vocal", "vocals", "singing"}
    guitar_names = {"guitar", "electric guitar", "acoustic guitar"}
    top_name, top_score = top5[0]
    piano_score = max((score for name, score in top5 if name in piano_names), default=0.0)
    string_score = max((score for name, score in top5 if name in bowed_strings), default=0.0)
    voice_score = max((score for name, score in top5 if name in voice_names), default=0.0)
    guitar_score = max((score for name, score in top5 if name in guitar_names), default=0.0)
    string_count = sum(1 for name, _ in top5 if name in bowed_strings)
    non_string_best = max(piano_score, voice_score, guitar_score)
    string_sum = sum(score for name, score in top5 if name in bowed_strings)

    notes = parse_notes_field(row.get("notes", ""))
    pitch_steps = []
    parsed_pitches = []
    for note in notes:
        try:
            pitch = float(note.get("pitch"))
        except (TypeError, ValueError):
            continue
        if pitch > 0:
            parsed_pitches.append(pitch)
    for prev, cur in zip(parsed_pitches, parsed_pitches[1:]):
        pitch_steps.append(abs(12.0 * math.log2(cur / prev)))
    big_pitch_jump_ratio = (
        sum(1 for step in pitch_steps if step >= 8.0) / len(pitch_steps)
        if pitch_steps else 0.0
    )
    if piano_score >= 0.60 and big_pitch_jump_ratio >= 0.50:
        return False

    if piano_score >= 0.55 and piano_score >= string_score * 1.5:
        return False
    if (
        string_count == 1
        and voice_score >= string_score * 0.90
        and guitar_score >= 0.60
    ):
        return False
    if (
        string_count >= 2
        and string_score >= 0.50
        and (
            string_sum >= non_string_best * 1.10
            or string_score >= non_string_best - 0.05
        )
    ):
        return True
    return (
        top_name in bowed_strings
        and string_score >= 0.70
        and string_score >= non_string_best + 0.15
    )


def _is_weak_string_texture_context(row: dict | None, behavior: dict | None) -> bool:
    """Promote ambiguous sustained mid/high-register textures toward strings.

    This is intentionally weaker than _is_confident_string_context and only
    runs after audio-only behavior has already identified a sustained texture.
    It avoids using filenames and keeps bright high-register wind-like cases
    and low-register piano-like cases out of the string branch.
    """
    if not row or not behavior or behavior.get("mode") != "sustained-texture":
        return False

    top5 = _parse_top5_for_hint(row.get("top5_instruments", ""))
    bowed_strings = {"violin", "viola", "cello", "doublebass", "strings"}
    string_score = max((score for name, score in top5 if name in bowed_strings), default=0.0)
    if string_score < 0.50:
        return False

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    pitch = safe_float(row.get("pitch"))
    onset_strength = safe_float(row.get("onset_strength"))
    note_count = safe_float(behavior.get("note_count"))
    gap_ratio = safe_float(behavior.get("gap_ratio"))
    pitch_change_ratio = safe_float(behavior.get("pitch_change_ratio"))

    return (
        175.0 <= pitch <= 280.0
        and onset_strength <= 1.05
        and note_count <= 45
        and gap_ratio <= 0.08
        and pitch_change_ratio >= 0.70
    )


def _connected_texture_style(row: dict | None,
                             behavior: dict | None,
                             string_mode: bool = False) -> str:
    if behavior is None or behavior.get("mode") != "sustained-texture":
        return "none"
    if string_mode:
        return "string-like"

    row = row or {}
    if _is_weak_string_texture_context(row, behavior):
        return "string-like"

    top5 = _parse_top5_for_hint(row.get("top5_instruments", ""))
    wind_names = {
        "flute", "recorder", "clarinet", "oboe", "bassoon", "saxophone",
        "sax", "trumpet", "trombone", "horn", "french horn", "tuba",
    }
    wind_score = max((score for name, score in top5 if name in wind_names), default=0.0)

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    pitch = safe_float(row.get("pitch"))
    onset_strength = safe_float(row.get("onset_strength"))
    gap_ratio = safe_float(behavior.get("gap_ratio"))
    pitch_change_ratio = safe_float(behavior.get("pitch_change_ratio"))

    high_pitch = pitch >= 220.0 or (0.6 <= pitch <= 2.0)
    connected_motion = gap_ratio <= 0.12 and pitch_change_ratio >= 0.45

    if wind_score >= 0.45:
        return "wind-like"
    if high_pitch and connected_motion and onset_strength <= 1.4:
        return "wind-like"
    return "generic-sustain"


def build_rapid_onset_hint(feature_row,
                           max_gap: float = 0.55,
                           min_cluster_size: int = 4) -> str | None:
    row = dict(feature_row)
    notes = parse_notes_field(row.get("notes", ""))
    if not notes:
        return None

    onsets = sorted(float(n.get("onset_time", 0.0)) for n in notes)
    clusters = []
    cur = [onsets[0]]
    for onset in onsets[1:]:
        if onset - cur[-1] <= max_gap:
            cur.append(onset)
        else:
            if len(cur) >= min_cluster_size:
                clusters.append(cur)
            cur = [onset]
    if len(cur) >= min_cluster_size:
        clusters.append(cur)

    if not clusters:
        return None

    cluster = max(clusters, key=len)
    start, end = cluster[0], cluster[-1]
    shown = _format_sampled_cluster_times(cluster)
    return (
        f"Rapid onset hint: {start:.2f}-{end:.2f}s has {len(cluster)} close onsets "
        f"at sampled timestamps [{shown}]. Ellipses inside timestamp lists mean "
        "additional detected onsets continue between the shown timestamps; do not "
        "treat omitted timestamps as silence or leave long gaps while audible notes "
        "continue. Use close captions only when these are clear separate attacks. "
        "If the cluster is a sustained texture, vibrato, resonance, or connected "
        "motion, group it into phrase-level Hangul captions instead of captioning "
        "each detected point. If pitch, duration, loudness, attack, resonance, or "
        "rests change, use a small related set of Hangul variants without adding "
        "unsupported variety."
    )


def build_caption_anchor_hint(feature_row,
                              max_items: int = 18,
                              min_pitch_step: float = 1.5,
                              min_rest_gap: float = 0.35,
                              min_anchor_gap: float = 0.0,
                              merge_window: float = 0.0,
                              compact_sustained: bool = False,
                              expanded_string_sustained: bool = False,
                              expanded_wind_sustained: bool = False,
                              expanded_generic_sustained: bool = False) -> str | None:
    compact_long_gap = 0.0
    if compact_sustained:
        min_pitch_step = max(min_pitch_step, 1.5)
        if (
            expanded_string_sustained
            or expanded_wind_sustained
            or expanded_generic_sustained
        ):
            max_items = max(max_items, 24)
            min_anchor_gap = max(min_anchor_gap, 0.25)
            merge_window = max(merge_window, 0.18)
        else:
            max_items = min(max_items, 16)
            min_anchor_gap = max(min_anchor_gap, 0.35)
            merge_window = max(merge_window, 0.25)
        compact_long_gap = 1.00

    row = dict(feature_row)
    notes = parse_notes_field(row.get("notes", ""))
    if not notes:
        return None

    parsed = []
    for note in notes:
        onset = _safe_float(note.get("onset_time"), None)
        duration = _safe_float(note.get("duration"), None)
        pitch = _safe_float(note.get("pitch"), None)
        if onset is None or duration is None or duration <= 0:
            continue
        parsed.append({"onset": onset, "duration": duration, "pitch": pitch})
    if not parsed:
        return None

    parsed = sorted(parsed, key=lambda item: item["onset"])
    anchors = [(parsed[0]["onset"], "start", float("inf"))]
    for prev, cur in zip(parsed, parsed[1:]):
        prev_end = prev["onset"] + prev["duration"]
        gap = cur["onset"] - prev_end
        step = 0.0
        if prev["pitch"] and cur["pitch"] and prev["pitch"] > 0 and cur["pitch"] > 0:
            step = 12.0 * math.log2(cur["pitch"] / prev["pitch"])

        reasons = []
        if abs(step) >= min_pitch_step:
            direction = "up" if step > 0 else "down"
            reasons.append(f"pitch {direction} {abs(step):.1f}st")
        if gap >= min_rest_gap:
            reasons.append(f"after rest {gap:.2f}s")
        if reasons:
            anchors.append((cur["onset"], ", ".join(reasons), max(abs(step), gap)))

    if len(anchors) <= 1:
        return None

    if merge_window > 0:
        merged = []
        group = [anchors[0]]
        for anchor in anchors[1:]:
            if anchor[0] - group[-1][0] <= merge_window:
                group.append(anchor)
            else:
                start = group[0]
                strongest = max(group, key=lambda item: item[2])
                reason = start[1]
                if strongest is not start and strongest[1] != "start":
                    reason = f"phrase start; strongest {strongest[1]}"
                merged.append((start[0], reason, strongest[2]))
                group = [anchor]
        start = group[0]
        strongest = max(group, key=lambda item: item[2])
        reason = start[1]
        if strongest is not start and strongest[1] != "start":
            reason = f"phrase start; strongest {strongest[1]}"
        merged.append((start[0], reason, strongest[2]))
        anchors = merged

    if min_anchor_gap > 0:
        gap_candidates = anchors
        spaced = []
        for anchor in anchors:
            if not spaced or anchor[0] - spaced[-1][0] >= min_anchor_gap:
                spaced.append(anchor)
            elif anchor[2] > spaced[-1][2] and spaced[-1][1] != "start":
                spaced[-1] = anchor
        anchors = spaced
        if compact_long_gap > 0:
            preserved = []
            for left, right in zip(anchors, anchors[1:]):
                preserved.append(left)
                if right[0] - left[0] <= compact_long_gap:
                    continue
                between = [
                    anchor for anchor in gap_candidates
                    if left[0] + min_anchor_gap <= anchor[0] <= right[0] - min_anchor_gap
                ]
                if between:
                    preserved.append(max(between, key=lambda item: item[2]))
            preserved.append(anchors[-1])
            anchors = sorted({anchor[0]: anchor for anchor in preserved}.values())

    if len(anchors) <= 1:
        return None

    indices = _sample_cluster_indices(len(anchors), max_items=max_items)
    pieces = []
    previous = None
    for index in indices:
        if previous is not None and index - previous > 1:
            pieces.append("...")
        t, reason, _ = anchors[index]
        pieces.append(f"{t:.2f}s ({reason})")
        previous = index

    anchor_label = (
        "compact phrase anchors" if compact_sustained else "caption anchors"
    )
    if expanded_string_sustained:
        mode_instruction = (
        "For bowed string-like textures, these are phrase-start candidates, "
        "not a beat grid. During active sustained phrases, use about 80-100% of "
        "the meaningful listed anchors. Skip or merge anchors only when the sound "
        "is genuinely stable or the detected changes are merely repeated vibrato "
        "or the same unchanging texture. Do not omit a clear pitch-direction "
        "change, accent, energy change, new melodic note, or rest. The listed "
        "anchors are representative boundary cues, not an exhaustive set of "
        "musical events. The audio itself has higher priority than this anchor "
        "list: if an audible meaningful change occurs at an unlisted timestamp, "
        "add a caption near that audible change even though no anchor is listed. "
        "Do not invent changes that are not audible, and do not force mechanical "
        "equal spacing. A caption may still cover multiple anchors only when the "
        "sound continues as one genuinely stable phrase. "
        )
    elif expanded_wind_sustained:
        mode_instruction = (
        "For wind-like sustained instrumental textures, these are phrase-start "
        "candidates, not a beat grid. During active sustained phrases, use about "
        "80-100% of the meaningful listed anchors. Skip or merge anchors only when "
        "the sound is genuinely stable or the detected changes are merely repeated "
        "airflow, vibrato, or the same unchanging texture. Do not omit a clear "
        "pitch-direction change, accent, energy change, new melodic note, or rest. "
        "The audio itself has higher priority than this non-exhaustive anchor list: "
        "if a meaningful audible change occurs at an unlisted timestamp, add a "
        "caption near that change. Do not invent inaudible changes or force "
        "mechanical equal spacing. "
        )
    elif expanded_generic_sustained:
        mode_instruction = (
        "For generic sustained instrumental textures, these are phrase-start "
        "candidates, not a beat grid. During active sustained phrases, use about "
        "80-100% of the meaningful listed anchors. Skip or merge anchors only when "
        "the sound is genuinely stable or the detected changes repeat the same "
        "unchanging texture. Do not omit a clear pitch-direction change, accent, "
        "energy change, new melodic note, or rest. The audio itself has higher "
        "priority than this non-exhaustive anchor list: if a meaningful audible "
        "change occurs at an unlisted timestamp, add a caption near that change. "
        "Do not invent inaudible changes or force mechanical equal spacing. "
        )
    elif compact_sustained:
        mode_instruction = (
        "For sustained or wind-like textures, these are phrase-start candidates, "
        "not a beat grid: use only a sparse subset, usually at least 0.35s apart, "
        "and prefer longer Hangul tokens over adding more timestamps. For active "
        "sustained phrases, choose about 60-80% of these anchors, not all of them; "
        "use fewer only when the sound is truly stable. Do not mirror every anchor "
        "one-to-one as a caption. The listed anchors are representative boundary "
        "cues, not an exhaustive set of musical events. There may be additional "
        "audible changes between listed anchors. A caption may cover a phrase "
        "spanning multiple anchors, and a listed anchor may be skipped if the "
        "phrase continues naturally. "
        )
    else:
        mode_instruction = (
        "Choose a subset according to the behavior mode: sustained textures should "
        "group adjacent anchors into phrase-level captions, clear attacks may use "
        "anchors more densely. "
        )
    return (
        f"Preferred {anchor_label} from detected note/pitch changes "
        f"({len(anchors)} candidates, sampled): [{', '.join(pieces)}]. "
        "Use these as phrase-boundary candidates, not direct caption timestamps. "
        + mode_instruction +
        "Do not invent mechanically even timestamps between these anchors, and do "
        "not choose timestamps merely because they form a regular interval."
    )


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values: list[float]) -> float:
    values = sorted(v for v in values if v is not None)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _caption_behavior_analysis(feature_row,
                               audio_onsets: list[float],
                               duration: float) -> dict | None:
    """Classify caption density from runtime notes and audio onset behavior only."""
    if not feature_row:
        return None

    notes = parse_notes_field(feature_row.get("notes", ""))
    if not notes:
        return None

    parsed = []
    for note in notes:
        onset = _safe_float(note.get("onset_time"), None)
        dur = _safe_float(note.get("duration"), None)
        pitch = _safe_float(note.get("pitch"), None)
        if onset is None or dur is None or dur <= 0:
            continue
        parsed.append({"onset": onset, "duration": dur, "pitch": pitch})
    if not parsed:
        return None

    parsed = sorted(parsed, key=lambda item: item["onset"])
    durations = [item["duration"] for item in parsed]
    note_count = len(parsed)
    onset_count = len(audio_onsets)
    ratio = onset_count / max(note_count, 1)
    median_duration = _median(durations)
    long_ratio = sum(1 for d in durations if d >= 0.45) / note_count
    short_ratio = sum(1 for d in durations if d <= 0.18) / note_count
    gaps = []
    for prev, cur in zip(parsed, parsed[1:]):
        gap = cur["onset"] - (prev["onset"] + prev["duration"])
        if gap > 0:
            gaps.append(gap)
    base_duration = duration if duration > 0 else max(
        item["onset"] + item["duration"] for item in parsed
    )
    gap_ratio = sum(gaps) / base_duration if base_duration > 0 else 0.0

    pitch_steps = []
    for prev, cur in zip(parsed, parsed[1:]):
        p1, p2 = prev["pitch"], cur["pitch"]
        if p1 and p2 and p1 > 0 and p2 > 0:
            pitch_steps.append(abs(12.0 * math.log2(p2 / p1)))
    pitch_change_ratio = (
        sum(1 for step in pitch_steps if step >= 1.5) / len(pitch_steps)
        if pitch_steps else 0.0
    )

    sustained_note_like = (
        (median_duration >= 0.38 or long_ratio >= 0.35)
        and gap_ratio <= 0.28
        and (ratio >= 1.35 or onset_count >= note_count + 6)
    )
    connected_pitch_texture_like = (
        gap_ratio <= 0.15
        and ratio >= 1.35
        and pitch_change_ratio >= 0.45
        and short_ratio < 0.45
    )
    sustained_like = sustained_note_like or connected_pitch_texture_like
    clear_attack_like = (
        median_duration <= 0.32
        and short_ratio >= 0.35
        and gap_ratio >= 0.04
        and ratio <= 2.25
    )

    stats = (
        f"notes={note_count}, audio_onsets={onset_count}, "
        f"onset_to_note_ratio={ratio:.2f}, median_note_duration={median_duration:.2f}s, "
        f"long_note_ratio={long_ratio:.2f}, short_note_ratio={short_ratio:.2f}, "
        f"gap_ratio={gap_ratio:.2f}, pitch_change_ratio={pitch_change_ratio:.2f}"
    )
    # Binary top-level behavior classification:
    # preserve the existing sustained/clear-attack thresholds and route only
    # the previously ambiguous fallback into sustained-texture. Sustained
    # texture is subdivided later into string-like, wind-like, or
    # generic-sustain by _connected_texture_style().
    mode = "sustained-texture"
    if sustained_like:
        mode = "sustained-texture"
    elif clear_attack_like:
        mode = "clear-attack"

    behavior = {
        "mode": mode,
        "stats": stats,
        "note_count": note_count,
        "audio_onset_count": onset_count,
        "onset_to_note_ratio": ratio,
        "median_note_duration": median_duration,
        "long_note_ratio": long_ratio,
        "short_note_ratio": short_ratio,
        "gap_ratio": gap_ratio,
        "pitch_change_ratio": pitch_change_ratio,
    }

    # Pedalled piano can look continuous in the pitch-derived notes and can
    # also be mislabeled as a bowed string by the instrument predictor. Limit
    # this correction to sustained inputs that would otherwise enter the
    # string-like branch, so wind-like and generic-sustain remain unchanged.
    attack = str(feature_row.get("attack", "")).strip().lower()
    envelope = str(feature_row.get("envelope", "")).strip().lower()
    transient_like = attack == "sharp" or envelope == "transient"
    string_candidate = bool(_is_confident_string_context(feature_row))
    if not string_candidate:
        string_candidate = _is_weak_string_texture_context(
            feature_row,
            behavior,
        )

    piano_pedal_attack_like = (
        behavior["mode"] == "sustained-texture"
        and string_candidate
        and transient_like
        and median_duration >= 0.38
        and ratio < 1.45
    )
    if piano_pedal_attack_like:
        behavior["mode"] = "clear-attack"
        behavior["piano_pedal_attack_override"] = True

    return behavior


def _caption_behavior_hint(analysis: dict | None) -> str | None:
    if not analysis:
        return None

    stats = analysis["stats"]
    mode = analysis["mode"]
    if mode == "clear-attack" and analysis.get("piano_pedal_attack_override"):
        return (
            "Caption behavior analysis from audio only: " + stats +
            ". Behavior mode: clear-attack with pedalled-piano correction. "
            "The note attacks are distinct, but pedal and resonance may connect the "
            "sound after each attack; do not describe these notes as uniformly short "
            "or fully separated. Place captions near the audible key attacks, and "
            "match each Hangul token length to the actual sustain and resonance. "
            "Possible references include short attacks such as 딩, 땡, 띵, 딴, or 단; "
            "medium resonances such as 따란, 따앙, 따단, or 다안; and longer resonances or "
            "connected figures such as 따라랑, 따라란, 따아안, 다아안, or 두웅. These are "
            "references, not a fixed "
            "vocabulary. Do not reduce the full passage to one-syllable repeats, and "
            "do not mechanically alternate a fixed pair. Change the consonant, vowel, "
            "and token length only when pitch, register, duration, loudness, chord "
            "density, attack, or resonance audibly changes. Follow the actual audio "
            "timing rather than forcing equal spacing."
        )
    if mode == "sustained-texture":
        return (
            "Caption behavior analysis from audio only: " + stats +
            ". Behavior mode: sustained-texture. Dense onset detections likely include "
            "vibrato, resonance, bow/noise texture, or connected motion. Do not turn "
            "every detected onset into a separate one-syllable caption. Use connected "
            "phrase-level Hangul captions, placed near audible pitch contour, energy, "
            "rest, or melody changes, with medium density and no mechanical equal spacing."
        )
    if mode == "clear-attack":
        return (
            "Caption behavior analysis from audio only: " + stats +
            ". Behavior mode: clear-attack. Detected notes are short and separated. "
            "Use concise Hangul captions near the actual attacks, but still avoid "
            "mechanically even spacing when the audio timing is uneven. For piano-like "
            "or clearly attacked instrumental notes, possible references include "
            "딩, 땡, 띵, 딴, 따란, 따앙, 따단, 따라랑, 따라란, 따아안, 둥, "
            "두웅, and 챙. "
            "These are references rather than a fixed vocabulary: do not alternate "
            "only one or two examples, and change the consonant, vowel, or token length "
            "only when pitch, register, duration, loudness, chord density, attack, or "
            "resonance audibly changes."
        )
    return (
        "Caption behavior analysis from audio only: " + stats +
        ". Behavior mode: sustained-texture. Use connected phrase-level captions "
        "unless there is a clearly audible new attack, rest, or melodic change."
    )


def _caption_behavior_from_wav(wav_path: str | None,
                               feature_row: dict | None) -> dict | None:
    stored_onsets, stored_duration = _stored_audio_onsets(feature_row)
    if stored_onsets and stored_duration is not None:
        return _caption_behavior_analysis(
            feature_row,
            stored_onsets,
            stored_duration,
        )

    if not wav_path:
        return None
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        duration = len(y) / sr if sr else 0.0
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, units='time', backtrack=False
        )
        onsets = [float(t) for t in onsets]
    except Exception:
        return None
    return _caption_behavior_analysis(feature_row, onsets, duration)


def build_audio_rapid_onset_hint(wav_path: str,
                                 max_gap: float = 0.55,
                                 min_cluster_size: int = 4,
                                 phrase_mode: bool = False,
                                 string_mode: bool = False,
                                 feature_row: dict | None = None) -> str | None:
    onsets, stored_duration = _stored_audio_onsets(feature_row)
    if onsets and stored_duration is not None:
        duration = stored_duration
    else:
        try:
            import librosa
            y, sr = librosa.load(wav_path, sr=22050, mono=True)
            duration = len(y) / sr if sr else 0.0
            onsets = librosa.onset.onset_detect(
                y=y, sr=sr, units='time', backtrack=False
            )
            onsets = [float(t) for t in onsets]
        except Exception:
            return None

    if not onsets:
        return None

    behavior = _caption_behavior_analysis(feature_row, onsets, duration)
    behavior_text = _caption_behavior_hint(behavior)

    clusters = []
    cur = [onsets[0]]
    for onset in onsets[1:]:
        if onset - cur[-1] <= max_gap:
            cur.append(onset)
        else:
            if len(cur) >= min_cluster_size:
                clusters.append(cur)
            cur = [onset]
    if len(cur) >= min_cluster_size:
        clusters.append(cur)

    if not clusters:
        return None

    clusters = sorted(clusters, key=lambda c: (c[0], -len(c)))
    selected = clusters[:2]
    if clusters and clusters[-1] not in selected:
        selected.append(clusters[-1])

    parts = []
    for cluster in selected:
        shown = _format_sampled_cluster_times(cluster)
        parts.append(
            f"{cluster[0]:.2f}-{cluster[-1]:.2f}s has {len(cluster)} close onsets "
            f"at sampled timestamps [{shown}]"
        )
    hidden = [cluster for cluster in clusters if cluster not in selected]
    hidden_text = ""
    if hidden:
        hidden_text = (
            f" {len(hidden)} additional onset cluster(s) exist between shown clusters "
            f"from {hidden[0][0]:.2f}s to {hidden[-1][-1]:.2f}s; shown clusters are "
            "representative, not exhaustive. Do not leave long silent gaps while "
            "audible notes continue."
        )
    late_final_text = ""
    last_selected_end = max(cluster[-1] for cluster in selected)
    last_onset = onsets[-1]
    if duration and last_onset >= duration * 0.75 and last_onset - last_selected_end > max_gap:
        late_final_text = (
            f" Late isolated final onset detected at {last_onset:.2f}s near the end; "
            "treat it as a strong final-caption anchor and normally include a caption "
            "near it. Omit it only if the audio clearly indicates non-musical noise "
            "or an analysis artifact. Do not omit it merely because the ending is sparse."
        )
    behavior_suffix = (" " + behavior_text) if behavior_text else ""
    texture_style = _connected_texture_style(feature_row, behavior, string_mode)
    if behavior and behavior.get("piano_pedal_attack_override"):
        return (
            "Rapid onset hint from audio: " + "; ".join(parts) +
            "." + hidden_text + late_final_text + behavior_suffix +
            " For pedalled piano, raw audio onsets can include the initial key attack, "
            "pedal resonance, and secondary energy changes. Do not create a separate "
            "caption for weak resonance-only fluctuations, but do not discard clearly "
            "audible key attacks merely because pedal resonance follows them. Rapid "
            "sequences of distinct key attacks should keep captions near those attacks "
            "instead of being merged into a sparse phrase-level summary. Use the "
            "pitch-derived note attacks together with the clearly audible musical "
            "changes. Match caption length to "
            "the assigned note duration: a note shorter than 0.25s may use a one- or "
            "two-syllable token; a note from 0.25s to 0.60s should normally use a "
            "two- or three-syllable token that includes its resonance; and a note "
            "longer than 0.60s should normally use at least four syllables or an "
            "extended vowel/ending to express the sustain. Because this passage has "
            "pedal resonance, one-syllable captions must not dominate the full result. "
            "Do not repeat the same one-syllable token consecutively, and do not "
            "mechanically alternate two short tokens. Follow actual pitch, register, "
            "duration, chord density, loudness, attack, and resonance changes."
        )
    connected_texture_mode = texture_style in {
        "string-like", "wind-like", "generic-sustain"
    }
    if connected_texture_mode:
        duration_length_rule = (
            " Match each caption token length to the locally audible note or phrase "
            "duration: under 0.25s may use 1-2 Hangul syllables; 0.25-0.60s should "
            "normally use 2-4 syllables; over 0.60s must normally use at least 4 "
            "syllables; and over 1.20s should use at least 6 syllables or a clearly "
            "extended vowel/ending. If the audio contains long sustained notes, do "
            "not make the entire result only 2-3 syllable tokens; represent those "
            "long sustains with 4-or-more-syllable captions. "
        )
        if texture_style == "wind-like":
            return (
                "Rapid onset hint from audio: " + "; ".join(parts) +
                "." + hidden_text + late_final_text + behavior_suffix +
                duration_length_rule +
                " Ellipses inside timestamp lists mean additional detected onsets continue "
                "between the shown timestamps; do not treat omitted timestamps as silence "
                "or leave long gaps while audible notes continue. "
                "For wind-like sustained melodic sounds, dense detected onsets often "
                "reflect connected melodic motion, resonance, or vibrato-like airflow "
                "rather than separate attacks. This connected-texture rule overrides "
                "the generic rapid onset rule: do not caption every detected onset "
                "unless there is a clear new attack, clear rest, or clearly new "
                "melodic note. Keep medium density during active sound and avoid "
                "mechanically even intervals. Group dense detected onsets into "
                "flowing phrase-level Hangul captions according to audible pitch "
                "direction, contour, accent, energy, rest, or melody changes. "
                "Wind-like flowing melodic examples may include 띠라라, 두루루, "
                "따라라, 루루루, 띠라라라, 뚜루루, 따라라라, 라라라, or 띠라라라라. "
                "Use longer syllable runs for longer connected phrases. If an opening "
                "or sustained phrase audibly continues across several anchor intervals, "
                "captions with 7 or more Hangul characters are allowed instead of "
                "ending the phrase too early. Do not cycle "
                "these examples repeatedly; choose each caption from the local sound "
                "change. Avoid bowed-string buzz examples like 지이잉 or 찌이잉 unless "
                "the audio clearly sounds string-like, and avoid breath-noise examples "
                "like 후우우 unless the audio clearly sounds airy."
            )
        if texture_style == "generic-sustain":
            return (
                "Rapid onset hint from audio: " + "; ".join(parts) +
                "." + hidden_text + late_final_text + behavior_suffix +
                duration_length_rule +
                " Ellipses inside timestamp lists mean additional detected onsets continue "
                "between the shown timestamps; do not treat omitted timestamps as silence "
                "or leave long gaps while audible notes continue. "
                "For sustained or vibrato-like connected sounds, dense detected onsets "
                "often reflect continuous motion rather than separate attacks. Use "
                "flowing phrase-level Hangul captions chosen from the actual audio "
                "timbre, placed near audible contour, accent, energy, rest, or melody "
                "changes. Neutral sustained instrumental references may include "
                "따란, 다라란, 또링, 도로롱, 따라라랑, 두루룽, 띠리링, 또로롱, "
                "둥, or 다단, "
                "with token length matched to the audible sustain. These are not a "
                "fixed vocabulary. Do not "
                "default to bowed-string buzz examples or piano-like one-syllable "
                "repeats, and do not place captions at mechanically even intervals."
            )
        texture_label = (
            "For bowed string-like connected sounds"
        )
        return (
            "Rapid onset hint from audio: " + "; ".join(parts) +
            "." + hidden_text + late_final_text + behavior_suffix +
            duration_length_rule +
            " Ellipses inside timestamp lists mean additional detected onsets continue "
            "between the shown timestamps; do not treat omitted timestamps as silence "
            "or leave long gaps while audible notes continue. " +
            texture_label + ", dense detected onsets often reflect "
            "tremble, connected texture, vibrato, resonance, or continuous motion "
            "rather than separate attacks. This connected-texture rule overrides the generic rapid "
            "onset rule: do not caption every detected onset unless there is a "
            "clear new attack, clear rest, or clearly new melodic note. Keep "
            "medium density during active sound: do not collapse a long "
            "active phrase into too few captions, and avoid long visible gaps while "
            "the sound is still continuing. As a guide, group dense detected "
            "onsets into flowing phrase-level Hangul captions according to audible "
            "changes, not fixed time spacing. Use a new phrase when direction, pitch "
            "contour, accent, energy, rest, or melody clearly changes. Express longer "
            "stable sustain inside the Hangul token by extending vowels or flowing "
            "syllable runs instead of adding evenly spaced timestamps. Flowing sustained "
            "examples may include short forms like 지잉, 따란, 찌잉, or 따안; medium "
            "forms like 디라라, 띠라라, 따라라, 지이잉, 라라라, or 찌이잉; and "
            "long forms like 지이이잉, 띠라라라라, 따라라라라, 디라라라라, "
            "라라라라라, or 찌이이이잉. Do not cycle "
            "these examples repeatedly or place them at mechanically even intervals. "
            "Do not let the 지잉/찌잉 family dominate the full caption. Reserve that "
            "buzz-like family for genuinely stable bow friction, tremble, or nearly "
            "unchanging sustain. When the audible melody moves, the pitch contour "
            "changes direction, or several distinct notes form a connected figure, "
            "prefer melodic families such as 따란/따안, 띠라라/따라라, or the softer "
            "디라라/라라라 family and lengthen "
            "them to match the phrase. In any active varied string passage with multiple "
            "captions, include both families: use at least one 지잉/찌잉-family caption "
            "for an audible bow-friction or stable-sustain moment, and at least one "
            "따란/띠라라/따라라/디라라/라라라-family caption for an audible "
            "melodic-motion moment. "
            "Only a genuinely unchanged single sustain may remain in one family. Do not "
            "use the same 지/찌 family for more than two consecutive captions unless the "
            "audio is truly unchanged. Choose each caption from the local sound change. "
            "Avoid percussive one-syllable repeats like 딩 딩 딩 or 띵 띵 띵 for "
            "sustained connected sounds."
        )

    if phrase_mode:
        return (
            "Rapid onset hint from audio: " + "; ".join(parts) +
            "." + hidden_text + late_final_text + behavior_suffix +
            " Ellipses inside timestamp lists mean additional detected onsets continue "
            "between the shown timestamps; do not treat omitted timestamps as silence "
            "or leave long gaps while audible notes continue." +
            " For sustained sounds, dense detected onsets may reflect connected texture "
            "or resonance rather than separate attacks. Do not distribute captions at "
            "mechanically even intervals. Place phrase-level captions near audible "
            "changes: new onset groups, pitch contour changes, accent/energy changes, "
            "rests, or melody changes. If the sound stays stable, let one sustained "
            "caption cover it. If the phrase is active but changing, use medium density "
            "without equal spacing. Choose Hangul syllables from the actual audio timbre "
            "instead of using string-like examples by default."
        )

    return (
        "Rapid onset hint from audio: " + "; ".join(parts) +
        "." + hidden_text + late_final_text + behavior_suffix +
        " Ellipses inside timestamp lists mean additional detected onsets continue "
        "between the shown timestamps; do not treat omitted timestamps as silence "
        "or leave long gaps while audible notes continue." +
        " Represent these as repeated Hangul caption units with close timestamps; "
        "do not summarize rapid repeated notes into one token. If the notes are truly "
        "uniform, the same syllable may repeat; if pitch, duration, loudness, attack, "
        "resonance, or rests change, use a small related set of Hangul variants without "
        "adding unsupported variety."
    )


def _skeleton_text_with_rapid_hint(row: dict, wav_path: str | None = None) -> str:
    text = build_skeleton_text(row)
    hints = []
    phrase_mode = "articulation: sustained" in text
    string_mode = _is_confident_string_context(row)
    texture_behavior = _caption_behavior_from_wav(wav_path, row)
    if texture_behavior and texture_behavior.get("piano_pedal_attack_override"):
        phrase_mode = False
    texture_style = _connected_texture_style(row, texture_behavior, string_mode)
    compact_anchor = (
        texture_behavior is not None
        and texture_behavior.get("mode") == "sustained-texture"
        and texture_style in {"string-like", "wind-like", "generic-sustain"}
    )
    note_hint = build_rapid_onset_hint(row)
    if note_hint:
        hints.append(note_hint)
    anchor_hint = build_caption_anchor_hint(
        row,
        compact_sustained=compact_anchor,
        expanded_string_sustained=texture_style == "string-like",
        expanded_wind_sustained=texture_style == "wind-like",
        expanded_generic_sustained=texture_style == "generic-sustain",
    )
    if anchor_hint:
        hints.append(anchor_hint)
    if wav_path:
        audio_hint = build_audio_rapid_onset_hint(
            wav_path,
            phrase_mode=phrase_mode,
            string_mode=string_mode,
            feature_row=row,
        )
        if audio_hint:
            hints.append(audio_hint)
    return text + (" | " + " | ".join(hints) if hints else "")


def get_skeleton_text_for_file(wav_path: str,
                               feature_csv_path: str | None = None,
                               runtime_features: dict | None = None) -> str | None:
    """
    ① 주입용: wav 파일에 해당하는 골격 텍스트를 만든다.

    interpret_music_features 가 받는 features dict 에는 notes 가 없으므로,
    music_features.csv 에서 file_name 으로 행을 찾아 notes 를 가져온다.
    csv 에 없으면(런타임 새 음원) None 반환 → 호출부에서 골격 없이 진행.

    반환 예: "Skeleton: 5 onsets | length pattern: short-short-... |
              breaks(rests): 1 (before onsets [3]) | ... | articulation: sustained"
    """
    if feature_csv_path is None:
        feature_csv_path = join(dirname(__file__), '..', 'music_features.csv')
    if os.path.exists(feature_csv_path):
        df = _load_feature_df(feature_csv_path)
        fname = _audio_key(wav_path)
        if 'file_name' in df.columns:
            matched = df[df['file_name'] == fname]
            if not matched.empty:
                row = matched.iloc[0].to_dict()
                if 'notes' in row and not pd.isna(row.get('notes')):
                    return _skeleton_text_with_rapid_hint(row, wav_path)

    if runtime_features and runtime_features.get('notes'):
        return _skeleton_text_with_rapid_hint(runtime_features, wav_path)
    return None


def get_consensus_text_for_file(wav_path: str,
                                annotation_json_path: str | None = None) -> str | None:
    """
    ② 주입용: wav 에 달린 사람 전사(여러 명)의 합의도 요약.

    generate_caption.py 가 이미 쓰는 human annotation json
    (rag/mucap_annotation_fin.json 형식)에서 같은 파일의 전사들을 모아
    합의도를 요약한다. 매칭/전사 없으면 None.
    """
    import json
    if annotation_json_path is None:
        annotation_json_path = join(dirname(__file__), '..',
                                    'human_annotation', 'human_annotation.json')
    if not os.path.exists(annotation_json_path):
        return None

    fname = _audio_key(wav_path)
    transcripts = []
    try:
        with open(annotation_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            audio_url = item.get('data', {}).get('audio', '')
            if _audio_key(audio_url.split('/')[-1]) != fname:
                continue
            for ann in item.get('annotations', []):
                for result in ann.get('result', []):
                    val = result.get('value', {})
                    if 'text' in val and val['text']:
                        transcripts.append(val['text'][0])
    except Exception:
        return None

    if not transcripts:
        return None
    return annotation_consensus_summary(transcripts)


# ============================================================================
# 패치 적용 함수: build_prompt_rhythm 의 feature_str 를 보강
# ============================================================================

def augment_feature_str(feature_str: str,
                        wav_path: str,
                        feature_csv_path: str | None = None,
                        annotation_json_path: str | None = None,
                        runtime_features: dict | None = None,
                        add_skeleton: bool = True,
                        add_consensus: bool = True) -> str:
    """
    interpret_music_features 가 만든 feature_str 에 골격/합의도를 덧붙인다.

    generate_caption.py 의 build_prompt_rhythm 에서:
        feature_str = interpret_music_features(music_features)
        feature_str = augment_feature_str(feature_str, wav_path)   # ← 이 한 줄 추가
        if use_rag: ...
    """
    additions = []
    if add_skeleton:
        skel_txt = get_skeleton_text_for_file(
            wav_path,
            feature_csv_path,
            runtime_features=runtime_features,
        )
        if skel_txt:
            additions.append(skel_txt)
    if add_consensus:
        cons_txt = get_consensus_text_for_file(wav_path, annotation_json_path)
        if cons_txt:
            additions.append(cons_txt)

    if not additions:
        return feature_str

    # [Context] 블록 뒤에 골격/합의도를 별도 라인으로 추가
    extra = "\n".join(f"- {a}" for a in additions)
    return feature_str + "\n- [Rhythm skeleton & consensus from analysis]\n" + extra


_PATCH_GUIDE = r"""
=== mucap/generate_caption.py 패치 가이드 (목적 A) ===

1) import 추가 (파일 상단):
   from caption_skeleton_patch import augment_feature_str

2) build_prompt_rhythm() 안에서, feature_str 만든 직후 한 줄 추가:

   [변경 전]
       if music_features is not None:
           feature_str = interpret_music_features(music_features)
       else:
           feature_str = "3) [Context]: - No music features available; ..."

       if use_rag:
           rag_context = retrieve_rag_context(feature_str, threshold=0.1)
           ...

   [변경 후]
       if music_features is not None:
           feature_str = interpret_music_features(music_features)
       else:
           feature_str = "3) [Context]: - No music features available; ..."

       # ★ 골격 + 합의도 주입 (wav_path를 build_prompt_rhythm 인자로 받아야 함)
       feature_str = augment_feature_str(feature_str, wav_path)

       if use_rag:
           rag_context = retrieve_rag_context(feature_str, threshold=0.1)
           ...

3) build_prompt_rhythm 시그니처에 wav_path 추가:
   [변경 전] def build_prompt_rhythm(audio_b64, design_prompt=None, music_features=None, use_rag=None):
   [변경 후] def build_prompt_rhythm(audio_b64, design_prompt=None, music_features=None, use_rag=None, wav_path=None):

   그리고 annotate_and_show 안의 호출부:
   [변경 전] messages = build_prompt_rhythm(audio_b64, design_txt, music_features, use_rag)
   [변경 후] messages = build_prompt_rhythm(audio_b64, design_txt, music_features, use_rag, wav_path=wav_path)

4) (RAG DB에도 골격 합류 — rag_skeleton_patch.py 가이드 참조)
   json2ragdb 노트북의 row_txt 에 build_skeleton_text/consensus 를 넣어야
   검색 대상과 프롬프트 주입이 일관됨.

이렇게 하면 LLM 프롬프트의 [Context]에 다음이 추가된다:
  - [Rhythm skeleton & consensus from analysis]
  - Skeleton: 5 onsets | length pattern: ... | breaks(rests): 1 ...
  - Consensus: 5 annotators | median syllables ~7 | syllable agreement 0.41 ...

LLM은 이 골격(음절수/장단/쉼표)과 합의도를 참고해 음절수·띄어쓰기를 맞춘다.
"""

if __name__ == "__main__":
    print(_PATCH_GUIDE)
