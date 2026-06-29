"""
data_io.py — 실제 MuCap 데이터 파일을 읽어 text 입력으로 변환

파일 구조 (실제 데이터 기준):
  1. audio feature CSV
     컬럼: file_name, genre, dataset, sample_rate, length, volume, pitch,
           tempo, attack, envelope, onset_strength, resonance, notes,
           beat_frames, top5_instruments, file
     - notes: "[{'onset_time':..,'pitch':..(Hz),'duration':..}, ...]" (문자열)
     - pitch는 Hz 단위 (MIDI 번호 아님 → 변환 필요)
     - attack(sharp/soft), envelope(transient/...) 는 이미 라벨로 존재

  2. 전사 CSV
     컬럼: clip_id, instrument, song_id, audio_url, U0063, U0365, ... (20명)
     - 한 행 = 한 클립, 주석자 20명이 열
     - 한 클립당 5명만 채워짐 (나머지 빈칸)
     - 조인 키: clip_id

  ※ audio feature CSV의 file_name 과 전사 CSV의 clip_id 가 같은 음원을 가리킨다는
    전제로 조인. 키 이름이 다르면 FEATURE_ID_COL / TRANSCRIPT_ID_COL 만 바꾸면 됨.

  audio feature JSON 도 있을 수 있으나, notes/attack/envelope 가 CSV에 이미
  있으므로 골격 추출에는 CSV만으로 충분하다. JSON에 검색용 임베딩 벡터가 따로
  있으면 load_audio_feature_json 으로 읽어 retrieval 에 쓴다.
"""

from __future__ import annotations
import csv
import ast
import json
import math
from dataclasses import dataclass

import numpy as np

from mucap.sound_to_skeleton_core import (
    Skeleton, ConsensusModel,
    count_syllables, normalize_transcript,
)

# ---- 조인 키 / 컬럼 이름 (실제 파일에서 다르면 여기만 수정) ----------------
FEATURE_ID_COL = "file_name"      # audio feature CSV의 음원 식별 컬럼
TRANSCRIPT_ID_COL = "clip_id"     # 전사 CSV의 음원 식별 컬럼
TRANSCRIPT_META_COLS = {"clip_id", "instrument", "song_id", "audio_url"}
# 위 메타를 뺀 나머지 컬럼이 주석자(U0063 등)로 간주됨


# ============================================================================
# 1. Hz → MIDI 변환 (audio feature의 pitch가 Hz이므로)
# ============================================================================

def hz_to_midi(hz: float) -> int:
    """주파수(Hz) → MIDI 노트 번호. A4=440Hz=69.
    0 이하/비정상 값은 중앙 C(60)로 폴백."""
    if hz is None or hz <= 0 or not math.isfinite(hz):
        return 60
    midi = 69 + 12 * math.log2(hz / 440.0)
    # 악기 음역(대략 21~108)으로 클램프
    return int(round(max(21, min(108, midi))))


# ============================================================================
# 2. audio feature CSV의 notes → Skeleton (MIDI 단계 생략)
#    extract_skeleton(midi_path)을 대체. 같은 Skeleton 구조를 반환하므로
#    이후 파이프라인은 동일하게 동작한다.
# ============================================================================

def parse_notes_field(notes_str) -> list[dict]:
    """notes 문자열을 dict 리스트로 안전 파싱.
    "[{'onset_time':..,'pitch':..,'duration':..}, ...]" 형식."""
    if isinstance(notes_str, list):
        return [n for n in notes_str if isinstance(n, dict)]
    if not notes_str or not notes_str.strip():
        return []
    try:
        parsed = ast.literal_eval(notes_str)
        if isinstance(parsed, list):
            return [n for n in parsed if isinstance(n, dict)]
    except (ValueError, SyntaxError):
        pass
    return []


def feature_row_to_skeleton(row: dict,
                            rest_threshold: float = 0.18,
                            percussive_dur_thresh: float = 0.18) -> Skeleton:
    """
    audio feature CSV 한 행 → Skeleton.

    - notes에서 onset/pitch(Hz→MIDI)/duration 추출
    - attack(sharp) → onset_strength 보조, envelope → 길이 해석에 활용 가능
    - is_percussive: top5_instruments에 drum/percussion이 우세하거나
                     attack=sharp & 음 길이가 짧으면 타악적으로 판정
                     (악기 라벨이 아니라 실제 음색/주법 기준)
    """
    notes = parse_notes_field(row.get("notes", ""))

    if not notes:
        return Skeleton(0, [], [], [], [], False, [])

    notes.sort(key=lambda n: n.get("onset_time", 0.0))

    onsets = [float(n.get("onset_time", 0.0)) for n in notes]
    durations = [float(n.get("duration", 0.0)) for n in notes]
    pitches = [hz_to_midi(float(n.get("pitch", 0.0))) for n in notes]

    # onset_strength: CSV에 전체 곡 단위 값 1개만 있으므로, 음별로는 균일 적용.
    # attack=sharp 이면 어택 강함으로 본다.
    base_strength = _safe_float(row.get("onset_strength"), 0.5)
    attack = (row.get("attack") or "").strip().lower()
    if attack == "sharp":
        base_strength = max(base_strength, 0.7)
    elif attack == "soft":
        base_strength = min(base_strength, 0.4)
    strengths = [base_strength] * len(notes)

    # 쉼표(무음 구간) → 띄어쓰기 후보 + syllable_groups
    rest_gaps = []
    syllable_groups = [0]
    cur = 0
    for i in range(1, len(onsets)):
        gap = onsets[i] - (onsets[i - 1] + durations[i - 1])
        if gap > rest_threshold:
            rest_gaps.append((onsets[i - 1] + durations[i - 1], onsets[i]))
            cur += 1
        syllable_groups.append(cur)

    # is_percussive: top5에서 타악기 우세 or (sharp 어택 & 짧은 음)
    is_perc = _is_percussive_from_feature(row, durations, percussive_dur_thresh)

    return Skeleton(
        onset_count=len(notes),
        durations=durations,
        rest_gaps=rest_gaps,
        pitches=pitches,
        onset_strengths=strengths,
        is_percussive=is_perc,
        syllable_groups=syllable_groups,
    )


def _is_percussive_from_feature(row: dict, durations: list[float],
                                dur_thresh: float) -> bool:
    """음색/주법 기반 타악성 판정 (악기 라벨 고정이 아님)."""
    top5 = _parse_top5(row.get("top5_instruments", ""))
    perc_score = sum(s for name, s in top5
                     if name in ("drums", "percussion"))
    melodic_score = sum(s for name, s in top5
                        if name in ("violin", "viola", "cello", "guitar",
                                    "electricguitar", "acousticguitar",
                                    "flute", "clarinet", "trumpet", "voice"))
    # 타악기 신뢰도가 선율악기보다 높으면 타악적
    if perc_score > melodic_score and perc_score > 0.3:
        return True
    # 또는 음들이 전반적으로 짧으면(어택 위주) 타악적
    if durations:
        med = float(np.median(durations))
        attack = (row.get("attack") or "").strip().lower()
        if med < dur_thresh and attack == "sharp":
            return True
    return False


def _parse_top5(s) -> list[tuple[str, float]]:
    """top5_instruments string/list -> [(name, score), ...]."""
    if isinstance(s, list):
        out = []
        for item in s:
            try:
                if isinstance(item, dict):
                    name = item.get("instrument") or item.get("name") or item.get("label")
                    score = item.get("score") or item.get("probability") or item.get("confidence")
                else:
                    name, score = item
                out.append((str(name).lower(), float(score)))
            except (TypeError, ValueError):
                continue
        return out
    if not s or not s.strip():
        return []
    try:
        parsed = ast.literal_eval(s)
        return [(str(n).lower(), float(sc)) for n, sc in parsed]
    except (ValueError, SyntaxError):
        return []


def dominant_instrument(row: dict, fallback: str = "default") -> str:
    """전사 CSV의 instrument 컬럼이 있으면 그걸, 없으면 top5 1위를 악기로."""
    inst = (row.get("instrument") or "").strip().lower()
    if inst:
        return inst
    top5 = _parse_top5(row.get("top5_instruments", ""))
    return top5[0][0] if top5 else fallback


def _safe_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ============================================================================
# 3. audio feature CSV 로드
# ============================================================================

def load_audio_features(csv_path: str) -> dict[str, dict]:
    """audio feature CSV → {file_name: row_dict}."""
    features = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get(FEATURE_ID_COL, "").strip()
            if key:
                features[key] = row
    return features


# ============================================================================
# 4. 전사 CSV 로드 (형태 A: 주석자 20명이 열, 클립당 5명만 채워짐)
# ============================================================================

@dataclass
class ClipTranscripts:
    """한 클립에 달린 전사들."""
    clip_id: str
    instrument: str
    song_id: str
    transcripts: dict[str, str]   # {annotator_id: text} (채워진 것만)

    def texts(self) -> list[str]:
        return list(self.transcripts.values())

    def annotators(self) -> list[str]:
        return list(self.transcripts.keys())


def load_transcripts(csv_path: str) -> dict[str, ClipTranscripts]:
    """전사 CSV(형태 A) → {clip_id: ClipTranscripts}.
    빈 셀은 제외하고, 실제 전사한 주석자만 담는다."""
    clips = {}
    with open(csv_path, encoding="utf-8") as f:
        # 탭 구분일 수도, 콤마 구분일 수도 있어 자동 감지
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delim)
        fieldnames = reader.fieldnames or []
        annotator_cols = [c for c in fieldnames
                          if c and c not in TRANSCRIPT_META_COLS]
        for row in reader:
            clip_id = (row.get(TRANSCRIPT_ID_COL) or "").strip()
            if not clip_id:
                continue
            filled = {}
            for ann in annotator_cols:
                val = (row.get(ann) or "").strip()
                if val:
                    filled[ann] = val
            clips[clip_id] = ClipTranscripts(
                clip_id=clip_id,
                instrument=(row.get("instrument") or "").strip(),
                song_id=(row.get("song_id") or "").strip(),
                transcripts=filled,
            )
    return clips


# ============================================================================
# 5. audio feature JSON 로드 (검색용 임베딩이 따로 있는 경우)
# ============================================================================

def load_audio_feature_json(json_path: str) -> dict[str, dict]:
    """audio feature JSON → {audio_id: {...}}.
    구조가 dict-of-dict 또는 list-of-dict 둘 다 처리."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out = {}
        for item in data:
            key = item.get(FEATURE_ID_COL) or item.get("audio_id") or item.get("id")
            if key:
                out[str(key)] = item
        return out
    return {}


# ============================================================================
# 6. 조인 + 합의도 측정용 annotation 레코드 생성
# ============================================================================

def build_annotation_records(transcripts: dict[str, ClipTranscripts],
                             features: dict[str, dict]
                             ) -> list[dict]:
    """
    전사 + audio feature를 clip_id로 조인해, measure_consensus가 먹는
    레코드 리스트로 변환.

    반환 각 항목:
      { "clip_id", "song_id", "instrument",
        "onset_count": int,            # 골격에서
        "transcripts": [str, ...],     # 채워진 5명
        "annotators": [str, ...],      # 주석자 ID (U9608 등 추적)
        "skeleton": Skeleton }
    매칭되는 feature가 없으면 onset_count는 None으로 두고 그대로 기록
    (전사 합의도는 골격 없이도 음절/띄어쓰기 축에서 측정 가능).
    """
    records = []
    for clip_id, ct in transcripts.items():
        feat = features.get(clip_id)
        skel = feature_row_to_skeleton(feat) if feat else None
        records.append({
            "clip_id": clip_id,
            "song_id": ct.song_id,
            "instrument": ct.instrument,
            "onset_count": skel.onset_count if skel else None,
            "transcripts": ct.texts(),
            "annotators": ct.annotators(),
            "skeleton": skel,
        })
    return records


if __name__ == "__main__":
    import sys
    # 사용 예: python data_io.py transcripts.csv features.csv
    if len(sys.argv) >= 3:
        trans = load_transcripts(sys.argv[1])
        feats = load_audio_features(sys.argv[2])
        recs = build_annotation_records(trans, feats)
        matched = sum(1 for r in recs if r["onset_count"] is not None)
        print(f"전사 클립: {len(trans)} | feature: {len(feats)} | "
              f"조인 성공: {matched}")
        for r in recs[:3]:
            print(f"  {r['clip_id']}: 주석자 {len(r['transcripts'])}명, "
                  f"onset={r['onset_count']}, 악기={r['instrument']}")
    else:
        print("usage: python data_io.py <transcripts.csv> <features.csv>")
