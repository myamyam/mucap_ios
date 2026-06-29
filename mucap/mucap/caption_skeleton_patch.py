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
    shown = ", ".join(f"{t:.2f}" for t in cluster[:10])
    if len(cluster) > 10:
        shown += ", ..."
    return (
        f"Rapid onset hint: {start:.2f}-{end:.2f}s has {len(cluster)} close onsets "
        f"at [{shown}]. Represent this as repeated Hangul caption units with close "
        "timestamps; do not summarize the rapid notes into one token."
    )


def build_audio_rapid_onset_hint(wav_path: str,
                                 max_gap: float = 0.55,
                                 min_cluster_size: int = 4) -> str | None:
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, units='time', backtrack=False
        )
        onsets = [float(t) for t in onsets]
    except Exception:
        return None

    if not onsets:
        return None

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

    clusters = sorted(clusters, key=lambda c: (c[0], -len(c)))[:2]
    parts = []
    for cluster in clusters:
        shown = ", ".join(f"{t:.2f}" for t in cluster[:10])
        if len(cluster) > 10:
            shown += ", ..."
        parts.append(
            f"{cluster[0]:.2f}-{cluster[-1]:.2f}s has {len(cluster)} close onsets "
            f"at [{shown}]"
        )
    return (
        "Rapid onset hint from audio: " + "; ".join(parts) +
        ". Represent these as repeated Hangul caption units with close timestamps; "
        "do not summarize rapid repeated notes into one token."
    )


def _skeleton_text_with_rapid_hint(row: dict, wav_path: str | None = None) -> str:
    text = build_skeleton_text(row)
    hints = []
    note_hint = build_rapid_onset_hint(row)
    if note_hint:
        hints.append(note_hint)
    if wav_path:
        audio_hint = build_audio_rapid_onset_hint(wav_path)
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
