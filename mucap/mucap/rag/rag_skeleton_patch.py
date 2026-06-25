"""
rag_skeleton_patch.py
=====================
MuCap의 mucap/rag/json2ragdb Jupyter Notebook (.ipynb 파일)에 "골격(Skeleton)"을 합류시키는 패치.

원본 Jupyter Notebook (.ipynb 파일)의 docs 생성 루프에서 row_txt를 만드는 부분:

    row_txt = f"Music Feature: {music_feature_txt}\n"
              f"Music Feeling: {music_feeling_txt}\n"
              f"Annotation: {annotation}"

여기에 Skeleton 한 블록을 추가하는 것이 목표.
notes(이미 music_features.csv에 있음)에서 골격을 뽑아 LLM이 읽을 텍스트로 직렬화한다.

원본 코드를 최소 침습으로 바꾸기 위해, 두 개의 헬퍼만 호출하면 되도록 했다:
  - build_skeleton_text(feature_row)      → "Skeleton: ..." 텍스트
  - split_annotations(result)             → 5명 전사를 리스트로 분리 (합의도용)

사용법은 파일 하단 PATCH GUIDE 참고.
"""

from __future__ import annotations

# Skeleton 변환과 전사 분석 로직은 프로젝트 루트의 heuristic 패키지에서 공유한다.
from heuristic.data_io import feature_row_to_skeleton
from heuristic.sound_to_skeleton_core import (
    count_syllables,
    count_spacing_breaks,
    looks_like_solfege,
)


# ============================================================================
# 1. feature 행(pandas Series 또는 dict) → 골격 텍스트
#    notes 컬럼을 Skeleton으로 변환하고, LLM이 읽을 서술형으로 직렬화.
# ============================================================================

def build_skeleton_text(feature_row) -> str:
    """
    music_features.csv 한 행 → "Skeleton: ..." 텍스트 블록.

    feature_row: pandas Series 또는 dict (notes, attack, top5_instruments 등 포함)
    반환 예:
      "Skeleton: 5 onsets | length pattern: short-short-short-long-long |
       breaks(rests): 1 (after onset 3) | pitch contour: flat-flat-up |
       articulation: percussive"
    """
    row = dict(feature_row)  # Series든 dict든 dict로 통일
    skel = feature_row_to_skeleton(row)

    if skel.onset_count == 0:
        return "Skeleton: (no clear onsets detected)"

    # 길이 패턴 (장단) — 중앙값 기준 이진화
    import numpy as np
    durs = np.array(skel.durations) if skel.durations else np.array([0.0])
    median = float(np.median(durs)) if len(durs) else 0.0
    length_pat = []
    for d in skel.durations:
        if median > 0 and d >= median * 1.5:
            length_pat.append("long")
        elif median > 0 and d < median * 0.6:
            length_pat.append("short")
        else:
            length_pat.append("mid")
    length_str = "-".join(length_pat) if length_pat else "n/a"

    # 끊김(쉼표) 위치
    n_breaks = len(skel.rest_gaps)
    break_positions = []
    for i in range(1, len(skel.syllable_groups)):
        if skel.syllable_groups[i] != skel.syllable_groups[i - 1]:
            break_positions.append(i)  # i번째 onset 앞에서 끊김
    break_str = (f"{n_breaks} (before onsets {break_positions})"
                 if n_breaks else "0 (continuous)")

    # 음높이 윤곽 (시각화 보조 + 약한 표기 단서)
    contour = []
    for i in range(1, len(skel.pitches)):
        diff = skel.pitches[i] - skel.pitches[i - 1]
        contour.append("up" if diff > 1 else "down" if diff < -1 else "flat")
    contour_str = "-".join(contour) if contour else "n/a"

    articulation = "percussive" if skel.is_percussive else "sustained"

    return (f"Skeleton: {skel.onset_count} onsets | "
            f"length pattern: {length_str} | "
            f"breaks(rests): {break_str} | "
            f"pitch contour: {contour_str} | "
            f"articulation: {articulation}")


# ============================================================================
# 2. label-studio result → 5명 전사 분리 (합의도 측정용)
#    원본은 ' '.join(text_arr)로 뭉개는데, 우리는 리스트를 보존해 합의도를 잰다.
# ============================================================================

def split_annotations(result: list) -> list[str]:
    """
    원본 Jupyter Notebook (.ipynb 파일)의 item['annotations'][0]['result']에서 전사 텍스트만 추출해
    개별 전사 리스트로 반환 (합치지 않음).

    원본:
        for row in result:
            value = row['value']
            if 'text' not in value: continue
            text_arr.append(value['text'][0])
        annotation = ' '.join(text_arr)   # ← 뭉갬

    여기서는 text_arr 그대로(리스트) 반환해 measure_consensus에 넘긴다.
    """
    texts = []
    for row in result:
        value = row.get('value', {})
        if 'text' not in value:
            continue
        t = value['text'][0] if value['text'] else ""
        if t.strip():
            texts.append(t.strip())
    return texts


def annotation_consensus_summary(transcripts: list[str]) -> str:
    """
    5명(또는 N명) 전사 → 합의도 요약 텍스트.
    이걸 row_txt에 함께 넣으면, RAG 검색 시 "이 음원은 사람들이 몇 음절로,
    얼마나 일치하게 적었는지"를 LLM이 참고할 수 있다 (모드 B 신뢰도 신호).
    """
    import numpy as np
    musical = [t for t in transcripts if not looks_like_solfege(t)]
    if not musical:
        return "Consensus: (no usable transcripts)"

    syll_counts = [count_syllables(t, collapse_long=True) for t in musical]
    syll_counts = [c for c in syll_counts if c > 0]
    spacing = [count_spacing_breaks(t) for t in musical if " " in t.strip()]

    if not syll_counts:
        return "Consensus: (no syllables)"

    mean_syll = float(np.mean(syll_counts))
    syll_agree = (1.0 / (1.0 + float(np.std(syll_counts)))
                  if len(syll_counts) > 1 else 0.5)
    n_no_spacing = sum(1 for t in musical if " " not in t.strip())

    return (f"Consensus: {len(musical)} annotators | "
            f"median syllables ~{round(mean_syll)} | "
            f"syllable agreement {syll_agree:.2f} | "
            f"no-spacing annotators {n_no_spacing}")


# ============================================================================
# PATCH GUIDE — 원본 Jupyter Notebook (.ipynb 파일) 수정법
# ============================================================================
_PATCH_GUIDE = r"""
=== mucap/rag/json2ragdb Jupyter Notebook (.ipynb 파일) 수정 가이드 ===

원본 docs 생성 루프의 끝부분을 아래처럼 바꾼다.

[변경 전]
    annotation = ' '.join(text_arr)
    row_txt = f"Music Feature: {music_feature_txt}\nMusic Feeling: {music_feeling_txt}\nAnnotation: {annotation}"

[변경 후]
    from rag_skeleton_patch import (
        build_skeleton_text, split_annotations, annotation_consensus_summary)

    # 5명 전사를 합치지 말고 리스트로 (text_arr가 곧 전사 리스트)
    transcript_list = text_arr                     # 또는 split_annotations(result)
    annotation = ' '.join(transcript_list)         # 기존 표시는 유지

    # 골격 + 합의도 텍스트 추가
    skeleton_txt = build_skeleton_text(feature)    # feature = music_feature_df의 그 행
    consensus_txt = annotation_consensus_summary(transcript_list)

    row_txt = (
        f"Music Feature: {music_feature_txt}\n"
        f"{skeleton_txt}\n"                          # ← 골격 한 줄
        f"{consensus_txt}\n"                         # ← 합의도 한 줄
        f"Music Feeling: {music_feeling_txt}\n"
        f"Annotation: {annotation}"
    )

이렇게 하면:
  - 골격(음절수/장단/쉼표/음색)이 임베딩 텍스트에 포함되어 검색·생성에 활용됨
  - 합의도가 함께 들어가 모드 B의 신뢰도 신호가 됨
  - 기존 Music Feature/Feeling/Annotation 구조는 그대로 보존
"""

if __name__ == "__main__":
    print(_PATCH_GUIDE)
