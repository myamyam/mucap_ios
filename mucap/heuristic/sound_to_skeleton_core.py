"""
Sound-to-Text 파이프라인 (수정판)
====================================

원본 Algorithm 2 (전문가 guideline 의사코드)를 다음 분석에 따라 수정:

  [1] random() 음절 선택        → 검색 예시로 가중된 선택 (모드 B)
  [2] target_len = duration*2.5 → MIDI onset + 5명 합의도 보정 (grouping)
  [3] 악기 라벨 고정 스타일      → 주법/어택 기반 동적 판단 (타악/선율)

추가 반영:
  - 국립국어원 음운 표기 규칙 (스캣 관련 부분만 선별)
  - 코나콜/볼 음절 (타악적 음색에 한정)
  - guideline은 약한 prior, 검색 예시가 우선

guideline 자체는 datastore에 넣지 않고 추론 시 규칙으로 주입한다.
골격 추출 함수는 datastore 보강과 새 음원 처리에서 동일하게 쓰인다(일관성).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional
import numpy as np

# pretty_midi는 MIDI 파싱용. 환경에 없으면 골격 추출만 건너뛰면 됨.
try:
    import pretty_midi
    _HAS_MIDI = True
except ImportError:
    _HAS_MIDI = False


# ============================================================================
# 1. 전문가 guideline (Algorithm 2의 GetInstrumentStyle을 딕셔너리화)
#    - 의사코드의 [ㅌ, ㄸ, ...] 생략부는 대표 자모음으로 채운 골조.
#      실제 전체 풀이 확보되면 이 딕셔너리만 교체하면 된다.
# ============================================================================

INSTRUMENT_STYLE = {
    "percussion": {  # drum, percussion
        "consonants": ["ㅌ", "ㄸ", "ㄷ", "ㅋ", "ㅂ"],
        "vowels": ["ㅡ", "ㅜ", "ㅏ"],
        "tails": ["ㄱ", "ㅇ"],
        "use_connector": False,
        "repeat_emphasis": True,
    },
    "strings": {  # violin, viola (찰현)
        "consonants": ["ㄷ", "ㅈ", "ㄹ", "ㅂ"],
        "vowels": ["ㅣ", "ㅏ", "ㅔ"],
        "tails": ["ㄹ", "ㅇ"],
        "use_connector": True,
        "repeat_emphasis": True,
    },
    "piano": {  # piano, keyboard
        "consonants": ["ㄷ", "ㅂ", "ㄸ", "ㄱ"],
        "vowels": ["ㅏ", "ㅗ", "ㅜ"],
        "tails": ["ㄴ", "ㅇ"],
        "use_connector": True,
        "repeat_emphasis": True,
    },
    "default": {
        "consonants": ["ㄷ", "ㅂ", "ㄱ"],
        "vowels": ["ㅏ", "ㅜ"],
        "tails": ["ㅇ"],
        "use_connector": True,
        "repeat_emphasis": False,
    },
}

# 악기 라벨 → 스타일 키 (정규화)
_LABEL_TO_STYLE = {
    "drum": "percussion", "drums": "percussion", "percussion": "percussion",
    "violin": "strings", "viola": "strings", "cello": "strings",
    "piano": "piano", "keyboard": "piano",
}

# 코나콜/볼 음절 (타악적 음색에 한정해 추가 후보로만 사용 — 직접 정답 아님)
# 인도 음운을 한글로 근사: 권설음/유기음 구분은 한글로 뭉개지므로 대표형만.
KONNAKOL_BOL = {
    "consonants": ["ㄷ", "ㄸ", "ㅌ", "ㄱ", "ㄴ"],
    "vowels": ["ㅏ", "ㅣ", "ㅜ", "ㅗ"],
    "tails": ["ㄴ", "ㅇ", "ㅁ"],
}


# ============================================================================
# 2. 국립국어원 음운 표기 규칙 (스캣 관련 부분만 선별)
#    - 표준 맞춤법 전체가 아니라, 한글 음절이 "소리를 적을 수 있는 범위"에
#      대한 규칙만. 스캣의 비규범성을 해치지 않는 선에서 형태 정합성만 보장.
# ============================================================================

# 한글 음절 구조: 초성(19) + 중성(21) + 종성(27, 없음 포함)
# (출처: 국립국어원 한글 음절 구성 원리)
CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")          # 19
JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")        # 21
JONGSEONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")  # 28

# 음절말 자음 중화: 종성에서 실제로 "터지지 않고" 닫히는 소리는 7개로 수렴
# (국립국어원 표준발음법 — 받침소리는 ㄱㄴㄷㄹㅁㅂㅇ 7개로 중화).
# 스캣에서 받침은 "음이 어떻게 끝나는가(닫힘)"를 적는 것이므로 이 7개만 의미 있음.
# guideline의 tails가 이 집합을 벗어나면 가장 가까운 중화음으로 보정한다.
NEUTRALIZED_CODA = set("ㄱㄴㄷㄹㅁㅂㅇ")
_CODA_NEUTRALIZE_MAP = {
    "ㄲ": "ㄱ", "ㅋ": "ㄱ",                 # 연구개 파열 → ㄱ
    "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ",
    "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",      # 치경/경구개 → ㄷ
    "ㅍ": "ㅂ",                             # 양순 파열 → ㅂ
}

def neutralize_coda(coda: str) -> str:
    """종성을 표준발음법의 7종성으로 중화. 스캣 받침 표기의 형태 정합성 보장."""
    if not coda or coda in NEUTRALIZED_CODA:
        return coda
    return _CODA_NEUTRALIZE_MAP.get(coda, coda if coda in NEUTRALIZED_CODA else "")


def compose_syllable(cho: str, jung: str, jong: str = "") -> str:
    """초성+중성+종성을 한글 음절로 합성 (유니코드 조합).
    중화 규칙을 적용해 음절말 표기를 정규화한다."""
    jong = neutralize_coda(jong)
    try:
        ci = CHOSEONG.index(cho)
        ji = JUNGSEONG.index(jung)
        ki = JONGSEONG.index(jong)
    except ValueError:
        return cho + jung + jong  # 합성 불가 시 자모 나열로 폴백
    code = 0xAC00 + (ci * 21 + ji) * 28 + ki
    return chr(code)


# ============================================================================
# 3. MIDI 골격 추출 ([2] 해결: duration*2.5 대신 onset 기반)
#    - datastore 보강과 새 음원에서 동일하게 호출 (표현 일관성)
# ============================================================================

@dataclass
class Skeleton:
    """MIDI에서 뽑은 리듬 골격. LLM이 읽을 직렬화의 원천."""
    onset_count: int                 # 물리적 음표 수 (사람 글자수와 다를 수 있음)
    durations: list[float]           # 각 음의 길이(초)
    rest_gaps: list[tuple[float, float]]  # 무음 구간 = 띄어쓰기 후보 (가장 신뢰)
    pitches: list[int]               # 음높이 (시각화 + 약한 표기 보조)
    onset_strengths: list[float]     # 어택 세기 (velocity 기반)
    is_percussive: bool              # [3] 음색 트리거: 타악적 연주인가
    syllable_groups: list[int]       # 쉼표 기준으로 묶은 음절 그룹 인덱스

    def total_duration(self) -> float:
        return sum(self.durations) + sum(e - s for s, e in self.rest_gaps)


def extract_skeleton(midi_path: str,
                     merge_chord_window: float = 0.05,
                     rest_threshold: float = 0.18,
                     percussive_dur_thresh: float = 0.18) -> Skeleton:
    """
    MIDI → 리듬 골격.

    [3] is_percussive: 악기 라벨이 아니라 '실제 연주'로 판정.
        - 드럼 트랙이거나
        - 음들이 짧고(어택 위주) 끊겨 있으면 타악적으로 본다.
        같은 베이스라도 슬랩(짧음)=타악적, 아르코(긺)=선율적.
    """
    if not _HAS_MIDI:
        raise RuntimeError("pretty_midi 필요")

    pm = pretty_midi.PrettyMIDI(midi_path)

    raw = []  # (start, end, pitch, velocity, is_drum)
    for inst in pm.instruments:
        for n in inst.notes:
            raw.append((n.start, n.end, n.pitch, n.velocity, inst.is_drum))
    raw.sort(key=lambda x: x[0])

    if not raw:
        return Skeleton(0, [], [], [], [], False, [])

    # 화음 묶기: 거의 동시에 시작한 음을 한 '음절 후보'로
    merged = []
    group = [raw[0]]
    for n in raw[1:]:
        if n[0] - group[-1][0] < merge_chord_window:
            group.append(n)
        else:
            merged.append(group)
            group = [n]
    merged.append(group)

    onsets, durations, pitches, strengths, drum_flags = [], [], [], [], []
    for g in merged:
        start = min(x[0] for x in g)
        end = max(x[1] for x in g)
        onsets.append(start)
        durations.append(end - start)
        pitches.append(int(np.mean([x[2] for x in g])))
        strengths.append(float(np.mean([x[3] for x in g])) / 127.0)
        drum_flags.append(any(x[4] for x in g))

    # 쉼표(무음 구간) 검출 → 띄어쓰기 후보 (음색 무관, 가장 신뢰)
    rest_gaps = []
    syllable_groups = [0]
    cur_group = 0
    for i in range(1, len(onsets)):
        gap = onsets[i] - (onsets[i - 1] + durations[i - 1])
        if gap > rest_threshold:
            rest_gaps.append((onsets[i - 1] + durations[i - 1], onsets[i]))
            cur_group += 1
        syllable_groups.append(cur_group)

    # [3] 타악적 판정: 드럼이거나, 음 길이 중앙값이 짧으면 타악적
    median_dur = float(np.median(durations)) if durations else 1.0
    is_percussive = (sum(drum_flags) > len(drum_flags) / 2) or \
                    (median_dur < percussive_dur_thresh)

    return Skeleton(
        onset_count=len(onsets),
        durations=durations,
        rest_gaps=rest_gaps,
        pitches=pitches,
        onset_strengths=strengths,
        is_percussive=is_percussive,
        syllable_groups=syllable_groups,
    )


# ============================================================================
# 4. 합의도 보정 ([2] 핵심: 물리 onset → 사람 글자수)
#    - 5명 주석에서 학습한 (onset_count → 글자수) 매핑을 적용.
#    - 매핑이 없으면 보정 없이 onset_count를 그대로 쓰되 '추정' 플래그.
# ============================================================================

@dataclass
class ConsensusModel:
    """
    5명 주석에서 측정한 grouping 보정 모델.
    onset_count → 사람이 실제로 쓴 글자수의 기대값 + 신뢰도(합의도).

    measure_from_annotations()로 datastore에서 학습한다.
    """
    # onset_count 구간별 (평균 글자수 비율, 합의도) 테이블
    ratio_by_onset: dict[int, tuple[float, float]] = field(default_factory=dict)
    global_ratio: float = 1.0       # 전체 평균 글자/음표 비율
    global_agreement: float = 0.5   # 전체 평균 합의도
    diagnostics: dict = field(default_factory=dict)  # 측정 품질 진단 통계

    def expected_char_count(self, onset_count: int) -> tuple[int, float]:
        """onset_count → (보정된 글자수, 합의도)."""
        if onset_count == 0:
            return 0, self.global_agreement
        if onset_count in self.ratio_by_onset:
            ratio, agree = self.ratio_by_onset[onset_count]
        else:
            ratio, agree = self.global_ratio, self.global_agreement
        return max(1, round(onset_count * ratio)), agree


def measure_consensus(annotations: list[dict],
                      collapse_long: bool = True) -> ConsensusModel:
    """
    조인된 레코드에서 합의도/매핑 측정. (data_io.build_annotation_records 출력)

    annotations: [{ "onset_count": int | None,   # 골격 (None이면 매칭 실패)
                    "transcripts": [str, ...],     # 채워진 5명
                    "annotators": [str, ...],      # 주석자 ID (U9608 등)
                    "song_id": str }]              # 그룹 분리용

    측정 내용 (축 분리):
      - 음절 합의도: 채워진 주석자 전원 (장음은 collapse_long으로 축약)
      - 띄어쓰기 합의도: 공백을 쓴 주석자만 (U9608류 미표기자 제외)
      - 솔페지오 주석자: 감지해 글자수/비율 집계에서 제외 (음높이 전사라 다름)
      - 글자수 비율: 사람 글자수 / onset_count (onset 있는 경우만)
      ※ song_id가 곧 그룹이므로, 그룹마다 다른 5명이라는 점을 감안해
        per_onset 비율은 그룹 편향을 평균으로 흡수한다(향후 그룹별 분리 가능).
    """
    per_onset = {}        # onset_count -> ([ratios], [syllable_agreements])
    all_ratios, all_agrees = [], []
    # 진단용 통계 (반환 객체에 부가)
    spacing_agreements = []
    n_no_spacing_annotators = 0
    n_solfege = 0
    n_unmatched = 0

    for a in annotations:
        raw_texts = a.get("transcripts", [])
        if not raw_texts:
            continue

        # 솔페지오 주석 분리 (음색 전사가 아니므로 음절/비율 집계서 제외)
        musical_texts = []
        for t in raw_texts:
            if looks_like_solfege(t):
                n_solfege += 1
            else:
                musical_texts.append(t)
        if not musical_texts:
            continue

        # --- 음절 축: 장음 축약 카운트 (전원) ---
        char_counts = [count_syllables(t, collapse_long=collapse_long)
                       for t in musical_texts]
        char_counts = [c for c in char_counts if c > 0]
        if not char_counts:
            continue
        mean_chars = float(np.mean(char_counts))
        syll_agreement = (1.0 / (1.0 + float(np.std(char_counts)))
                          if len(char_counts) > 1 else 0.5)

        # --- 띄어쓰기 축: 공백 쓴 주석자만 ---
        spacing_breaks = []
        for t in musical_texts:
            if " " in t.strip():
                spacing_breaks.append(count_spacing_breaks(t))
            else:
                n_no_spacing_annotators += 1
        if len(spacing_breaks) > 1:
            spacing_agreements.append(1.0 / (1.0 + float(np.std(spacing_breaks))))

        # --- onset과의 비율 (골격 매칭된 경우만) ---
        oc = a.get("onset_count")
        if oc is None:
            n_unmatched += 1
        elif oc > 0:
            ratio = mean_chars / oc
            per_onset.setdefault(oc, ([], []))
            per_onset[oc][0].append(ratio)
            per_onset[oc][1].append(syll_agreement)
            all_ratios.append(ratio)
            all_agrees.append(syll_agreement)

    ratio_by_onset = {
        oc: (float(np.mean(rs)), float(np.mean(ags)))
        for oc, (rs, ags) in per_onset.items()
    }
    model = ConsensusModel(
        ratio_by_onset=ratio_by_onset,
        global_ratio=float(np.mean(all_ratios)) if all_ratios else 1.0,
        global_agreement=float(np.mean(all_agrees)) if all_agrees else 0.5,
    )
    # 진단 통계 부착 (측정 품질 확인용, 생성에는 안 쓰임)
    model.diagnostics = {
        "mean_spacing_agreement": (float(np.mean(spacing_agreements))
                                   if spacing_agreements else None),
        "no_spacing_annotator_count": n_no_spacing_annotators,
        "solfege_transcript_count": n_solfege,
        "unmatched_clip_count": n_unmatched,
        "onset_buckets_learned": len(ratio_by_onset),
    }
    return model





# ============================================================================
# 5. 한글 음절 카운트 / 정규화 (국어원 음절 단위 규칙)
#    실제 데이터의 장음은 모음 반복형: "두우우우", "니이이이이", "바아안"
#    → 원시 음절수(raw)와 장음 축약 음절수(collapsed) 둘 다 제공.
#      합의도 측정 시 어느 축이 사람 지각과 잘 맞는지 비교할 수 있게 한다.
# ============================================================================

# 한글 음절 → (초성, 중성, 종성) 분해는 decompose_syllable (아래 정의) 사용

def count_syllables(text: str, collapse_long: bool = False) -> int:
    """한글 완성형 음절 수.
    collapse_long=False: 표면 음절 그대로 카운트 ("두우우우"=4)
    collapse_long=True : 연장 모음을 1음으로 축약 ("두우우우"=1, "라라라"=3 유지)
    띄어쓰기는 무시 (음절 축만 카운트)."""
    text = normalize_transcript(text)
    sylls = [ch for ch in text if 0xAC00 <= ord(ch) <= 0xD7A3]
    if not collapse_long:
        return len(sylls)
    return _collapse_long_vowels(sylls)


def _collapse_long_vowels(sylls: list[str]) -> int:
    """연장 모음 축약: 앞 음절의 모음이 'ㅇ+같은모음'으로 이어지면 한 음으로 본다.
    예) '두','우','우','우' → 첫 '두'의 모음 ㅜ가 'ㅇㅜ(우)'로 연장된 것 → 1음.
        '라','라','라' → 자음 ㄹ이 매번 새로 와서 연장 아님 → 3음 유지."""
    if not sylls:
        return 0
    count = 1
    prev_cho, prev_jung, _ = decompose_syllable(sylls[0])
    for s in sylls[1:]:
        cho, jung, _ = decompose_syllable(s)
        # 'ㅇ' 초성 + 직전과 같은 모음 = 연장으로 간주 (새 음절 아님)
        is_extension = (cho == "ㅇ" and jung == prev_jung)
        if not is_extension:
            count += 1
        prev_cho, prev_jung = cho, jung
    return count


def normalize_transcript(text: str) -> str:
    """장음/연장 기호 정규화.
    - 명시적 연장 기호(~, ー, —)는 제거 (길이는 별도 채널에서 다룸)
    - 모음 반복형 장음('두우우우')은 여기서 건드리지 않고,
      카운트 단계(collapse_long)에서 처리한다.
    - 공백은 띄어쓰기 축에서 따로 쓰므로 음절 카운트용으로는 무시됨."""
    for mark in ["~", "ー", "—"]:
        text = text.replace(mark, "")
    return text.strip()


def count_spacing_breaks(text: str) -> int:
    """띄어쓰기(공백)로 나뉜 그룹 경계 수. 띄어쓰기 합의도 축에 사용.
    공백이 전혀 없으면 0 (U9608 같은 미표기자 감지)."""
    parts = [p for p in text.split() if p.strip()]
    return max(0, len(parts) - 1)


def looks_like_solfege(text: str) -> bool:
    """솔페지오(도레미·계이름) 전사 감지. U6936 '레도레도 레에에' 류.
    음색 모방이 아니라 음높이 전사라 자모음 분포가 다르므로 플래그한다."""
    t = normalize_transcript(text).replace(" ", "")
    if not t:
        return False
    solfege = set("도레미파솔라시")
    hits = sum(1 for ch in t if ch in solfege)
    return hits / max(1, len(t)) > 0.6  # 60% 이상이 계이름이면 솔페지오





# ============================================================================
# 6. 음절 생성 ([1] 해결: random → 검색 예시로 가중된 선택, 모드 B)
# ============================================================================

@dataclass
class RetrievalExamples:
    """검색된 유사 음원들의 실제 전사 (모드 B의 핵심 신호)."""
    transcripts: list[str] = field(default_factory=list)

    def char_frequency(self) -> Counter:
        """예시 전사에서 음절별 등장 빈도. 자모음 선택을 이쪽으로 끌어당김."""
        cnt = Counter()
        for t in self.transcripts:
            for ch in normalize_transcript(t):
                if 0xAC00 <= ord(ch) <= 0xD7A3:
                    cnt[ch] += 1
        return cnt


def get_instrument_style(instrument: str, is_percussive: bool) -> dict:
    """
    [3] 악기 라벨 + 실제 음색으로 스타일 결정.
    선율악기라도 타악적으로 연주되면(is_percussive) 코나콜/볼 후보를 섞는다.
    """
    key = _LABEL_TO_STYLE.get(instrument.lower(), "default")
    style = dict(INSTRUMENT_STYLE[key])  # 복사

    # 타악적 음색이면 코나콜/볼 음절을 후보 풀에 합류 (한정 반영)
    if is_percussive:
        style = dict(style)
        style["consonants"] = list(set(style["consonants"] + KONNAKOL_BOL["consonants"]))
        style["vowels"] = list(set(style["vowels"] + KONNAKOL_BOL["vowels"]))
        style["tails"] = list(set(style["tails"] + KONNAKOL_BOL["tails"]))
    return style


def _weighted_pick_consonant_vowel(style: dict,
                                   examples: RetrievalExamples,
                                   rng: np.random.Generator) -> tuple[str, str]:
    """
    [1] random 대신: guideline 풀을 후보로 두되,
    검색 예시에서 자주 쓰인 자모음에 가중치를 준다 (모드 B).

    우선순위: 검색 예시 일관성 > guideline 풀.
    예시가 비면 guideline 풀에서 균등 선택으로 폴백.
    """
    freq = examples.char_frequency()

    # 예시에 등장한 음절을 초성/중성으로 분해해 자모 빈도 집계
    cho_w = {c: 1.0 for c in style["consonants"]}  # guideline prior (약하게 1.0)
    jung_w = {v: 1.0 for v in style["vowels"]}
    for syl, n in freq.items():
        cho, jung, _ = decompose_syllable(syl)
        if cho in cho_w:
            cho_w[cho] += n * 2.0   # 예시 신호에 더 큰 가중 (예시 우선)
        if jung in jung_w:
            jung_w[jung] += n * 2.0

    cons = _weighted_choice(cho_w, rng)
    vow = _weighted_choice(jung_w, rng)
    return cons, vow


def _weighted_choice(weights: dict, rng: np.random.Generator) -> str:
    keys = list(weights.keys())
    w = np.array([weights[k] for k in keys], dtype=float)
    w = w / w.sum()
    return keys[int(rng.choice(len(keys), p=w))]


def decompose_syllable(syl: str) -> tuple[str, str, str]:
    """한글 음절 → (초성, 중성, 종성)."""
    if not (0xAC00 <= ord(syl) <= 0xD7A3):
        return ("", "", "")
    code = ord(syl) - 0xAC00
    ci = code // (21 * 28)
    ji = (code % (21 * 28)) // 28
    ki = code % 28
    return (CHOSEONG[ci], JUNGSEONG[ji], JONGSEONG[ki])


def generate_syllable(pitch_level: str, rhythm: str, attack: str,
                      envelope: str, style: dict,
                      examples: RetrievalExamples,
                      rng: np.random.Generator) -> str:
    """
    한 음절 생성. Algorithm 2의 GenerateSyllable을 모드 B로 수정.

    유지한 골조: pitch=low→받침, rhythm=fast+connector→ㄹ,
                 high+soft+swell→ㅎ(air_marker)
    교체: random 선택 → 예시 가중 선택
    추가: 받침은 국어원 7종성 중화 규칙으로 정규화 (compose_syllable 내부)
    """
    cons, vow = _weighted_pick_consonant_vowel(style, examples, rng)

    tail = ""
    if pitch_level == "low":
        # 받침 선택도 예시 가중 (간략화: 풀에서 빈도 가중)
        tail_w = {t: 1.0 for t in style["tails"]}
        freq = examples.char_frequency()
        for syl, n in freq.items():
            _, _, jong = decompose_syllable(syl)
            if jong in tail_w:
                tail_w[jong] += n * 2.0
        tail = _weighted_choice(tail_w, rng)

    syllable = compose_syllable(cons, vow, tail)  # 7종성 중화 적용됨

    # connector: 빠른 리듬 + 찰현류 → 'ㄹ' (음 사이 미끄러짐 표현)
    connector = ""
    if rhythm == "fast" and style["use_connector"]:
        connector = "르"  # ㄹ 단독은 음절이 안 되므로 '르'로 표기

    # air_marker: 높은 음 + 부드러운 어택 + 부푸는 엔벨로프 → 'ㅎ' 숨소리
    air = ""
    if pitch_level == "high" and attack == "soft" and envelope == "swelled":
        air = "흐"

    return syllable + connector + air


# ============================================================================
# 7. 전체 조립 (SoundToText) — 모든 수정 통합
# ============================================================================

def sound_to_text(skeleton: Skeleton,
                  instrument: str,
                  consensus: ConsensusModel,
                  examples: RetrievalExamples,
                  loudness: str = "normal",
                  seed: int = 0) -> dict:
    """
    골격 + guideline(모드B) + 합의도 + 검색예시 → 스캣 전사.

    반환: {
      "text": "다다담 부웅",            # 띄어쓰기 포함 최종 전사
      "syllable_meta": [...],           # 글자별 (pitch, duration, strength) — 시각화 모듈용
      "char_count": int,                # 보정된 글자수
      "agreement": float,               # 이 음원의 합의도 (신뢰도)
      "is_estimated": bool,             # 합의도 매핑 없이 추정했는가
    }

    [2] 글자수 = 합의도 보정된 값 (duration*2.5 아님)
    띄어쓰기 = skeleton.rest_gaps 기반 (음색 무관, 가장 신뢰)
    시각화는 여기서 안 함 — syllable_meta만 내보내고 별도 모듈이 렌더링
    """
    rng = np.random.default_rng(seed)
    style = get_instrument_style(instrument, skeleton.is_percussive)

    # [2] 글자수: 물리 onset → 사람 글자수로 보정
    char_count, agreement = consensus.expected_char_count(skeleton.onset_count)
    is_estimated = skeleton.onset_count not in consensus.ratio_by_onset

    # onset 단위로 음절 생성하되, 합의도 보정 글자수에 맞춰 grouping
    # (보정 글자수 < onset이면 빠른 음들을 묶음 = 사람 grouping 근사)
    syllables = []
    syllable_meta = []
    n = skeleton.onset_count

    # onset → 글자 매핑 인덱스 (균등 분배로 묶기)
    if n > 0:
        assign = np.floor(np.linspace(0, char_count, n, endpoint=False)).astype(int)
    else:
        assign = np.array([], dtype=int)

    for target_idx in range(char_count):
        members = [i for i in range(n) if assign[i] == target_idx]
        if not members:
            members = [min(target_idx, n - 1)] if n > 0 else []
        if not members:
            break

        # 이 글자의 대표 음향 특성
        rep = members[len(members) // 2]
        pitch = skeleton.pitches[rep] if skeleton.pitches else 60
        dur = sum(skeleton.durations[i] for i in members)
        strength = float(np.mean([skeleton.onset_strengths[i] for i in members]))

        pitch_level = "low" if pitch < 52 else "high" if pitch > 72 else "mid"
        rhythm = "fast" if dur < 0.25 else "slow"
        attack = "soft" if strength < 0.4 else "hard"
        envelope = "swelled" if (dur > 0.6 and attack == "soft") else "flat"

        syl = generate_syllable(pitch_level, rhythm, attack, envelope,
                                style, examples, rng)

        # loudness 강조: 된소리 (repeat_emphasis 스타일에 한정)
        if loudness == "strong" and style["repeat_emphasis"]:
            syl = _emphasize_double_consonant(syl)

        # 길이 표기: 긴 음은 모음 연장 ('부' + dur 길면 '부웅' 류)
        if dur > 0.6:
            syl = _stretch_vowel(syl)

        syllables.append(syl)
        syllable_meta.append({
            "syllable": syl, "pitch": pitch, "duration": round(dur, 3),
            "strength": round(strength, 3), "onset_members": members,
        })

    # 띄어쓰기: rest_gaps가 만든 syllable_groups 경계에 공백 삽입
    text = _apply_spacing(syllables, skeleton, syllable_meta)

    return {
        "text": text,
        "syllable_meta": syllable_meta,
        "char_count": char_count,
        "agreement": round(agreement, 3),
        "is_estimated": is_estimated,
    }


def _emphasize_double_consonant(syl: str) -> str:
    """강세: 초성을 된소리로 (ㄷ→ㄸ, ㄱ→ㄲ, ㅂ→ㅃ ...)."""
    if not syl:
        return syl
    cho, jung, jong = decompose_syllable(syl[0])
    hard = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
    if cho in hard:
        return compose_syllable(hard[cho], jung, jong) + syl[1:]
    return syl


def _stretch_vowel(syl: str) -> str:
    """긴 음: 모음을 연장해 길이 표기.
    받침 없는 '부' → '부우', 받침 있는 '붐' → '부움' (연장 모음 삽입 후 받침 음절).
    첫 음절만 처리한다."""
    if not syl:
        return syl
    cho, jung, jong = decompose_syllable(syl[0])
    if not cho:
        return syl
    rest = syl[1:]  # connector/air_marker 등 뒤따르는 부분
    if jong:
        # '붐' → '부' + 'ㅇ+모음' 연장 + 'ㅇ+받침' (예: 부 우 움)
        head = compose_syllable(cho, jung, "")
        stretch = compose_syllable("ㅇ", jung, "")
        coda = compose_syllable("ㅇ", jung, jong)
        return head + stretch + coda + rest
    else:
        # '부' → '부우' (열린 음절은 모음만 한 번 더)
        head = compose_syllable(cho, jung, "")
        stretch = compose_syllable("ㅇ", jung, "")
        return head + stretch + rest


def _apply_spacing(syllables: list[str], skeleton: Skeleton,
                   syllable_meta: list[dict]) -> str:
    """rest_gaps → 띄어쓰기. 음색 무관, 가장 신뢰하는 골격 신호.
    각 글자의 onset_members(첫 멤버)가 속한 syllable_group 경계에 공백 삽입."""
    if not syllables:
        return "".join(syllables)
    char_group = []
    for m in syllable_meta:
        members = m["onset_members"]
        g = skeleton.syllable_groups[members[0]] if members else 0
        char_group.append(g)

    out = []
    for i, syl in enumerate(syllables):
        if i > 0 and char_group[i] != char_group[i - 1]:
            out.append(" ")  # 쉼표 경계 = 띄어쓰기
        out.append(syl)
    return "".join(out)


# 이 파일은 순수 로직 모듈입니다.
# 실제 데이터로 돌리는 진입점은 pipeline.py 를 사용하세요.

