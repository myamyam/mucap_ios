import base64
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / "mucap" / ".env")

files = [
    Path("./alarm/1-96890-A-37.wav"),
    Path("./alarm/5-223176-A-37.wav"),
    Path("./alarm/bbibbibbi.wav"),
    Path("./alarm/ppiippii.wav"),
    Path("./alarm/ppippippi.wav"),
    Path("./alarm/ttiroriro.wav"),
    # Path("./door_knock/1-81001-B-30.wav"),
    # Path("./door_knock/1-82817-A-30.wav"),
    # Path("./door_knock/1-103999-A-30.wav"),
    # Path("./door_knock/2-118625-A-30.wav"),
    # Path("./door_knock/2-133889-A-30.wav"),
    # Path("./door_knock/4-211502-A-30.wav")        
]

prompt = f"""
첨부한 {len(files)}개의 소리를 각각 다음 항목으로 정리해줘.
1. 오디오 특징: 음높이, 음색, 강약 변화, 반복 간격, 지속 시간 등
2. 카테고리
3. 어떤 상황에서 나는 소리인지 또는 사람이 해야 할 행동
4. 소리 그대로 전사(의성어 표현)

마크다운이나 설명 없이 유효한 JSON 객체 하나만 반환해줘.
최상위 키는 sounds이며, 소리 번호 순서의 배열을 넣어줘.
각 항목에는 다음 키를 모두 포함해줘:
- sound_id: 소리 번호(정수)
- audio_features: 음향 특징을 설명하는 한국어 문자열
- category: 카테고리 한국어 문자열
- situation: 가능한 상황 및 불확실성을 설명하는 한국어 문자열
- suggested_action: 조건부 행동 제안 한국어 문자열
- onomatopoeia: 소리를 표현하는 한글 의성어 문자열
직접 들리는 특징과 상황에 대한 추정을 구분해줘.
상황이나 필요한 행동을 소리만으로 판단하기 어렵다면 명시해줘.
"""

content = [{"type": "text", "text": prompt}]

for number, path in enumerate(files, 1):
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in ("wav", "mp3"):
        raise ValueError(f"WAV 또는 MP3가 필요합니다: {path}")

    content.extend([
        {"type": "text", "text": f"소리 {number}"},
        {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode((ROOT / path).read_bytes()).decode("ascii"),
                "format": fmt,
            },
        },
    ])

response = OpenAI().chat.completions.create(
    model="gpt-audio-1.5",
    modalities=["text"],
    messages=[{"role": "user", "content": content}],
    max_tokens=2500,
    store=False,
)

answer = response.choices[0].message.content or ""
result = {
    "model": response.model,
    "prompt": prompt,
    "inputs": [{"sound_id": i, "file": str(path)} for i, path in enumerate(files, 1)],
    "finish_reason": response.choices[0].finish_reason,
}
try:
    parsed = json.loads(answer)
    sounds = parsed["sounds"]
    required = {"sound_id", "audio_features", "category", "situation", "suggested_action", "onomatopoeia"}
    if not isinstance(sounds, list) or len(sounds) != len(files):
        raise ValueError("소리별 결과 개수가 입력 파일 개수와 다릅니다.")
    for i, item in enumerate(sounds, 1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("결과에 필요한 필드가 없습니다.")
        if type(item["sound_id"]) is not int or item["sound_id"] != i:
            raise ValueError("소리 번호가 입력 순서와 다릅니다.")
        if not all(isinstance(item[key], str) for key in required - {"sound_id"}):
            raise ValueError("설명 필드는 문자열이어야 합니다.")
    result.update(status="ok", sounds=sounds)
except (ValueError, KeyError, TypeError) as exc:
    # 응답이 JSON 형식과 달라도 유료 API 결과를 잃지 않도록 보존합니다.
    result.update(status="invalid_response", error=str(exc), raw_response=answer)

(ROOT / "api_description_results_alarm.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)