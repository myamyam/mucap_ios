# WAV → AHAP 테스트

프로젝트 루트에서 실행합니다. Python 3.10 이상이 필요합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy soundfile librosa

python -m multimodal_caption.wav_to_ahap --in /path/to/input.wav
python -m multimodal_caption.wav_to_ahap --in /path/to/input.wav --out /path/to/output.ahap
```

`python multimodal_caption/wav_to_ahap.py ...` 방식도 지원합니다.
출력 경로를 생략하면 입력 WAV 옆에 같은 이름의 `.ahap`을 저장합니다.
지정한 출력 파일이 이미 있으면 덮어씁니다. 입력 WAV는 보존합니다.

```sh
python -m multimodal_caption.wav_to_ahap --in input.wav --preset percussive --gain 0.7
python -m multimodal_caption.wav_to_ahap --in input.wav --preset melodic --strength 1.3
```

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--preset` | `ambience` | `generic`, `ambience`, `melodic`, `percussive` |
| `--quality` | `mid` | 기존 STFT 설정: `low`, `mid`, `high` |
| `--strength` | `1.0` | 기존 auto policy의 remap 혼합 비율 |
| `--gain` | `1.0` | 최종 AHAP 강도 배율. 전체 진동 세기 조절에 사용 |
| `--reference-rms` | `0.25` | 강도 1에 대응하는 고정 PCM RMS 기준 |
| `--control-hz` | `100` | 초당 제어점 수, 1–200 |
| `--curve-tolerance` | `0.01` | 제어점 축약의 최대 절대 보간 오차(0–1 척도). `0`이면 축약하지 않음 |

처리 흐름은 `WAV → 기존 hybrid 분석/합성 → 30 ms RMS 강도 + Mel 기반 sharpness → AHAP`입니다.
모노 변환과 48 kHz 리샘플링을 수행합니다. 기존의 필터, Mel envelope, gamma/gate,
auto remap, RMS matching, limiter를 재사용합니다. 한 STFT 창보다 짧은 입력은
분석용으로만 zero padding하며 출력 길이는 원본 길이로 유지합니다.
필터에 필요한 프레임이 부족한 짧은 입력에서는 Mel envelope smoothing을 생략합니다.

강도는 합성 결과의 RMS에서 계산합니다. Sharpness는 Mel 대역 위치의 에너지 가중 평균에
remap/저역 에너지 비율을 반영한 실험용 매핑입니다. 별도 주파수 파형을 재현하지는 않습니다.
기존 RMS matching 때문에 파일 간 강도에는 원본 RMS도 반영됩니다.
출력 곡선에는 시작/끝 fade를 적용하며 정확한 무음과 DC 입력은 무음으로 내보냅니다.
이벤트는 30초 이하, 곡선은 최대 16개 제어점으로 나누고 곡선 경계의 값을 공유합니다.
기본적으로 100 Hz로 계산한 뒤 선형 보간 오차가 0.01 이내인 제어점을 줄입니다.
오차는 원래 제어점 시각에서 검증하며 실제 촉감의 동일성을 보장하지 않습니다.
밀집된 곡선의 실기기 재생에서 발생한 -4899 오류에 대응하기 위한 축약으로,
축약된 파일의 재생 가능 여부도 대상 iPhone에서 확인해야 합니다.

AHAP에는 햅틱만 포함합니다. 원음 WAV 재생과 동기화는 앱 연동 시 구현해야 합니다.
기존 `hybrid_test.py` CLI와 WAV 출력은 그대로 사용할 수 있습니다.
프리셋 객체는 복사해서 사용하므로 연속 변환에서 설정이 누적되지 않습니다.

검증:

```sh
python -m unittest discover -s tests -p 'test_wav_to_ahap.py' -v
```

Python 검증은 신호 특성, JSON 구조, 시간과 값의 범위, 경계 연속성을 확인합니다.
실제 Core Haptics 파싱과 촉감은 지원 iPhone에서 별도 확인해야 합니다.

형식 참고: [Apple AHAP 문서](https://developer.apple.com/documentation/corehaptics/representing-haptic-patterns-in-ahap-files).
