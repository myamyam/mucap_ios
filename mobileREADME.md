# 로컬 MuCap 앱 테스트

서버는 `mucap/.env`의 `OPENAI_API_KEY`를 자동으로 읽습니다. 키는 응답이나 로그에 출력하지 않으며, 앱으로 전달하지 않습니다. 환경변수로 지정한 값이 파일보다 우선합니다.

## 서버 실행

프로젝트 루트에서 실행합니다.

```sh
./script/run_mobile_server.sh
```

기본 포트는 8765이며 같은 Wi-Fi에서 접근할 수 있도록 모든 인터페이스에 바인딩합니다. 신뢰하는 개발망에서 테스트하세요. 이 실행 방식은 인증·HTTPS를 갖춘 배포 서버가 아닙니다.

```sh
curl http://127.0.0.1:8765/health
# 합성된 3초 소리로 실제 모델 호출 + 전사 전용 결과 확인
mucap/.venv311/bin/python script/test_mobile_server.py --url http://127.0.0.1:8765/transcribe
```

첫 분석에는 라이브러리 로드, 폰트 캐시 및 오디오 분석 초기화 시간이 추가됩니다. 분석은 한 번에 하나만 처리합니다. 처리 중 추가 분석은 HTTP 503을 반환하며 health 요청은 별도로 응답합니다. RAG는 기본적으로 꺼져 있으며, Essentia가 없는 환경에서는 기존 librosa 기반 악기 추정 경로를 사용합니다.

## Flutter 앱 실행

`/Users/sci-sumin/Documents/flutter_test/first_flutter`에서:

```sh
# Mac: 녹음 → MuCap 분석 → 자막 시각화. 내장 햅틱은 지원하지 않음.
./tool/run_mucap_local.sh macos

# iPhone: 릴리스 모드로 실행. Mac과 같은 Wi-Fi에서 연결. 로컬 네트워크 권한 허용.
./tool/run_mucap_local.sh iPhone
```

스크립트는 Mac 테스트에는 localhost, iPhone에는 Mac의 Wi-Fi 주소를 사용합니다. 
인터페이스 자동 탐색이 맞지 않으면 `MUCAP_HOST=192.168.x.x ./tool/run_mucap_local.sh iPhone`으로 지정하세요. 여러 기기가 있으면 `iPhone` 대신 Flutter 기기 ID를 전달하세요.

## 환경 재설치

현재 `mucap/.venv311`에 Python 3.11 환경을 준비했습니다. 다른 PC에서는 Python 3.11 가상환경을 만들고 아래 의존성을 설치하세요.

```sh
python3.11 -m venv mucap/.venv311
mucap/.venv311/bin/pip install -r multimodal_caption/requirements-mobile.txt
```

실제 키는 Git에서 제외된 `mucap/.env`에 보관하세요. `.env.example`은 런타임에 읽지 않습니다.

API 분리: `/transcribe`는 WAV에서 전사만 생성합니다. `/outputs`는 JSON `{audio: base64 WAV, model, captions: [{time,end,text}]}`를 받아 특징을 다시 추출하고 시각화·햅틱만 생성합니다. 전사 모델을 재호출하지 않습니다. `/analyze`는 기존 일괄 처리를 유지합니다.

