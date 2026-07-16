#!/usr/bin/env python3
"""
MuCap 결과 햅틱 WAV 재생 UI 서버.

실행: python haptic_ui_server.py
브라우저: http://127.0.0.1:8001
"""

import io
import sys
import threading
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory

# 프로젝트 루트를 path에 넣어서 dual_play 임포트
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MUCAP_OUTPUT_DIR = PROJECT_ROOT / "mucap" / "output"
MUCAP_INPUT_DIR = PROJECT_ROOT / "mucap" / "input"
# DirectSound is used because this Windows driver fails when two WASAPI
# devices are opened concurrently (PortAudio PaErrorCode -9999).
AUDIO_DEVICE = "6"   # Realtek HD Audio 2nd output (earphones/headset)
HAPTIC_DEVICE = "9"  # US05
HAPTIC_GAIN = 1.0
# US05 is a stereo output. dual_play duplicates the mono haptic WAV to L/R,
# so a single transducer works whichever output jack/channel it is wired to.
HAPTIC_CHANNELS = 2
AUDIO_GAIN = 1.0
AUDIO_CHANNELS = 2
PLAYBACK_LOCK = threading.Lock()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = Flask(__name__, static_folder="static", static_url_path="")


def _output_file(relative_path: str, suffix: str | None = None) -> Path:
    """Resolve and validate a client-supplied path below mucap/output."""
    candidate = (MUCAP_OUTPUT_DIR / relative_path).resolve()
    output_root = MUCAP_OUTPUT_DIR.resolve()
    if candidate != output_root and output_root not in candidate.parents:
        raise ValueError("mucap/output 밖의 경로는 사용할 수 없습니다.")
    if suffix and not candidate.name.lower().endswith(suffix.lower()):
        raise ValueError(f"{suffix} 파일만 사용할 수 있습니다.")
    if not candidate.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {candidate.name}")
    return candidate


def run_dual_play(params: dict) -> tuple[bool, str]:
    """
    dual_play.main()을 주어진 params로 실행 (재생은 서버 머신에서 진행).
    Returns: (성공여부, 메시지)
    """
    a_path = (params.get("a") or "").strip()
    b_path = (params.get("b") or "").strip()
    dev_a = (params.get("dev_a") or "").strip()
    dev_b = (params.get("dev_b") or "").strip()
    if not all([a_path, b_path, dev_a, dev_b]):
        return False, "필수: WAV A 경로, WAV B 경로, 장치 A, 장치 B를 모두 입력하세요."
    # 상대 경로는 프로젝트 루트 기준
    if not Path(a_path).is_absolute():
        a_path = str((ROOT / a_path).resolve())
    if not Path(b_path).is_absolute():
        b_path = str((ROOT / b_path).resolve())

    argv = [
        "dual_play.py",
        "--a", a_path,
        "--b", b_path,
        "--dev-a", dev_a,
        "--dev-b", dev_b,
        "--gain-a", str(float(params.get("gain_a", 1))),
        "--gain-b", str(float(params.get("gain_b", 1))),
        "--start-delay-ms", str(float(params.get("start_delay_ms", 300))),
        "--channels-a", str(int(params.get("channels_a", 2))),
        "--channels-b", str(int(params.get("channels_b", 2))),
    ]
    if params.get("b_lr_move"):
        argv.extend([
            "--b-lr-move",
            "--b-lr-delay-ms", str(float(params.get("b_lr_delay_ms", 15))),
            "--b-lr-move-sec", str(float(params.get("b_lr_move_sec", 0))),
            "--b-lr-pattern", (params.get("b_lr_pattern") or "l2r").strip().lower(),
            "--b-lr-period-sec", str(float(params.get("b_lr_period_sec", 2))),
        ])
        if params.get("b_lr_hard"):
            argv.append("--b-lr-hard")
    old_argv = sys.argv
    try:
        sys.argv = argv
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            import dual_play
            dual_play.main()
            return True, buf.getvalue().strip() or "재생 완료."
        finally:
            sys.stdout = old_stdout
    except Exception as e:
        return False, str(e)
    finally:
        sys.argv = old_argv


def run_single_play(params: dict) -> tuple[bool, str]:
    """
    오디오만 / 햅틱만 단독 재생 (dual_play --a-only / --b-only).
    햅틱 단독 시 dev_a==dev_b 평균(*0.5) 경로를 타지 않음.
    """
    mode = (params.get("play_mode") or "").strip().lower()
    if mode == "audio_only":
        path = (params.get("path") or params.get("a") or "").strip()
        dev = (params.get("dev") or params.get("dev_a") or "").strip()
        if not path or not dev:
            return False, "audio_only: WAV 경로(path 또는 a)와 출력 장치(dev 또는 dev_a)가 필요합니다."
        if not Path(path).is_absolute():
            path = str((ROOT / path).resolve())
        gain = float(params.get("gain_a", params.get("gain", 1)))
        argv = [
            "dual_play.py", "--a-only",
            "--a", path,
            "--dev-a", dev,
            "--gain-a", str(gain),
            "--start-delay-ms", str(float(params.get("start_delay_ms", 300))),
            "--channels-a", str(int(params.get("channels_a", 2))),
        ]
        sr_a = int(params.get("samplerate_a", 0) or 0)
        if sr_a > 0:
            argv.extend(["--samplerate-a", str(sr_a)])
    elif mode == "haptic_only":
        path = (params.get("path") or params.get("b") or "").strip()
        dev = (params.get("dev") or params.get("dev_b") or "").strip()
        if not path or not dev:
            return False, "haptic_only: WAV 경로(path 또는 b)와 출력 장치(dev 또는 dev_b)가 필요합니다."
        if not Path(path).is_absolute():
            path = str((ROOT / path).resolve())
        gain = float(params.get("gain_b", params.get("gain", 1)))
        argv = [
            "dual_play.py", "--b-only",
            "--b", path,
            "--dev-b", dev,
            "--gain-b", str(gain),
            "--start-delay-ms", str(float(params.get("start_delay_ms", 300))),
            "--channels-b", str(int(params.get("channels_b", 2))),
        ]
        sr_b = int(params.get("samplerate_b", 0) or 0)
        if sr_b > 0:
            argv.extend(["--samplerate-b", str(sr_b)])
        if params.get("b_lr_move"):
            argv.extend([
                "--b-lr-move",
                "--b-lr-delay-ms", str(float(params.get("b_lr_delay_ms", 15))),
                "--b-lr-move-sec", str(float(params.get("b_lr_move_sec", 0))),
                "--b-lr-pattern", (params.get("b_lr_pattern") or "l2r").strip().lower(),
                "--b-lr-period-sec", str(float(params.get("b_lr_period_sec", 2))),
            ])
            if params.get("b_lr_hard"):
                argv.append("--b-lr-hard")
    else:
        return False, "play_mode는 audio_only 또는 haptic_only 여야 합니다."

    old_argv = sys.argv
    try:
        sys.argv = argv
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            import dual_play
            dual_play.main()
            return True, buf.getvalue().strip() or "재생 완료."
        finally:
            sys.stdout = old_stdout
    except Exception as e:
        return False, str(e)
    finally:
        sys.argv = old_argv


@app.route("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.route("/index2.html")
@app.route("/index2")
def index2():
    """프리셋 버튼 UI: presets.json 기준으로 버튼만 누르면 재생."""
    return send_from_directory(ROOT / "static", "index2.html")


@app.route("/index3.html")
@app.route("/index3")
def index3():
    """프리셋별 오디오/햅틱 분리 재생 UI (Mac 스피커 + US05 등)."""
    return send_from_directory(ROOT / "static", "index3.html")


def _load_presets():
    fp = ROOT / "assets" / "presets.json"
    if not fp.is_file():
        return None, "presets.json not found"
    import json
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("presets", []), None


def _load_user_groups():
    fp = ROOT / "assets" / "user_groups.json"
    if not fp.is_file():
        return None, "user_groups.json not found"
    import json
    with open(fp, encoding="utf-8") as f:
        return json.load(f), None


def _reorder_presets_by_group(presets: list, group_config: dict) -> list:
    """group_config의 preset_order 순서대로 presets 재정렬. order에 없는 항목은 뒤에 붙임."""
    order = group_config.get("preset_order") or []
    by_name = {p.get("name"): p for p in presets if p.get("name")}
    ordered = []
    seen = set()
    for name in order:
        if name in by_name and name not in seen:
            ordered.append(by_name[name])
            seen.add(name)
    for p in presets:
        if p.get("name") not in seen:
            ordered.append(p)
    return ordered


@app.route("/api/presets")
def api_presets():
    """presets.json 반환 (index2 버튼용). 쿼리 user=숫자 이면 해당 그룹의 preset_order로 재정렬."""
    presets, err = _load_presets()
    if err:
        return jsonify({"ok": False, "presets": [], "error": err}), 404
    user_param = request.args.get("user", "").strip()
    if user_param:
        try:
            user_id = int(user_param)
        except ValueError:
            user_id = None
        if user_id is not None:
            ug, ug_err = _load_user_groups()
            if not ug_err and ug:
                groups = ug.get("groups") or []
                group_config = None
                for g in groups:
                    p_nums = g.get("p_numbers") or g.get("user_ids") or []
                    if user_id in p_nums:
                        group_config = g
                        break
                if group_config is None:
                    default_id = ug.get("default_group_id")
                    for g in groups:
                        if g.get("group_id") == default_id:
                            group_config = g
                            break
                if group_config:
                    presets = _reorder_presets_by_group(presets, group_config)
    return jsonify({"ok": True, "presets": presets})


@app.route("/api/devices")
def api_devices():
    """dual_play --list 출력 반환 (장치 목록)."""
    old_argv = sys.argv
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.argv = ["dual_play.py", "--list"]
        sys.stdout = buf
        import dual_play
        dual_play.main()
        return jsonify({"ok": True, "text": buf.getvalue()})
    except Exception as e:
        return jsonify({"ok": False, "text": str(e)})
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout


@app.route("/api/results")
def api_results():
    """Return MuCap result folders that contain rendered MP4s or haptic WAVs."""
    results = []
    if MUCAP_OUTPUT_DIR.is_dir():
        for output_dir in sorted(p for p in MUCAP_OUTPUT_DIR.iterdir() if p.is_dir()):
            videos = sorted(output_dir.glob("*-audio_text_viz_direct.mp4"))
            haptics = sorted(output_dir.glob("*-haptic.wav"))
            results.append({
                "name": output_dir.name,
                "videos": [p.relative_to(MUCAP_OUTPUT_DIR).as_posix() for p in videos],
                "haptic": (
                    haptics[0].relative_to(MUCAP_OUTPUT_DIR).as_posix()
                    if haptics else None
                ),
                "complete": bool(videos and haptics),
            })
    return jsonify({"ok": True, "results": results})


@app.route("/api/media")
def api_media():
    """Serve an MP4 below mucap/output to the local video element."""
    try:
        media_path = _output_file(request.args.get("path", ""), ".mp4")
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return send_file(media_path, conditional=True)


@app.route("/api/play-result", methods=["POST"])
def api_play_result():
    """Play original music on A and haptic-only audio on B in sync."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        haptic_path = _output_file(data.get("path", ""), "-haptic.wav")
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400

    source_wav = MUCAP_INPUT_DIR / f"{haptic_path.parent.name}.wav"
    if not source_wav.is_file():
        return jsonify({
            "ok": False,
            "message": f"원본 WAV가 없습니다: {source_wav.name}",
        }), 400

    if not PLAYBACK_LOCK.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "message": "이미 재생 중입니다. 재생이 끝난 뒤 다시 눌러주세요.",
        }), 409

    import dual_play
    dual_play.PLAYBACK_STOP_EVENT.clear()
    dual_play.PLAYBACK_PAUSE_EVENT.clear()
    try:
        ok, message = run_dual_play({
        "a": str(source_wav),
        "b": str(haptic_path),
        "dev_a": AUDIO_DEVICE,
        "dev_b": HAPTIC_DEVICE,
        "gain_a": AUDIO_GAIN,
        "gain_b": HAPTIC_GAIN,
        "channels_a": AUDIO_CHANNELS,
        "channels_b": HAPTIC_CHANNELS,
        "start_delay_ms": float(data.get("start_delay_ms", 300)),
        })
    finally:
        PLAYBACK_LOCK.release()
    return jsonify({"ok": ok, "message": message})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    """Pause A/B without advancing their playback positions."""
    import dual_play
    dual_play.PLAYBACK_PAUSE_EVENT.set()
    return jsonify({"ok": True, "message": "일시정지했습니다."})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    """Resume A/B from their retained playback positions."""
    import dual_play
    if dual_play.PLAYBACK_STOP_EVENT.is_set():
        return jsonify({"ok": False, "message": "정지된 재생은 이어갈 수 없습니다."}), 409
    dual_play.PLAYBACK_PAUSE_EVENT.clear()
    return jsonify({"ok": True, "message": "재생을 이어갑니다."})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop A/B and clear any paused state."""
    import dual_play
    dual_play.PLAYBACK_PAUSE_EVENT.clear()
    dual_play.PLAYBACK_STOP_EVENT.set()
    return jsonify({"ok": True, "message": "재생을 정지했습니다."})


@app.route("/api/run", methods=["POST"])
def api_run():
    """dual_play 재생 (메인 UI). play_mode=audio_only|haptic_only 이면 단독 재생."""
    data = request.get_json(force=True, silent=True) or {}
    mode = (data.get("play_mode") or "").strip().lower()
    if mode in ("audio_only", "haptic_only"):
        ok, message = run_single_play(data)
    else:
        ok, message = run_dual_play(data)
    return jsonify({"ok": ok, "message": message})


@app.route("/api/download")
def api_download():
    """쿼리: path=파일경로 (서버 내 상대경로)"""
    rel = request.args.get("path", "").strip()
    if not rel or ".." in rel:
        return jsonify({"error": "invalid path"}), 400
    fp = ROOT / rel
    if not fp.is_file():
        return jsonify({"error": "file not found"}), 404
    return send_file(fp, as_attachment=True, download_name=fp.name)


if __name__ == "__main__":
    (ROOT / "static").mkdir(exist_ok=True)
    print("MuCap 결과 및 햅틱 재생 UI: http://127.0.0.1:8001")
    print(f"MuCap output: {MUCAP_OUTPUT_DIR}")
    print(f"Audio device: {AUDIO_DEVICE} (Realtek earphones/headset)")
    print(f"Haptic device: {HAPTIC_DEVICE} (US05, same signal on L/R)")
    app.run(host="127.0.0.1", port=8001, debug=False, threaded=True)
