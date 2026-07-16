#!/usr/bin/env python3
"""
Dual-device WAV player for macOS (or any PortAudio-supported OS).

Goal:
- Play two WAV files in sync (same start time)
  - WAV A -> output device 1 (e.g., headphones)
  - WAV B -> output device 2 (e.g., BSA30 input device / USB DAC)

Approach:
- Use sounddevice (PortAudio) OutputStream with callback per device.
- Start both streams "armed", then begin output at a shared future t0 (using silence until t0).
- Keep both streams on a common wall-clock (time.monotonic) to reduce start skew.
- (Optional) resample each file to its device stream sample rate.

Install:
  pip install sounddevice soundfile numpy scipy

Usage:
  python dual_play.py --list
  python dual_play.py --a original.wav --b haptic.wav --dev-a "<name or index>" --dev-b "<name or index>"
  # B(햅틱) 1개 WAV로 좌→우 이동감: --b-lr-move (R 지연/크로스페이드로 2ch 구성)
  python dual_play.py --a orig.wav --b haptic.wav --dev-a 0 --dev-b 1 --b-lr-move

Notes:
- This gives good start sync. Long-term drift can still occur if devices have independent clocks.
  For typical short clips (seconds to a few minutes), it's usually acceptable.
"""

import argparse
import time
import threading
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import soundfile as sf
import sounddevice as sd


# Process-wide controls used by haptic_ui_server. Pausing outputs silence
# without advancing either stream index, so A and B resume together.
PLAYBACK_PAUSE_EVENT = threading.Event()
PLAYBACK_STOP_EVENT = threading.Event()
from scipy.signal import resample_poly


def list_output_devices():
    devs = sd.query_devices()
    rows = []
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0:
            rows.append((i, d["name"], d["hostapi"], d.get("default_samplerate", None), d["max_output_channels"]))
    print("Output devices:")
    for i, name, hostapi, dsr, ch in rows:
        print(f"  [{i}] {name} | hostapi={hostapi} | default_sr={dsr} | max_out_ch={ch}")


def resolve_device(dev: str) -> int:
    # Accept numeric index or substring match in name
    if dev.isdigit():
        return int(dev)
    devs = sd.query_devices()
    cand = []
    for i, d in enumerate(devs):
        if d.get("max_output_channels", 0) > 0 and dev.lower() in d["name"].lower():
            cand.append((i, d["name"]))
    if not cand:
        raise ValueError(f"Could not find output device matching: {dev!r}")
    if len(cand) > 1:
        names = ", ".join([f"[{i}] {n}" for i, n in cand[:10]])
        raise ValueError(f"Ambiguous device match for {dev!r}. Matches: {names}")
    return cand[0][0]


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def to_stereo(x: np.ndarray) -> np.ndarray:
    # shape (N,) -> (N,2)
    return np.column_stack([x, x]).astype(np.float32)


def make_lr_haptic(
    x: np.ndarray,
    sr: int,
    delay_ms: float = 8.0,
    move_sec: float = 1.0,
    pattern: str = "l2r",
    period_sec: float = 2.0,
    hard: bool = False,
) -> np.ndarray:
    """
    모노 햅틱 신호를 L/R 이동감이 있는 스테레오(2ch)로 변환.

    - pattern "l2r": 왼쪽→오른쪽 한 방향 (move_sec 동안 L 1→0, R 0→1, 이후 R만 유지)
    - pattern "alternate": 좌↔우 번갈아 (period_sec마다 L→R→L 반복)
    - delay_ms: R 채널 지연 (촉감 이동감에 자주 사용)
    - hard: True면 크로스페이드 없이 한쪽만 1 (이동 중에도 두 채널 동시에 안 나옴)

    Returns:
        (N, 2) float32, [L, R]
    """
    x = x.astype(np.float32)

    # R 채널 지연
    d = int(sr * (delay_ms / 1000.0))
    x_r = np.pad(x, (d, 0), mode="constant", constant_values=0)[: len(x)]
    x_l = x

    n = len(x)
    t = np.arange(n, dtype=np.float32) / sr  # 시간(초)

    if pattern == "alternate":
        period_samples = max(1, int(sr * period_sec))
        phase = (np.arange(n, dtype=np.float32) % period_samples) / period_samples
        if hard:
            wL = np.where(phase < 0.5, 1.0, 0.0).astype(np.float32)
        else:
            wL = np.where(phase < 0.5, 1.0 - 2.0 * phase, 2.0 * phase - 1.0)
        wR = 1.0 - wL
    else:
        move_n = max(1, int(sr * move_sec))
        if hard:
            # 전반부 L만, 후반부 R만 (크로스페이드 없음)
            wL = np.zeros(n, dtype=np.float32)
            wL[: move_n // 2] = 1.0
            wR = 1.0 - wL
        else:
            ramp = np.linspace(0.0, 1.0, min(move_n, n), dtype=np.float32)
            wL = np.zeros(n, dtype=np.float32)
            wL[: len(ramp)] = 1.0 - ramp
            wR = 1.0 - wL

    outL = x_l * wL
    outR = x_r * wR
    return np.column_stack([outL, outR]).astype(np.float32)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down).astype(np.float32)


@dataclass
class StreamPlayer:
    data: np.ndarray           # (N, C)
    sr: int
    channels: int
    start_wall: float          # time.monotonic() target
    blocksize: int = 0         # 0 lets PortAudio choose
    idx: int = 0               # current frame index

    def callback(self, outdata, frames, time_info, status):
        # time_info: {'inputBufferAdcTime', 'currentTime', 'outputBufferDacTime'} in seconds (PortAudio clock)
        # We gate output until wall-clock reaches start_wall to align both streams.
        if PLAYBACK_STOP_EVENT.is_set():
            outdata[:] = 0
            raise sd.CallbackStop()

        if PLAYBACK_PAUSE_EVENT.is_set():
            outdata[:] = 0
            return

        now = time.monotonic()
        if now < self.start_wall:
            outdata[:] = 0
            return

        end = self.idx + frames
        if self.idx >= len(self.data):
            outdata[:] = 0
            raise sd.CallbackStop()

        chunk = self.data[self.idx:end]
        if len(chunk) < frames:
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):] = 0
            self.idx = len(self.data)
            raise sd.CallbackStop()
        else:
            outdata[:] = chunk
            self.idx = end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list output devices and exit")
    ap.add_argument("--a", help="WAV A (original audio) path")
    ap.add_argument("--b", help="WAV B (haptic) path")
    ap.add_argument("--dev-a", help="output device for WAV A (index or substring match)")
    ap.add_argument("--dev-b", help="output device for WAV B (index or substring match)")
    ap.add_argument("--gain-a", type=float, default=1.0, help="gain for WAV A")
    ap.add_argument("--gain-b", type=float, default=1.0, help="gain for WAV B")
    ap.add_argument("--start-delay-ms", type=float, default=300.0,
                    help="schedule both outputs to start this many ms in the future (reduces skew)")
    ap.add_argument("--channels-a", type=int, default=2, choices=[1, 2], help="output channels for A")
    ap.add_argument("--channels-b", type=int, default=2, choices=[1, 2], help="output channels for B")
    ap.add_argument("--b-lr-move", action="store_true",
                    help="B(햅틱) 1개 WAV로 L→R 이동감 생성 (to_stereo 대신 make_lr_haptic 사용)")
    ap.add_argument("--b-lr-delay-ms", type=float, default=15.0,
                    help="B L→R 이동 시 R 채널 지연 ms (기본 15, 촉감에 더 잘 느껴짐)")
    ap.add_argument("--b-lr-move-sec", type=float, default=0.0,
                    help="B L→R 크로스페이드 시간(초). 0이면 클립 전체 길이에 걸쳐 이동 (l2r일 때만)")
    ap.add_argument("--b-lr-pattern", type=str, default="l2r", choices=["l2r", "alternate"],
                    help="l2r=왼쪽→오른쪽 한 방향, alternate=좌↔우 번갈아 이동")
    ap.add_argument("--b-lr-period-sec", type=float, default=2.0,
                    help="alternate일 때 한 사이클(L→R→L) 시간(초)")
    ap.add_argument("--b-lr-hard", action="store_true",
                    help="L/R 이동 시 크로스페이드 없이 한쪽만 나오기 (이동 중에도 두 스피커 동시에 안 나옴)")
    ap.add_argument("--samplerate-a", type=int, default=0, help="force samplerate for A (0=use device default)")
    ap.add_argument("--samplerate-b", type=int, default=0, help="force samplerate for B (0=use device default)")
    ap.add_argument("--a-only", action="store_true",
                    help="WAV A만 dev-a로 재생 (B 불필요). 오디오 단독 출력용.")
    ap.add_argument("--b-only", action="store_true",
                    help="WAV B만 dev-b로 재생 (A 불필요). 햅틱 단독 시 A+B 평균(*0.5) 없음.")
    args = ap.parse_args()

    if args.list:
        list_output_devices()
        return

    if args.a_only and args.b_only:
        raise SystemExit("Cannot use both --a-only and --b-only")

    # ----- 단독 재생: A만 또는 B만 (듀얼 스트림/합성 없음) -----
    if args.a_only:
        if not all([args.a, args.dev_a]):
            raise SystemExit("Required for --a-only: --a, --dev-a")
        dev_a = resolve_device(args.dev_a)
        xa, sra = load_wav_mono(args.a)
        xa *= float(args.gain_a)
        da = sd.query_devices(dev_a)
        sr_out_a = int(args.samplerate_a) if args.samplerate_a > 0 else int(da["default_samplerate"])
        xa_rs = resample_to(xa, sra, sr_out_a)
        if args.channels_a == 2:
            xa_out = to_stereo(xa_rs)
        else:
            xa_out = xa_rs.reshape(-1, 1)
        start_wall = time.monotonic() + (args.start_delay_ms / 1000.0)
        pa = StreamPlayer(data=xa_out, sr=sr_out_a, channels=args.channels_a, start_wall=start_wall)
        stream_a = sd.OutputStream(
            device=dev_a,
            samplerate=sr_out_a,
            channels=args.channels_a,
            dtype="float32",
            callback=pa.callback,
        )
        print(f"A-only: {args.a} -> dev [{dev_a}] {da['name']} @ {sr_out_a} Hz, ch={args.channels_a}")
        print(f"Scheduled start in {args.start_delay_ms:.0f} ms ...")
        with stream_a:
            while stream_a.active:
                time.sleep(0.05)
        print("Done.")
        return

    if args.b_only:
        if not all([args.b, args.dev_b]):
            raise SystemExit("Required for --b-only: --b, --dev-b")
        dev_b = resolve_device(args.dev_b)
        xb, srb = load_wav_mono(args.b)
        xb *= float(args.gain_b)
        db = sd.query_devices(dev_b)
        sr_out_b = int(args.samplerate_b) if args.samplerate_b > 0 else int(db["default_samplerate"])
        xb_rs = resample_to(xb, srb, sr_out_b)
        b_lr_move_sec_used = None
        move_sec = None
        if args.b_lr_move:
            move_sec = args.b_lr_move_sec
            if move_sec <= 0:
                move_sec = len(xb_rs) / sr_out_b
            b_lr_move_sec_used = move_sec
            hard = getattr(args, "b_lr_hard", False)
            xb_out = make_lr_haptic(
                xb_rs, sr_out_b,
                delay_ms=args.b_lr_delay_ms,
                move_sec=move_sec,
                pattern=args.b_lr_pattern,
                period_sec=args.b_lr_period_sec,
                hard=hard,
            )
            out_ch = 2
        else:
            if args.channels_b == 2:
                xb_out = to_stereo(xb_rs)
            else:
                xb_out = xb_rs.reshape(-1, 1)
            out_ch = args.channels_b
        start_wall = time.monotonic() + (args.start_delay_ms / 1000.0)
        pb = StreamPlayer(data=xb_out, sr=sr_out_b, channels=out_ch, start_wall=start_wall)
        stream_b = sd.OutputStream(
            device=dev_b,
            samplerate=sr_out_b,
            channels=out_ch,
            dtype="float32",
            callback=pb.callback,
        )
        print(f"B-only: {args.b} -> dev [{dev_b}] {db['name']} @ {sr_out_b} Hz, ch={out_ch}", end="")
        if args.b_lr_move and b_lr_move_sec_used is not None:
            pat = f"pattern={args.b_lr_pattern}"
            if args.b_lr_pattern == "alternate":
                pat += f", period={args.b_lr_period_sec}s"
            else:
                pat += f", move={b_lr_move_sec_used:.2f}s"
            hard_s = " (한쪽만, hard)" if getattr(args, "b_lr_hard", False) else ""
            print(f" | L/R 이동 ({pat}, delay={args.b_lr_delay_ms}ms){hard_s}")
        else:
            print()
        print(f"Scheduled start in {args.start_delay_ms:.0f} ms ...")
        with stream_b:
            while stream_b.active:
                time.sleep(0.05)
        print("Done.")
        return

    if not all([args.a, args.b, args.dev_a, args.dev_b]):
        raise SystemExit("Required: --a, --b, --dev-a, --dev-b (or use --list)")

    dev_a = resolve_device(args.dev_a)
    dev_b = resolve_device(args.dev_b)

    # Load files
    xa, sra = load_wav_mono(args.a)
    xb, srb = load_wav_mono(args.b)

    xa *= float(args.gain_a)
    xb *= float(args.gain_b)

    # Determine stream sample rates
    da = sd.query_devices(dev_a)
    db = sd.query_devices(dev_b)
    sr_out_a = int(args.samplerate_a) if args.samplerate_a > 0 else int(da["default_samplerate"])
    sr_out_b = int(args.samplerate_b) if args.samplerate_b > 0 else int(db["default_samplerate"])

    # Resample to device rates
    xa_rs = resample_to(xa, sra, sr_out_a)
    xb_rs = resample_to(xb, srb, sr_out_b)

    # L→R 이동 시 사용할 공통 move_sec (오디오·햅틱 동기화)
    b_lr_move_sec_used = None
    move_sec = None
    if args.b_lr_move:
        move_sec = args.b_lr_move_sec
        if move_sec <= 0:
            move_sec = len(xb_rs) / sr_out_b  # 클립 전체에 걸쳐 L→R 이동
        b_lr_move_sec_used = move_sec

    # Channel formatting
    if args.b_lr_move:
        # 오디오(A)도 햅틱과 같은 L/R 패닝 → 소리·진동 함께 이동 (l2r 또는 alternate)
        # 두 장치가 "왼쪽 스피커 / 오른쪽 스피커"일 때: L만 dev_a, R만 dev_b로 보내야
        # 한쪽만 소리/햅틱 나와야 왔다갔다 효과가 난다. 스테레오 전체를 각 장치에 보내면 둘 다 동시에 나온다.
        xb_rs_a = resample_to(xb_rs, sr_out_b, sr_out_a)  # B를 A 샘플레이트로 맞춰 길이 동기화
        n_common = min(len(xa_rs), len(xb_rs_a))
        xa_rs = xa_rs[:n_common]
        xb_rs_a = xb_rs_a[:n_common]
        hard = getattr(args, "b_lr_hard", False)
        xa_out = make_lr_haptic(
            xa_rs, sr_out_a,
            delay_ms=args.b_lr_delay_ms,
            move_sec=move_sec,
            pattern=args.b_lr_pattern,
            period_sec=args.b_lr_period_sec,
            hard=hard,
        )
        xb_out = make_lr_haptic(
            xb_rs_a, sr_out_a,
            delay_ms=args.b_lr_delay_ms,
            move_sec=move_sec,
            pattern=args.b_lr_pattern,
            period_sec=args.b_lr_period_sec,
            hard=hard,
        )
        # dev_a = 왼쪽 → L채널만 (오디오 L + 햅틱 L). dev_b = 오른쪽 → R채널만. (한쪽만 소리 나야 함)
        left_mono = (xa_out[:, 0] + xb_out[:, 0]) * 0.5
        right_mono = (xa_out[:, 1] + xb_out[:, 1]) * 0.5
        # 한 모듈에 L,R이 붙어 있으면 같은 장치(dev_a==dev_b) → 스테레오 한 스트림으로 전송
        if dev_a == dev_b:
            xa_out = np.column_stack([left_mono, right_mono]).astype(np.float32)
            xb_out = None  # 스트림 하나만 사용
            args.channels_a = 2
            args.channels_b = 1  # unused
        else:
            xa_out = left_mono.reshape(-1, 1).astype(np.float32)
            right_mono = right_mono.astype(np.float32)
            if sr_out_b != sr_out_a:
                right_mono = resample_to(right_mono, sr_out_a, sr_out_b)
            xb_out = right_mono.reshape(-1, 1).astype(np.float32)
            args.channels_a = 1
            args.channels_b = 1
    else:
        if args.channels_a == 2:
            xa_out = to_stereo(xa_rs)
        else:
            xa_out = xa_rs.reshape(-1, 1)
        if args.channels_b == 2:
            xb_out = to_stereo(xb_rs)
        else:
            xb_out = xb_rs.reshape(-1, 1)

    # Shared start time (wall-clock)
    start_wall = time.monotonic() + (args.start_delay_ms / 1000.0)

    single_device_lr = args.b_lr_move and (dev_a == dev_b)  # 한 모듈에 L,R → 스테레오 한 스트림

    pa = StreamPlayer(data=xa_out, sr=sr_out_a, channels=args.channels_a, start_wall=start_wall)
    stream_a = sd.OutputStream(
        device=dev_a,
        samplerate=sr_out_a,
        channels=args.channels_a,
        dtype="float32",
        callback=pa.callback,
    )

    if single_device_lr:
        print(f"A+B (L/R): {args.a} + {args.b} -> dev [{dev_a}] {da['name']} 스테레오(L,R) @ {sr_out_a} Hz", end="")
    else:
        pb = StreamPlayer(data=xb_out, sr=sr_out_b, channels=args.channels_b, start_wall=start_wall)
        stream_b = sd.OutputStream(
            device=dev_b,
            samplerate=sr_out_b,
            channels=args.channels_b,
            dtype="float32",
            callback=pb.callback,
        )
        print(f"A: {args.a} -> dev [{dev_a}] {da['name']} @ {sr_out_a} Hz")
        print(f"B: {args.b} -> dev [{dev_b}] {db['name']} @ {sr_out_b} Hz", end="")

    if args.b_lr_move and b_lr_move_sec_used is not None:
        pat = f"pattern={args.b_lr_pattern}"
        if args.b_lr_pattern == "alternate":
            pat += f", period={args.b_lr_period_sec}s"
        else:
            pat += f", move={b_lr_move_sec_used:.2f}s"
        hard_s = " (한쪽만, hard)" if getattr(args, "b_lr_hard", False) else ""
        print(f" | L/R 이동 ({pat}, delay={args.b_lr_delay_ms}ms){hard_s}")
        if single_device_lr:
            print("  -> 한 장치에 스테레오(L=왼쪽, R=오른쪽) 전송")
        else:
            print("  -> dev_a=왼쪽(L만 모노), dev_b=오른쪽(R만 모노) 전송")
    else:
        print()
    print(f"Scheduled start in {args.start_delay_ms:.0f} ms ...")

    if single_device_lr:
        with stream_a:
            while stream_a.active:
                time.sleep(0.05)
    else:
        with stream_a, stream_b:
            while stream_a.active or stream_b.active:
                time.sleep(0.05)

    print("Done.")


if __name__ == "__main__":
    main()
