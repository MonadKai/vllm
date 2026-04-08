# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The Qwen team.
"""Streaming VAD segmentation for Qwen3-ASR realtime (wavekat-vad optional)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

VAD_WORKING_SR = 16_000


def _resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    duration = len(x) / float(orig_sr)
    new_len = max(1, int(duration * target_sr))
    t_old = np.linspace(0.0, duration, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.clip(a * 32768.0, -32768, 32767).astype(np.int16)


@dataclass
class VadSegmentResult:
    audio: np.ndarray
    start_s: float
    end_s: float


class _VadFsm:
    def __init__(
        self,
        *,
        frame_duration_ms: float,
        speech_threshold: float,
        min_speech_ms: float,
        min_silence_ms: float,
        max_segment_ms: float,
        is_binary_vad: bool,
    ) -> None:
        self.frame_duration_ms = frame_duration_ms
        self.speech_threshold = speech_threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.max_segment_ms = max_segment_ms
        self.is_binary_vad = is_binary_vad

        self.in_speech = False
        self.pending_speech_ms = 0.0
        self.silence_ms = 0.0
        self.speech_ms = 0.0
        self.seg_vad_start: int | None = None

    def _is_speech(self, p: float) -> bool:
        if self.is_binary_vad:
            return p >= 0.5
        return p >= self.speech_threshold

    def step(
        self,
        prob: float,
        *,
        vad_sample_after_frame: int,
        frame_samples: int,
    ) -> tuple[int, int] | None:
        """Return ``(start_v, end_v)`` vad sample indices when a segment ends."""

        fd = self.frame_duration_ms
        speech = self._is_speech(prob)

        if not self.in_speech:
            if speech:
                self.pending_speech_ms += fd
                self.silence_ms = 0.0
                if self.pending_speech_ms >= self.min_speech_ms:
                    self.in_speech = True
                    self.seg_vad_start = vad_sample_after_frame - frame_samples
                    self.speech_ms = self.pending_speech_ms
                    self.silence_ms = 0.0
                    self.pending_speech_ms = 0.0
            else:
                self.pending_speech_ms = 0.0
            return None

        if speech:
            self.silence_ms = 0.0
            self.speech_ms += fd
        else:
            self.silence_ms += fd

        should_emit = (
            self.speech_ms >= self.max_segment_ms
            or self.silence_ms >= self.min_silence_ms
        )

        if should_emit and self.seg_vad_start is not None:
            start_v = self.seg_vad_start
            end_v = vad_sample_after_frame
            self._reset()
            return (start_v, end_v)

        return None

    def _reset(self) -> None:
        self.in_speech = False
        self.pending_speech_ms = 0.0
        self.silence_ms = 0.0
        self.speech_ms = 0.0
        self.seg_vad_start = None


def try_create_wavekat_adapter(stt: SpeechToTextConfig):
    """Returns ``(FrameAdapter or None, is_binary)``.

    ``is_binary`` is True only for WebRTC (hard 0/1); Silero, TEN-VAD, and
    FireRedVAD use continuous speech probabilities in ``[0, 1]``.
    """
    try:
        from wavekat_vad import FrameAdapter
    except ImportError:
        return None, True

    if stt.realtime_vad_backend == "webrtc":
        try:
            from wavekat_vad.backends.webrtc import WebRtcVad, WebRtcVadMode
        except ImportError:
            return None, True
        mode = WebRtcVadMode(int(stt.realtime_vad_webrtc_mode))
        fd = int(stt.realtime_vad_webrtc_frame_duration_ms)
        if fd not in (10, 20, 30):
            logger.warning(
                "webrtc frame duration %s ms invalid; using 30 ms", fd
            )
            fd = 30
        inner = WebRtcVad(
            sample_rate=VAD_WORKING_SR,
            mode=mode,
            frame_duration_ms=fd,
        )
        return FrameAdapter(inner), True
    if stt.realtime_vad_backend == "ten_vad":
        try:
            from wavekat_vad.backends.ten_vad import TenVad
        except ImportError:
            return None, True
        inner = TenVad()
        return FrameAdapter(inner), False
    if stt.realtime_vad_backend == "firered":
        try:
            from wavekat_vad.backends.firered import FireRedVad
        except ImportError:
            return None, True
        inner = FireRedVad()
        return FrameAdapter(inner), False
    if stt.realtime_vad_backend == "silero":
        try:
            from wavekat_vad.backends.silero import SileroVad
        except ImportError:
            return None, True
        inner = SileroVad(sample_rate=VAD_WORKING_SR)
        return FrameAdapter(inner), False
    return None, True


class Qwen3ASRVadSegmentBuffer:
    """Buffer incoming float32 PCM and emit speech segments from VAD."""

    def __init__(self, sampling_rate: int, stt: SpeechToTextConfig) -> None:
        self.sr = int(sampling_rate)
        self.stt = stt
        adapter, binary = try_create_wavekat_adapter(stt)
        if adapter is None:
            raise RuntimeError("wavekat_vad is not installed")

        self._adapter = adapter
        cap = self._adapter.capabilities
        frame_ms = float(cap.frame_duration_ms)
        self._frame_samples = cap.frame_size
        self._fsm = _VadFsm(
            frame_duration_ms=frame_ms,
            speech_threshold=stt.realtime_vad_speech_threshold,
            min_speech_ms=float(stt.realtime_vad_min_speech_ms),
            min_silence_ms=float(stt.realtime_vad_min_silence_ms),
            max_segment_ms=float(stt.realtime_vad_max_segment_s) * 1000.0,
            is_binary_vad=binary,
        )

        self._orig: np.ndarray = np.array([], dtype=np.float32)
        self._vad_total_samples = 0
        self._trim_orig_samples = 0

    def write_audio(self, chunk: np.ndarray) -> list[VadSegmentResult]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []

        self._orig = np.concatenate([self._orig, chunk])
        w16 = _resample_linear(chunk, self.sr, VAD_WORKING_SR)
        int16 = _float_to_int16(w16)
        probs = self._adapter.process_all(int16, VAD_WORKING_SR)

        out: list[VadSegmentResult] = []
        frame_samples = self._frame_samples

        for p in probs:
            self._vad_total_samples += frame_samples
            span = self._fsm.step(
                float(p),
                vad_sample_after_frame=self._vad_total_samples,
                frame_samples=frame_samples,
            )
            if span is not None:
                start_v, end_v = span
                seg = self._slice_segment(start_v, end_v)
                if seg is not None:
                    out.append(seg)
                self._adapter.reset()

        return out

    def _slice_segment(
        self, start_v: int, vad_end_exclusive: int
    ) -> VadSegmentResult | None:
        start_o = self._trim_orig_samples + int(
            round(start_v * (self.sr / float(VAD_WORKING_SR)))
        )
        end_o = self._trim_orig_samples + int(
            round(vad_end_exclusive * (self.sr / float(VAD_WORKING_SR)))
        )
        end_o = min(end_o, self._trim_orig_samples + len(self._orig))
        start_o = max(start_o, self._trim_orig_samples)
        if end_o <= start_o:
            return None

        audio = self._orig[
            start_o - self._trim_orig_samples : end_o - self._trim_orig_samples
        ].copy()
        audio = self._maybe_pad_min_length(audio)
        if audio.size == 0:
            return None

        start_s = start_v / float(VAD_WORKING_SR)
        end_s = vad_end_exclusive / float(VAD_WORKING_SR)
        self._trim_upto(end_o)
        return VadSegmentResult(audio=audio, start_s=start_s, end_s=end_s)

    def _maybe_pad_min_length(self, audio: np.ndarray) -> np.ndarray:
        min_s = float(self.stt.realtime_min_asr_segment_s)
        min_len = int(min_s * self.sr)
        if audio.size < min_len:
            pad = min_len - int(audio.size)
            audio = np.pad(
                audio,
                (0, pad),
                mode="constant",
                constant_values=0.0,
            ).astype(np.float32)
        return audio

    def _trim_upto(self, absolute_orig_end: int) -> None:
        cut = absolute_orig_end - self._trim_orig_samples
        if cut <= 0:
            return
        self._orig = self._orig[cut:]
        self._trim_orig_samples = absolute_orig_end

    def flush(self) -> VadSegmentResult | None:
        """Emit remaining speech or tail audio."""
        if self._orig.size == 0:
            return None

        if self._fsm.in_speech and self._fsm.seg_vad_start is not None:
            start_v = self._fsm.seg_vad_start
            end_v = max(self._vad_total_samples, start_v + 1)
            seg = self._slice_segment(start_v, end_v)
            self._fsm._reset()
            self._adapter.reset()
            return seg

        if self._orig.size > 0:
            tail = self._orig.copy()
            t0 = self._trim_orig_samples / float(self.sr)
            dur = len(tail) / float(self.sr)
            self._orig = np.array([], dtype=np.float32)
            self._trim_orig_samples += len(tail)
            tail = self._maybe_pad_min_length(tail)
            return VadSegmentResult(audio=tail, start_s=t0, end_s=t0 + dur)

        return None
