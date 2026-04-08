# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from vllm.config.utils import config

RealtimeSegmentationMode = Literal["fixed", "vad"]
RealtimeVadBackend = Literal["webrtc", "silero", "ten_vad", "firered"]


@config
class SpeechToTextConfig:
    """Configuration for speech-to-text models."""

    sample_rate: float = 16_000
    """Sample rate (Hz) to resample input audio to. Most speech models expect
    16kHz audio input. The input audio will be automatically resampled to this
    rate before processing."""

    max_audio_clip_s: int | None = 30
    """Maximum duration in seconds for a single audio clip without chunking.
    Audio longer than this will be split into smaller chunks if
    `allow_audio_chunking` evaluates to True, otherwise it will be rejected. 
    `None` means audio duration can be unlimited and won't be chunked."""

    overlap_chunk_second: int = 1
    """Overlap duration in seconds between consecutive audio chunks when
    splitting long audio. This helps maintain context across chunk boundaries
    and improves transcription quality at split points."""

    min_energy_split_window_size: int | None = 1600
    """Window size in samples for finding low-energy (quiet) regions to split
    audio chunks. The algorithm looks for the quietest moment within this
    window to minimize cutting through speech. Default 1600 samples ≈ 100ms
    at 16kHz. If None, no chunking will be done."""

    @property
    def allow_audio_chunking(self) -> bool:
        return (
            self.min_energy_split_window_size is not None
            and self.max_audio_clip_s is not None
        )

    # --- Realtime (WebSocket) streaming: Qwen3-ASR etc. ---

    realtime_segmentation_mode: RealtimeSegmentationMode = "vad"
    """How to split streaming audio into ASR segments: fixed-duration windows
    or voice-activity (VAD) boundaries."""

    realtime_fixed_segment_duration_s: float = 5.0
    """When ``realtime_segmentation_mode`` is ``\"fixed\"``, emit one segment
    every this many seconds of audio."""

    realtime_vad_backend: RealtimeVadBackend = "webrtc"
    """VAD backend when ``realtime_segmentation_mode`` is ``\"vad\"`` and
    wavekat-vad is installed: ``webrtc`` (webrtcvad); ``silero``, ``ten_vad``,
    and ``firered`` use ONNX models with continuous speech probabilities (16 kHz;
    frame sizes differ per backend — see wavekat-vad)."""

    realtime_vad_speech_threshold: float = 0.5
    """For Silero, TEN-VAD, and FireRedVAD: probability above this counts as
    speech. For WebRTC, values are binary (0/1); this threshold is not used."""

    realtime_vad_min_speech_ms: int = 100
    """Minimum voiced duration before a segment can start."""

    realtime_vad_min_silence_ms: int = 300
    """Minimum trailing silence duration to end a speech segment."""

    realtime_vad_max_segment_s: float = 30.0
    """Hard cap on segment length; long speech is split at this duration."""

    realtime_vad_webrtc_mode: int = 0
    """WebRTC VAD aggressiveness (0=quality .. 3=very aggressive)."""

    realtime_vad_webrtc_frame_duration_ms: int = 30
    """WebRTC frame size: 10, 20, or 30 ms."""

    realtime_min_asr_segment_s: float = 0.5
    """Segments shorter than this (seconds) are zero-padded to this length,
    matching Qwen3-ASR offline chunking heuristics."""
