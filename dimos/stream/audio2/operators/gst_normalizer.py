#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GStreamer-based adaptive audio normalizer operator.

This normalizer works with any audio format (compressed or raw) by using
GStreamer's level and audioamplify elements for dynamic normalization.
"""

import threading
import time
from typing import Optional

import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp
from reactivex import create
from reactivex.abc import ObservableBase

from dimos.stream.audio2.gstreamer import (
    GStreamerPipelineBase,
    ensure_mainloop_running,
    release_mainloop,
)
from dimos.stream.audio2.types import AudioEvent, RawAudioEvent, CompressedAudioEvent, AudioSpec
from dimos.stream.audio2.utils import buffer_to_audio_event, validate_pipeline_element
from dimos.utils.logging_config import setup_logger

logger = setup_logger("dimos.stream.audio2.operators.gst_normalizer")


class GStreamerNormalizer(GStreamerPipelineBase):
    """GStreamer-based adaptive normalizer using level + audioamplify elements."""

    def __init__(
        self,
        target_level: float = 0.8,
        min_volume_threshold: float = 0.01,
        max_gain: float = 10.0,
        decay_factor: float = 0.999,
        adapt_speed: float = 0.05,
        level_interval_ms: int = 50,
    ):
        super().__init__()

        # Normalization parameters
        self.target_level = target_level
        self.min_volume_threshold = min_volume_threshold
        self.max_gain = max_gain
        self.decay_factor = decay_factor
        self.adapt_speed = adapt_speed
        self.level_interval_ms = level_interval_ms

        # Pipeline elements
        self._appsrc: Optional[GstApp.AppSrc] = None
        self._appsink: Optional[GstApp.AppSink] = None
        self._level: Optional[Gst.Element] = None
        self._amplify: Optional[Gst.Element] = None

        # State for adaptive normalization
        self._max_volume = 0.0
        self._current_gain = 1.0
        self._gain_lock = threading.Lock()

        # Input/output format tracking
        self._input_format: Optional[AudioSpec] = None
        self._output_rate: Optional[int] = None
        self._output_channels: Optional[int] = None

        # Threading
        self._pull_thread: Optional[threading.Thread] = None
        self._observer = None

    def _update_gain_from_level(self, structure):
        """Update gain based on level measurements."""
        try:
            # Get peak values (one per channel)
            peak_values = structure.get_value("peak")
            if not peak_values:
                return

            # Convert dB to linear scale and find max across channels
            max_peak_db = max(peak_values)

            # Convert dB to linear: linear = 10^(dB/20)
            if max_peak_db > -200:  # Ignore silence (-inf dB)
                current_volume = pow(10.0, max_peak_db / 20.0)
            else:
                current_volume = 0.0

            with self._gain_lock:
                # Update max volume with decay
                self._max_volume = max(current_volume, self._max_volume * self.decay_factor)

                # Calculate ideal gain
                if self._max_volume > self.min_volume_threshold:
                    ideal_gain = self.target_level / self._max_volume
                else:
                    ideal_gain = 1.0

                # Limit gain
                ideal_gain = min(ideal_gain, self.max_gain)

                # Smoothly adapt current gain
                self._current_gain = (
                    1 - self.adapt_speed
                ) * self._current_gain + self.adapt_speed * ideal_gain

                # Apply to audioamplify
                if self._amplify:
                    self._amplify.set_property("amplification", self._current_gain)

                # Debug logging for every level update
                logger.debug(
                    f"Normalizer: peak={max_peak_db:.1f}dB, "
                    f"vol={current_volume:.3f}, "
                    f"max={self._max_volume:.3f}, "
                    f"ideal={ideal_gain:.2f}x, "
                    f"gain={self._current_gain:.2f}x"
                )

        except Exception as e:
            logger.error(f"Error updating gain: {e}")

    def _on_bus_message(self, bus, message):
        """Handle bus messages, particularly from level element."""
        # Call parent handler first
        super()._on_bus_message(bus, message)

        # Handle level messages
        if message.type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure and structure.get_name() == "level":
                self._update_gain_from_level(structure)

    def _create_pipeline(self):
        """Create the GStreamer pipeline: appsrc ! decodebin ! audioconvert ! level ! audioamplify ! appsink."""

        # Build pipeline string
        # IMPORTANT: level must come BEFORE audioamplify to measure the original signal,
        # not the amplified signal (otherwise we get a feedback loop)
        pipeline_str = (
            "appsrc name=src ! "
            "decodebin ! "
            "audioconvert ! "
            "audio/x-raw,format=F32LE ! "  # Convert to float32 for processing
            "level name=level ! "
            "audioamplify name=amplify ! "
            "appsink name=sink"
        )

        logger.debug(f"Creating normalizer pipeline: {pipeline_str}")
        self._pipeline = Gst.parse_launch(pipeline_str)

        # Get elements
        self._appsrc = validate_pipeline_element(self._pipeline, "src")
        self._appsink = validate_pipeline_element(self._pipeline, "sink")
        self._level = validate_pipeline_element(self._pipeline, "level")
        self._amplify = validate_pipeline_element(self._pipeline, "amplify")

        # Configure appsrc
        self._appsrc.set_property("is-live", False)
        self._appsrc.set_property("format", Gst.Format.TIME)
        self._appsrc.set_property("block", False)

        # Configure appsink
        self._appsink.set_property("emit-signals", True)
        self._appsink.set_property("sync", False)

        # Configure level element
        interval_ns = self.level_interval_ms * 1000000  # Convert ms to ns
        self._level.set_property("interval", interval_ns)
        self._level.set_property("post-messages", True)

        # Configure audioamplify
        self._amplify.set_property("amplification", 1.0)
        self._amplify.set_property("clipping-method", 0)  # Normal clipping

        # Set up bus
        self._setup_bus(self._pipeline)

    def _push_event_to_pipeline(self, event: AudioEvent):
        """Push an AudioEvent to the pipeline via appsrc."""
        # Set caps on first event
        if self._input_format is None:
            self._input_format = AudioSpec(
                format=event.format, sample_rate=event.sample_rate, channels=event.channels
            )
            caps_str = self._input_format.to_gst_caps_string()
            if "layout=" not in caps_str:
                caps_str += ",layout=interleaved"
            caps = Gst.Caps.from_string(caps_str)
            self._appsrc.set_property("caps", caps)
            logger.info(f"Set input caps: {caps_str}")

        # Create buffer
        if isinstance(event, RawAudioEvent):
            if event.channels > 1:
                data = np.ascontiguousarray(event.data)
            else:
                data = event.data
            buffer = Gst.Buffer.new_wrapped(data.tobytes())
        elif isinstance(event, CompressedAudioEvent):
            buffer = Gst.Buffer.new_wrapped(event.data)
        else:
            logger.warning(f"Unknown event type: {type(event)}")
            return

        # Set timestamp
        buffer.pts = int(event.timestamp * Gst.SECOND)

        # Push to pipeline
        ret = self._appsrc.emit("push-buffer", buffer)
        if ret != Gst.FlowReturn.OK:
            logger.warning(f"Failed to push buffer: {ret}")

    def _pull_from_pipeline(self):
        """Pull processed samples from appsink and emit as AudioEvents."""
        logger.debug("Pull thread starting")

        buffer_count = 0
        try:
            while self._running:
                sample = self._appsink.emit("pull-sample")
                if sample is None:
                    if not self._running:
                        break
                    time.sleep(0.001)
                    continue

                buffer = sample.get_buffer()
                if buffer is None:
                    continue

                # Parse caps on first sample
                if self._output_rate is None:
                    caps = sample.get_caps()
                    if caps:
                        structure = caps.get_structure(0)
                        self._output_rate = structure.get_int("rate")[1]
                        self._output_channels = structure.get_int("channels")[1]
                        logger.info(
                            f"Output format: {self._output_rate}Hz, {self._output_channels}ch"
                        )

                # Convert to AudioEvent
                # audioamplify outputs F32LE format
                from dimos.stream.audio2.types import AudioFormat

                output_spec = AudioSpec(
                    format=AudioFormat.PCM_F32LE,
                    sample_rate=self._output_rate,
                    channels=self._output_channels,
                )

                # Extract timestamp from buffer PTS (preserve timing from input)
                buffer_timestamp = None
                if buffer.pts != Gst.CLOCK_TIME_NONE:
                    buffer_timestamp = buffer.pts / Gst.SECOND

                event = buffer_to_audio_event(
                    buffer=buffer,
                    spec=output_spec,
                    detected_rate=self._output_rate,
                    detected_channels=self._output_channels,
                    timestamp=buffer_timestamp,
                )

                buffer_count += 1
                if buffer_count == 1 or buffer_count % 20 == 0:
                    logger.debug(
                        f"Processed {buffer_count} buffers (gain={self._current_gain:.2f}x)"
                    )

                if self._observer:
                    self._observer.on_next(event)

        except Exception as e:
            logger.error(f"Pull thread error: {e}")
            if self._observer:
                self._observer.on_error(e)

        finally:
            logger.info(f"Pull thread exiting (processed {buffer_count} buffers)")

    def process_observable(self, source: ObservableBase) -> ObservableBase:
        """Process an observable of AudioEvents through the normalizer."""

        def subscribe(observer, scheduler=None):
            self._observer = observer

            try:
                # Ensure mainloop is running
                self._ensure_pipeline_ready()
                self._create_pipeline()

                # Start pipeline
                ret = self._pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("Failed to start pipeline")

                # Start pull thread
                self._pull_thread = threading.Thread(
                    target=self._pull_from_pipeline, daemon=True, name="GstNormalizer-pull"
                )
                self._pull_thread.start()

                logger.info(
                    f"GStreamer normalizer started "
                    f"(target={self.target_level}, max_gain={self.max_gain})"
                )

                # Subscribe to source and push events to pipeline
                def on_next(event):
                    self._push_event_to_pipeline(event)

                def on_error(error):
                    logger.error(f"Source error: {error}")
                    self._cleanup_pipeline()
                    observer.on_error(error)

                def on_completed():
                    logger.info("Source completed, sending EOS")
                    if self._appsrc:
                        self._appsrc.emit("end-of-stream")
                    # Wait a bit for pipeline to finish processing
                    time.sleep(0.1)
                    self._running = False
                    if self._pull_thread:
                        self._pull_thread.join(timeout=2.0)

                    # Notify observer FIRST - downstream operators need the mainloop
                    # to still be running so they can process their EOS messages
                    observer.on_completed()

                    # Now cleanup and release mainloop reference
                    self._cleanup_pipeline()

                source.subscribe(on_next, on_error, on_completed, scheduler=scheduler)

            except Exception as e:
                observer.on_error(e)
                self._cleanup_pipeline()

            def dispose():
                logger.info("Disposing GStreamer normalizer")
                self._running = False
                if self._pipeline:
                    self._pipeline.set_state(Gst.State.NULL)
                if self._pull_thread:
                    self._pull_thread.join(timeout=2.0)
                self._cleanup_pipeline()

            from reactivex import disposable

            return disposable.Disposable(dispose)

        return create(subscribe)


def normalizer(
    target_level: float = 0.8,
    min_volume_threshold: float = 0.01,
    max_gain: float = 10.0,
    decay_factor: float = 0.999,
    adapt_speed: float = 0.05,
    level_interval_ms: int = 50,
):
    """Create an adaptive GStreamer-based audio normalizer operator.

    This operator applies dynamic normalization to audio events using GStreamer's
    level and audioamplify elements. It works with both raw and compressed audio
    formats by automatically decoding/encoding as needed.

    The normalizer tracks maximum volume with decay and smoothly adapts gain to
    reach the target level, similar to raw_normalizer but using native GStreamer
    processing.

    Args:
        target_level: Target normalization level (0.0 to 1.0, default: 0.8)
        min_volume_threshold: Minimum volume to apply normalization (default: 0.01)
        max_gain: Maximum allowed gain to prevent excessive amplification (default: 10.0)
        decay_factor: Decay factor for max volume (0.0-1.0, higher = slower decay, default: 0.999)
        adapt_speed: How quickly to adapt to new volume levels (0.0-1.0, default: 0.05)
        level_interval_ms: Interval for level measurements in milliseconds (default: 50)

    Returns:
        An operator function that can be used with pipe()

    Examples:
        # Normalize compressed audio
        file_input("audio.mp3").pipe(
            normalizer(target_level=0.8),
            speaker()
        ).run()

        # Normalize raw audio with custom parameters
        microphone().pipe(
            normalizer(target_level=0.7, adapt_speed=0.1),
            speaker()
        ).run()

        # Quick adaptation for varying content
        file_input("podcast.opus").pipe(
            normalizer(adapt_speed=0.2, decay_factor=0.95),
            file_output("normalized.opus")
        ).run()
    """

    norm = GStreamerNormalizer(
        target_level=target_level,
        min_volume_threshold=min_volume_threshold,
        max_gain=max_gain,
        decay_factor=decay_factor,
        adapt_speed=adapt_speed,
        level_interval_ms=level_interval_ms,
    )

    def _normalizer(source: ObservableBase) -> ObservableBase:
        return norm.process_observable(source)

    return _normalizer
