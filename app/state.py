"""
Shared session state.

One `Session` object is created by the main window and handed to every page.  It owns the
signals, the images, the processing history and the event log, and it notifies pages when
any of those change.  Pages therefore never reach into each other: a filter applied on the
Filter page shows up on the Spectral page because both are looking at the same session.

The three-signal model is the important part:

    original  - the untouched source (loaded, generated or recorded)
    reference - the clean ground truth, when one exists
    current   - the working signal, i.e. the result of the pipeline so far

`reference` is what makes honest metrics possible.  When noise is added inside the
application the clean signal is kept, so SNR improvement is a real measurement.  When a
noisy file is loaded from disk there is no ground truth, `reference` stays None, and the UI
reports only the metrics that do not need one rather than inventing a baseline.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("spectra.state")

MAX_EVENTS = 400


@dataclass
class Signal:
    """A 1-D signal with its sample rate and provenance."""
    data: np.ndarray
    fs: float
    name: str = "signal"
    source: str = "generated"          # generated | file | recorded | derived

    @property
    def duration(self) -> float:
        return float(self.data.size) / self.fs if self.fs else 0.0

    @property
    def samples(self) -> int:
        return int(self.data.size)

    def copy(self, data=None, name=None, source=None) -> "Signal":
        return Signal(data=np.array(self.data if data is None else data, dtype=np.float64),
                      fs=self.fs, name=name or self.name, source=source or self.source)

    def describe(self) -> str:
        return (f"{self.name} · {self.samples:,} samples · {self.fs:,.0f} Hz · "
                f"{self.duration:.2f} s")


@dataclass
class Step:
    """One applied processing stage, for the pipeline view and the exported report."""
    stage: str
    detail: str
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    at: dt.datetime = field(default_factory=dt.datetime.now)

    def line(self) -> str:
        return f"{self.at:%H:%M:%S}  {self.stage:<22} {self.detail}"


@dataclass
class Event:
    level: str                          # INFO | OK | WARN | ERROR
    message: str
    at: dt.datetime = field(default_factory=dt.datetime.now)


class Session:
    """Everything the application knows about the work in progress."""

    def __init__(self, fs_default: float = 8000.0):
        self.fs_default = fs_default
        self.original: Signal | None = None
        self.reference: Signal | None = None      # clean ground truth when known
        self.current: Signal | None = None
        self.noise_clip: np.ndarray | None = None  # noise-only excerpt, when available

        self.image_original = None
        self.image_current = None
        self.image_reference = None
        self.image_psf = None          # blur kernel applied in-app, for a true deconvolution
        self.image_name = ""
        self.image_edges = None        # EdgeSettings last used on the Image page

        self.video_path: str | None = None
        self.video_info: dict = {}
        self.video_preview: dict | None = None   # last previewed frame pair, for the report

        self.steps: list[Step] = []
        self.events: list[Event] = []
        self._listeners: list = []
        self.started = dt.datetime.now()

    # ------------------------------------------------------------- notification
    def subscribe(self, callback) -> None:
        """Register `callback(topic)`; topics are 'signal', 'image', 'video' or 'log'."""
        self._listeners.append(callback)

    def notify(self, topic: str = "signal") -> None:
        for cb in list(self._listeners):
            try:
                cb(topic)
            except Exception:
                log.exception("A session listener failed for topic %s", topic)

    # ------------------------------------------------------------- logging
    def log_event(self, message: str, level: str = "INFO") -> None:
        self.events.append(Event(level=level, message=message))
        if len(self.events) > MAX_EVENTS:
            del self.events[:-MAX_EVENTS]
        log.info("[%s] %s", level, message)
        self.notify("log")

    def add_step(self, stage: str, detail: str, params: dict | None = None,
                 metrics: dict | None = None) -> None:
        self.steps.append(Step(stage=stage, detail=detail, params=dict(params or {}),
                               metrics=dict(metrics or {})))
        self.log_event(f"{stage}: {detail}", "OK")

    # ------------------------------------------------------------- signals
    def set_signal(self, data, fs: float, name: str, source: str = "generated",
                   reference=None, reset: bool = True) -> Signal:
        """Install a new source signal.

        `reset` clears the pipeline history, which is what should happen whenever the source
        changes - carrying steps over from a previous signal would make the report lie.
        """
        sig = Signal(data=np.asarray(data, dtype=np.float64), fs=float(fs), name=name,
                     source=source)
        self.original = sig
        self.current = sig.copy(source="derived")
        self.reference = (Signal(np.asarray(reference, dtype=np.float64), float(fs),
                                 name + " (clean)", "generated")
                          if reference is not None else None)
        self.noise_clip = None
        if reset:
            self.steps = []
        self.log_event(f"Loaded {sig.describe()}", "OK")
        self.notify("signal")
        return sig

    def set_current(self, data, name: str | None = None, fs: float | None = None) -> None:
        """Replace the working signal, leaving the original and reference untouched."""
        if self.current is None:
            return
        self.current = Signal(data=np.asarray(data, dtype=np.float64),
                              fs=float(fs or self.current.fs),
                              name=name or self.current.name, source="derived")
        self.notify("signal")

    def set_reference(self, data, fs: float | None = None, name: str = "clean reference") -> None:
        self.reference = Signal(np.asarray(data, dtype=np.float64),
                                float(fs or (self.current.fs if self.current else self.fs_default)),
                                name, "generated")
        self.notify("signal")

    def revert(self) -> None:
        """Discard the pipeline and go back to the source signal."""
        if self.original is None:
            return
        self.current = self.original.copy(source="derived")
        self.steps = []
        self.log_event("Reverted to the original signal", "INFO")
        self.notify("signal")

    def commit_current_as_original(self) -> None:
        """Promote the processed signal to be the new source - useful when chaining
        experiments, and it keeps the history honest by starting a fresh pipeline."""
        if self.current is None:
            return
        self.original = self.current.copy(source="derived")
        self.steps = []
        self.log_event("Promoted the processed signal to the new source", "INFO")
        self.notify("signal")

    # ------------------------------------------------------------- images
    def set_image(self, array, name: str, reference=None, reset: bool = True) -> None:
        self.image_original = np.asarray(array, dtype=np.float64)
        self.image_current = self.image_original.copy()
        self.image_reference = (np.asarray(reference, dtype=np.float64)
                                if reference is not None else None)
        self.image_psf = None
        self.image_name = name
        if reset:
            self.steps = [s for s in self.steps if not s.stage.startswith("Image")]
        h, w = self.image_original.shape[:2]
        self.log_event(f"Loaded image {name} ({w} x {h})", "OK")
        self.notify("image")

    def set_image_current(self, array) -> None:
        self.image_current = np.asarray(array, dtype=np.float64)
        self.notify("image")

    def set_image_input(self, array, psf=None) -> None:
        """Replace the working input while keeping the pristine reference.

        Used when an image is deliberately degraded: `image_original` becomes the damaged
        version that restoration starts from, but `image_reference` still holds the original,
        which is what makes the PSNR and SSIM afterwards a real measurement rather than a
        comparison against the damage.  `psf` records the blur kernel that was applied, so a
        later deblur can be a true inverse instead of a guess.
        """
        self.image_original = np.asarray(array, dtype=np.float64)
        self.image_current = self.image_original.copy()
        self.image_psf = psf
        self.notify("image")

    def revert_image(self) -> None:
        if self.image_original is None:
            return
        self.image_current = self.image_original.copy()
        self.log_event("Reverted to the original image", "INFO")
        self.notify("image")

    # ------------------------------------------------------------- video
    def set_video(self, path: str, info: dict) -> None:
        self.video_path = path
        self.video_info = dict(info)
        self.video_preview = None
        self.log_event(f"Loaded video {path}", "OK")
        self.notify("video")

    # ------------------------------------------------------------- queries
    @property
    def has_signal(self) -> bool:
        return self.current is not None and self.current.data.size > 0

    @property
    def has_image(self) -> bool:
        return self.image_current is not None

    @property
    def has_video(self) -> bool:
        return bool(self.video_path)

    @property
    def fs(self) -> float:
        return self.current.fs if self.current else self.fs_default

    def reference_array(self):
        """The clean array to measure against, trimmed to the current length, or None.

        Trimming matters because several enhancement methods return a signal a few samples
        shorter or longer than their input, and a length mismatch would otherwise silently
        skew every reference-based metric.
        """
        if self.reference is None or self.current is None:
            return None
        n = min(self.reference.data.size, self.current.data.size)
        return self.reference.data[:n] if n else None

    def signal_metrics(self) -> dict:
        """The metric set for the current signal, including reference-based rows when a
        ground truth exists."""
        from dsp import metrics as M
        if not self.has_signal:
            return {}
        return M.signal_report(self.reference_array(), self.current.data, self.current.fs)

    def improvement(self) -> dict:
        """Before/after comparison against the original - the headline result of the session.

        Returns an empty dict when there is no ground truth, because without a clean
        reference an SNR 'improvement' cannot be measured, only guessed at.
        """
        from dsp import metrics as M
        ref = self.reference_array()
        if ref is None or self.original is None or self.current is None:
            return {}
        n = min(ref.size, self.original.data.size, self.current.data.size)
        if n == 0:
            return {}
        ref, before, after = ref[:n], self.original.data[:n], self.current.data[:n]
        fs = self.current.fs
        return {"SNR before (dB)": M.snr_db(ref, before),
                "SNR after (dB)": M.snr_db(ref, after),
                "SNR improvement (dB)": M.snr_db(ref, after) - M.snr_db(ref, before),
                "Segmental SNR before (dB)": M.segmental_snr_db(ref, before, fs),
                "Segmental SNR after (dB)": M.segmental_snr_db(ref, after, fs),
                "MSE before": M.mse(ref, before),
                "MSE after": M.mse(ref, after),
                "Correlation before": M.correlation(ref, before),
                "Correlation after": M.correlation(ref, after)}

    def estimated_improvement(self) -> dict:
        """Before/after for a recording with no clean copy, measured from the signal alone.

        Background noise is read from the pauses and speech level from the loud frames (see
        `metrics.activity_levels`), so a recording can still show what a clean-up achieved.
        Empty when nothing has been processed yet.
        """
        from dsp import metrics as M
        if self.original is None or self.current is None:
            return {}
        before, after = self.original.data, self.current.data
        if before.size == after.size and np.array_equal(before, after):
            return {}
        n0, s0 = M.activity_levels(before, self.original.fs)
        n1, s1 = M.activity_levels(after, self.current.fs)
        if not all(np.isfinite(v) for v in (n0, s0, n1, s1)):
            return {}
        return {"Background before (dB)": n0, "Background after (dB)": n1,
                "Background change (dB)": n1 - n0,
                "Speech level change (dB)": s1 - s0}

    def image_improvement(self) -> dict:
        from dsp import imaging as I
        if self.image_reference is None or self.image_current is None:
            return {}
        before = I.compare(self.image_reference, self.image_original)
        after = I.compare(self.image_reference, self.image_current)
        out = {}
        for key in after:
            out[f"{key} before"] = before.get(key, float("nan"))
            out[f"{key} after"] = after[key]
        return out

    def pipeline_lines(self) -> list:
        if not self.steps:
            return ["No processing applied yet."]
        return [s.line() for s in self.steps]
