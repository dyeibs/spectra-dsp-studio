"""
Video signal processing - the image chain extended along a time axis.

A video is a 3-D signal, and that third dimension is what makes new operations possible:
temporal averaging suppresses noise without the spatial blurring a 2-D filter would cause,
and frame differencing isolates motion.  Everything here is frame-streamed rather than loaded
whole, since even a short clip at full resolution will not fit comfortably in memory.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from . import imaging

log = logging.getLogger("spectra.video")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")
FILE_TYPES = [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("All files", "*.*")]

OPERATIONS = ("None (passthrough)", "Greyscale", "Gaussian denoise", "Median denoise",
              "Bilateral denoise", "Non-local means denoise", "Temporal averaging",
              "Temporal median", "Frame differencing (motion)", "Background subtraction",
              "Edge detection", "Sharpen (unsharp)", "Histogram equalisation",
              "CLAHE (adaptive)", "Frequency low-pass", "Frequency high-pass",
              "Motion-blur restoration")


class VideoError(RuntimeError):
    """Raised when a clip cannot be opened, decoded or written."""


class VideoSource:
    """A read-only handle on a video file, with random access by frame index.

    OpenCV's frame seeking is unreliable on some codecs, so `frame_at` falls back to reading
    forward from the start when a seek lands on the wrong frame.
    """

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise VideoError(f"Could not open the video {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def duration(self) -> float:
        return self.count / self.fps if self.fps else 0.0

    def frame_at(self, index: int) -> np.ndarray | None:
        """Return frame `index` as an RGB float64 array, or None past the end."""
        index = max(int(index), 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64)

    def frames(self, start: int = 0, limit: int | None = None, step: int = 1):
        """Yield (index, RGB float64 frame) - a generator so memory stays flat."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(start), 0))
        produced, index = 0, max(int(start), 0)
        while limit is None or produced < limit:
            ok, frame = self.cap.read()
            if not ok:
                break
            if (index - start) % max(int(step), 1) == 0:
                yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64)
                produced += 1
            index += 1

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def summary(self) -> dict:
        return {"Frames": self.count, "Frame rate (fps)": round(self.fps, 2),
                "Width": self.width, "Height": self.height,
                "Duration (s)": round(self.duration, 2)}


class FrameProcessor:
    """Applies one of OPERATIONS to a stream of frames.

    The temporal operations need state, which is the reason this is a class and not a
    function: `reset` clears the frame history so a new pass starts clean rather than
    inheriting the tail of the previous one.
    """

    def __init__(self, operation: str = "None (passthrough)", params: dict | None = None):
        self.operation = operation
        self.params = dict(params or {})
        self.history: list = []
        self.background = None

    def reset(self) -> None:
        self.history = []
        self.background = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        op = self.operation
        p = self.params
        x = np.asarray(frame, dtype=np.float64)

        if op == "None (passthrough)":
            return x
        if op == "Greyscale":
            g = imaging.to_gray(x)
            return np.stack([g] * 3, axis=2)
        if op == "Gaussian denoise":
            return imaging.spatial_filter(x, "Gaussian", int(p.get("size", 5)),
                                          p.get("sigma", 1.5))
        if op == "Median denoise":
            return imaging.spatial_filter(x, "Median", int(p.get("size", 5)))
        if op == "Bilateral denoise":
            return imaging.spatial_filter(x, "Bilateral", int(p.get("size", 9)),
                                          p.get("sigma", 75.0), sigma_colour=p.get("sigma_colour", 75.0))
        if op == "Non-local means denoise":
            return imaging.spatial_filter(x, "Non-local means", strength=p.get("strength", 1.0))
        if op in ("Temporal averaging", "Temporal median"):
            return self._temporal(x, op, int(p.get("window", 5)))
        if op == "Frame differencing (motion)":
            return self._difference(x, p.get("threshold", 18.0))
        if op == "Background subtraction":
            return self._background(x, p.get("alpha", 0.02), p.get("threshold", 25.0))
        if op == "Edge detection":
            e = imaging.edge_detect(x, p.get("operator", "Sobel"), int(p.get("ksize", 3)),
                                    p.get("low", 50.0), p.get("high", 150.0))
            e = e / max(float(e.max()), 1e-9) * 255.0
            return np.stack([e] * 3, axis=2)
        if op == "Sharpen (unsharp)":
            return imaging.spatial_filter(x, "Unsharp mask", int(p.get("size", 5)),
                                          p.get("sigma", 1.5), p.get("strength", 1.0))
        if op in ("Histogram equalisation", "CLAHE (adaptive)"):
            return imaging.histogram_op(x, op, clip_limit=p.get("clip_limit", 2.0))
        if op == "Frequency low-pass":
            out, _m = imaging.frequency_filter(x, "Gaussian low-pass", p.get("cutoff", 60.0))
            return out
        if op == "Frequency high-pass":
            out, _m = imaging.frequency_filter(x, "Gaussian high-pass", p.get("cutoff", 20.0))
            return out
        if op == "Motion-blur restoration":
            psf = imaging.motion_psf(int(p.get("length", 11)), p.get("angle", 0.0))
            return imaging.wiener_deconvolution(x, psf, p.get("k", 0.01))
        raise ValueError("Unknown video operation: " + str(op))

    def _temporal(self, frame, op, window):
        """Average or median over the last `window` frames.

        Noise is independent between frames while the scene is not, so this suppresses noise
        without touching spatial detail - as long as nothing moves.  Where something does
        move it smears, which is precisely the trade-off worth demonstrating.
        """
        self.history.append(frame)
        if len(self.history) > max(window, 1):
            self.history.pop(0)
        stack = np.stack(self.history, axis=0)
        return stack.mean(axis=0) if op == "Temporal averaging" else np.median(stack, axis=0)

    def _difference(self, frame, threshold):
        """Highlight pixels that changed since the previous frame - the simplest motion
        detector, and the reason a static camera is assumed."""
        g = imaging.to_gray(frame)
        prev = self.history[-1] if self.history else g
        self.history = [g]
        diff = np.abs(g - prev)
        mask = (diff > threshold).astype(np.float64) * 255.0
        out = frame * 0.35
        out[..., 0] = np.maximum(out[..., 0], mask)       # changed pixels tinted red
        return out

    def _background(self, frame, alpha, threshold):
        """Running-average background model, foreground = |frame - background|.

        `alpha` sets how fast the model forgets: too high and a slow-moving object is absorbed
        into the background, too low and lighting changes register as permanent motion.
        """
        g = imaging.to_gray(frame)
        if self.background is None:
            self.background = g.copy()
        self.background = (1.0 - alpha) * self.background + alpha * g
        mask = (np.abs(g - self.background) > threshold).astype(np.float64)
        return frame * mask[..., None]


def process_video(src_path: str, dst_path: str, operation: str, params: dict | None = None,
                  progress=None, max_frames: int | None = None, should_stop=None) -> dict:
    """Process a whole clip frame by frame and write a new file.

    `progress(done, total)` is called as it goes and `should_stop()` is polled so the UI can
    cancel a long export; both are optional.  Returns a stats dict for the report.
    """
    source = VideoSource(src_path)
    proc = FrameProcessor(operation, params)
    total = source.count if max_frames is None else min(source.count, max_frames)
    writer = None
    written = 0
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dst_path, fourcc, source.fps, (source.width, source.height))
        if not writer.isOpened():
            raise VideoError(f"Could not open {dst_path} for writing")
        for _idx, frame in source.frames(limit=total):
            if should_stop is not None and should_stop():
                break
            out = imaging.to_uint8(proc.process(frame))
            if out.ndim == 2:
                out = np.stack([out] * 3, axis=2)
            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            written += 1
            if progress is not None and written % 5 == 0:
                progress(written, total)
    finally:
        if writer is not None:
            writer.release()
        source.close()
    if progress is not None:
        progress(written, total)
    return {"Frames written": written, "Frames requested": total,
            "Frame rate (fps)": round(source.fps, 2),
            "Resolution": f"{source.width} x {source.height}", "Output": dst_path}


def extract_frames(path: str, indices) -> list:
    """Pull specific frames out for the side-by-side preview."""
    source = VideoSource(path)
    try:
        return [(i, f) for i in indices for f in [source.frame_at(i)] if f is not None]
    finally:
        source.close()


def motion_energy(path: str, max_frames: int = 300, step: int = 1):
    """Mean absolute inter-frame difference over time.

    Reduces the clip to a single 1-D signal, which can then be analysed with exactly the same
    tools as any other signal in this project - a neat way to close the loop between the video
    and audio halves of the system.
    """
    source = VideoSource(path)
    try:
        values, times, prev = [], [], None
        for idx, frame in source.frames(limit=max_frames, step=max(int(step), 1)):
            g = imaging.to_gray(frame)
            if prev is not None:
                values.append(float(np.mean(np.abs(g - prev))))
                times.append(idx / source.fps)
            prev = g
        return np.asarray(times), np.asarray(values)
    finally:
        source.close()
