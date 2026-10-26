"""
Audio file I/O, recording and playback.

Everything that touches hardware or the filesystem lives here, so the DSP modules stay pure
and testable.  Playback and recording are deliberately non-blocking: the GUI thread must
never wait on an audio device, or the whole window freezes for the length of the clip.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("spectra.audio")

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a")
FILE_TYPES = [("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3"), ("All files", "*.*")]


class AudioError(RuntimeError):
    """Raised when a file cannot be read/written or no audio device is available."""


def _soundfile():
    try:
        import soundfile as sf
        return sf
    except Exception as exc:                                   # pragma: no cover
        raise AudioError("soundfile is not installed - run: pip install soundfile") from exc


def _sounddevice():
    try:
        import sounddevice as sd
        return sd
    except Exception as exc:                                   # pragma: no cover
        raise AudioError("sounddevice is not installed - run: pip install sounddevice") from exc


# ----------------------------------------------------------------- files
def load(path: str, target_fs: float | None = None, mono: bool = True):
    """Read an audio file as float64.  Returns (samples, sample rate).

    Multi-channel input is averaged to mono: every analysis in this project is single-channel,
    and averaging is both the honest default and cheaper than asking at load time.
    """
    sf = _soundfile()
    try:
        data, fs = sf.read(path, dtype="float64", always_2d=True)
    except Exception as exc:
        raise AudioError(f"Could not read {path}: {exc}") from exc
    x = data.mean(axis=1) if mono else data
    x = np.asarray(x, dtype=np.float64)
    if target_fs and abs(target_fs - fs) > 1e-6:
        from .signals import resample_to
        x = resample_to(x, fs, target_fs)
        fs = target_fs
    return x, float(fs)


def save(path: str, x: np.ndarray, fs: float, subtype: str = "PCM_16") -> None:
    """Write a WAV/FLAC/OGG file, peak-limiting first so nothing clips on write."""
    sf = _soundfile()
    y = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.99
    try:
        sf.write(path, y, int(round(fs)), subtype=subtype)
    except Exception as exc:
        raise AudioError(f"Could not write {path}: {exc}") from exc


def duration_of(x: np.ndarray, fs: float) -> float:
    return float(x.size) / float(fs) if fs else 0.0


# ----------------------------------------------------------------- devices
def list_devices() -> list:
    """Input/output devices as display strings, for the settings page."""
    try:
        sd = _sounddevice()
        out = []
        for i, d in enumerate(sd.query_devices()):
            kind = []
            if d.get("max_input_channels", 0) > 0:
                kind.append("in")
            if d.get("max_output_channels", 0) > 0:
                kind.append("out")
            out.append(f"[{i}] {d.get('name', '?')} ({'/'.join(kind) or 'none'})")
        return out
    except Exception as exc:
        log.info("Could not query audio devices: %s", exc)
        return []


def has_output() -> bool:
    try:
        sd = _sounddevice()
        return sd.query_devices(kind="output") is not None
    except Exception:
        return False


def has_input() -> bool:
    try:
        sd = _sounddevice()
        return sd.query_devices(kind="input") is not None
    except Exception:
        return False


# ----------------------------------------------------------------- playback
class Player:
    """Non-blocking playback of a numpy array.

    sounddevice's own `play` is already asynchronous, but it keeps one global stream; wrapping
    it lets several pages share a single player and makes "stop whatever is playing" a single
    call that cannot leave a stream running after the window closes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._playing = False

    def play(self, x: np.ndarray, fs: float, on_done=None) -> None:
        sd = _sounddevice()
        y = np.asarray(x, dtype=np.float32)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:
            y = y / peak * 0.99
        with self._lock:
            try:
                sd.stop()
                sd.play(y, int(round(fs)))
                self._playing = True
            except Exception as exc:
                raise AudioError(f"Playback failed: {exc}") from exc

        if on_done is not None:
            def waiter():
                try:
                    sd.wait()
                except Exception:
                    pass
                self._playing = False
                on_done()
            threading.Thread(target=waiter, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            try:
                _sounddevice().stop()
            except Exception:
                pass

    @property
    def playing(self) -> bool:
        return self._playing


class Recorder:
    """Microphone capture on a worker thread.

    The buffer is collected in the audio callback and only assembled when `stop` is called, so
    the GUI can poll `seconds` for a live duration readout without touching the stream.
    """

    def __init__(self, fs: float = 44100.0, channels: int = 1):
        self.fs = float(fs)
        self.channels = int(channels)
        self._frames: list = []
        self._stream = None
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def seconds(self) -> float:
        with self._lock:
            n = sum(f.shape[0] for f in self._frames)
        return n / self.fs if self.fs else 0.0

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _sounddevice()
        with self._lock:
            self._frames = []

        def callback(indata, _frames, _time, status):
            if status:
                log.debug("Recording status: %s", status)
            with self._lock:
                self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(samplerate=self.fs, channels=self.channels,
                                          dtype="float32", callback=callback)
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioError(f"Could not open the microphone: {exc}") from exc

    def stop(self):
        """Stop capture and return (samples, sample rate).  Empty array if nothing arrived."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.info("Error closing the input stream: %s", exc)
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return np.zeros(0, dtype=np.float64), self.fs
        data = np.concatenate(frames, axis=0).astype(np.float64)
        return (data.mean(axis=1) if data.ndim > 1 else data), self.fs
