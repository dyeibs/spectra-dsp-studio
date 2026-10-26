"""
Spectral and time-frequency analysis.

The transform layer: DFT magnitude/phase, power spectral density, STFT spectrogram, cepstrum,
mel-frequency cepstral coefficients, autocorrelation pitch tracking and short-time features.
MFCC and the mel filterbank are implemented here rather than pulled from librosa, so every
step of the speech front end is visible and explainable during the defence.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as spfft, signal as sig

WINDOWS = ("hann", "hamming", "blackman", "blackmanharris", "bartlett", "flattop",
           "boxcar (none)", "kaiser")
EPS = 1e-12


def _window(name: str, n: int) -> np.ndarray:
    """Build an analysis window.  The choice is a resolution/leakage trade: boxcar gives the
    narrowest main lobe but -13 dB sidelobes, flattop the best amplitude accuracy."""
    if name.startswith("boxcar"):
        return np.ones(n)
    if name == "kaiser":
        return np.kaiser(n, 8.6)
    try:
        return sig.get_window(name, n, fftbins=True)
    except Exception:
        return np.hanning(n)


# ----------------------------------------------------------------- DFT
def spectrum(x: np.ndarray, fs: float, window: str = "hann", db: bool = True,
             n_fft: int | None = None, normalise: bool = True):
    """One-sided DFT magnitude.

    Returns (frequencies, magnitude).  The window's coherent gain is divided out so that a
    full-scale sine reads its true amplitude instead of being attenuated by the window.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return np.zeros(1), np.zeros(1)
    n = int(n_fft or x.size)
    w = _window(window, x.size)
    xw = x * w
    if n > x.size:
        xw = np.pad(xw, (0, n - x.size))
    spec = np.abs(spfft.rfft(xw[:n], n=n))
    if normalise:
        spec = spec * 2.0 / max(np.sum(w), EPS)
    freqs = spfft.rfftfreq(n, 1.0 / fs)
    return freqs, (20.0 * np.log10(np.maximum(spec, EPS)) if db else spec)


def phase_spectrum(x: np.ndarray, fs: float, window: str = "hann", unwrap: bool = True):
    w = _window(window, x.size)
    spec = spfft.rfft(x * w)
    ph = np.angle(spec)
    return spfft.rfftfreq(x.size, 1.0 / fs), np.degrees(np.unwrap(ph) if unwrap else ph)


def psd_welch(x: np.ndarray, fs: float, nperseg: int = 1024, window: str = "hann",
              db: bool = True):
    """Welch's averaged periodogram - splits the record into overlapping segments and averages
    their periodograms, trading frequency resolution for a far lower-variance estimate."""
    nper = int(min(max(nperseg, 32), max(x.size, 32)))
    f, pxx = sig.welch(x, fs=fs, window=_window(window, nper), nperseg=nper,
                       noverlap=nper // 2, detrend="constant")
    return f, (10.0 * np.log10(np.maximum(pxx, EPS)) if db else pxx)


def spectrogram(x: np.ndarray, fs: float, frame_ms: float = 25.0, overlap: float = 0.5,
                window: str = "hann", db: bool = True, n_fft: int | None = None):
    """STFT magnitude as (times, frequencies, matrix[freq, time]).

    Frame length sets the resolution trade directly: short frames localise transients in time
    but smear them in frequency, long frames do the reverse.  This is the uncertainty
    principle in practice, and the Spectral page exposes the slider so it can be shown live.
    """
    nper = int(2 ** round(np.log2(max(frame_ms * fs / 1000.0, 32))))
    nper = min(nper, max(x.size, 32))
    nov = int(np.clip(nper * overlap, 0, nper - 1))
    f, t, Z = sig.stft(x, fs=fs, window=_window(window, nper), nperseg=nper, noverlap=nov,
                       nfft=n_fft, boundary=None, padded=True)
    mag = np.abs(Z)
    return t, f, (20.0 * np.log10(np.maximum(mag, EPS)) if db else mag)


def cepstrum(x: np.ndarray, fs: float, real: bool = True):
    """Real cepstrum: IFFT of the log magnitude spectrum, plotted against quefrency (seconds).

    It turns convolution into addition, so the slowly varying vocal-tract envelope separates
    from the periodic excitation; the peak in the 2-20 ms range is the pitch period.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return np.zeros(1), np.zeros(1)
    spec = spfft.rfft(x * np.hanning(x.size))
    log_mag = np.log(np.maximum(np.abs(spec), EPS))
    c = spfft.irfft(log_mag if real else log_mag + 1j * np.unwrap(np.angle(spec)), n=x.size)
    quefrency = np.arange(c.size) / fs
    half = c.size // 2
    return quefrency[:half], np.asarray(c[:half], dtype=np.float64)


# ----------------------------------------------------------------- mel / MFCC
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, n_fft: int, fs: float, fmin: float = 0.0,
                   fmax: float | None = None) -> np.ndarray:
    """Triangular mel filters as a (n_filters, n_fft//2+1) matrix.

    Centres are spaced evenly on the mel scale, which approximates how the ear's frequency
    resolution coarsens with frequency - the reason MFCCs compress speech so efficiently.
    """
    fmax = float(fmax or fs / 2.0)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2))
    bins = np.floor((n_fft + 1) * pts / fs).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        lo, mid, hi = bins[i], bins[i + 1], bins[i + 2]
        if mid > lo:
            fb[i, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[i, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return fb


def mfcc(x: np.ndarray, fs: float, n_mfcc: int = 13, n_filters: int = 26,
         frame_ms: float = 25.0, hop_ms: float = 10.0, pre_emph: float = 0.97,
         lifter: int = 22):
    """Mel-frequency cepstral coefficients, computed end to end.

    Pre-emphasis -> frame + Hamming window -> power spectrum -> mel filterbank -> log ->
    DCT-II.  The DCT decorrelates the log-mel energies so the first dozen coefficients carry
    nearly all of the spectral-envelope information.  Returns (times, coefficients[n_mfcc, T]).
    """
    x = np.asarray(x, dtype=np.float64)
    if pre_emph:
        x = np.append(x[:1], x[1:] - pre_emph * x[:-1])
    frame = max(int(frame_ms * fs / 1000.0), 16)
    hop = max(int(hop_ms * fs / 1000.0), 1)
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    n_fft = int(2 ** np.ceil(np.log2(frame)))

    idx = np.arange(0, x.size - frame + 1, hop)
    frames = np.stack([x[i:i + frame] for i in idx]) * np.hamming(frame)
    power = (np.abs(spfft.rfft(frames, n=n_fft, axis=1)) ** 2) / n_fft

    fb = mel_filterbank(n_filters, n_fft, fs)
    energies = np.log(np.maximum(power @ fb.T, EPS))
    coeffs = spfft.dct(energies, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    if lifter > 0:                                     # sinusoidal liftering, as in HTK
        n = np.arange(n_mfcc)
        coeffs *= 1.0 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
    return idx / fs, coeffs.T


def delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression-based first difference of a feature matrix - the standard delta features."""
    f = np.asarray(features, dtype=np.float64)
    pad = np.pad(f, ((0, 0), (width, width)), mode="edge")
    denom = 2.0 * sum(i * i for i in range(1, width + 1))
    out = np.zeros_like(f)
    for i in range(1, width + 1):
        out += i * (pad[:, width + i: width + i + f.shape[1]] -
                    pad[:, width - i: width - i + f.shape[1]])
    return out / denom


# ----------------------------------------------------------------- short-time features
def frame_signal(x: np.ndarray, fs: float, frame_ms: float = 25.0, hop_ms: float = 10.0):
    frame = max(int(frame_ms * fs / 1000.0), 8)
    hop = max(int(hop_ms * fs / 1000.0), 1)
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    idx = np.arange(0, x.size - frame + 1, hop)
    return idx / fs, np.stack([x[i:i + frame] for i in idx])


def short_time_energy(x: np.ndarray, fs: float, frame_ms: float = 25.0, hop_ms: float = 10.0,
                      db: bool = True):
    """Frame energy - the basic loudness contour and the front half of any VAD."""
    t, frames = frame_signal(x, fs, frame_ms, hop_ms)
    e = np.mean(frames ** 2, axis=1)
    return t, (10.0 * np.log10(np.maximum(e, EPS)) if db else e)


def short_time_zcr(x: np.ndarray, fs: float, frame_ms: float = 25.0, hop_ms: float = 10.0):
    """Zero-crossing rate per frame - high for fricatives, low for voiced sounds, so together
    with energy it separates voiced / unvoiced / silence."""
    t, frames = frame_signal(x, fs, frame_ms, hop_ms)
    return t, np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)


def autocorrelation(x: np.ndarray, fs: float, max_lag_ms: float = 30.0):
    """Normalised autocorrelation over positive lags - periodicity shows as a clear peak."""
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    if x.size < 4:
        return np.zeros(1), np.zeros(1)
    r = sig.correlate(x, x, mode="full")[x.size - 1:]
    r = r / max(r[0], EPS)
    lag = min(int(max_lag_ms * fs / 1000.0), r.size - 1)
    return np.arange(lag + 1) / fs, r[:lag + 1]


def pitch_track(x: np.ndarray, fs: float, fmin: float = 60.0, fmax: float = 400.0,
                frame_ms: float = 40.0, hop_ms: float = 10.0, threshold: float = 0.3):
    """Frame-wise F0 by autocorrelation peak-picking.

    Lags outside [fs/fmax, fs/fmin] are excluded so octave errors and the trivial zero-lag
    peak cannot win; frames whose best peak is below `threshold` are marked unvoiced (NaN),
    which is what keeps the contour from wandering through silence.
    """
    t, frames = frame_signal(x, fs, frame_ms, hop_ms)
    lag_min = max(int(fs / fmax), 2)
    lag_max = min(int(fs / fmin), frames.shape[1] - 1)
    if lag_max <= lag_min:
        return t, np.full(t.size, np.nan)
    f0 = np.full(frames.shape[0], np.nan)
    for i, fr in enumerate(frames):
        fr = fr - fr.mean()
        energy = float(fr @ fr)
        if energy < EPS:
            continue
        r = sig.correlate(fr, fr, mode="full")[fr.size - 1:] / energy
        seg = r[lag_min:lag_max]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg))
        if seg[k] >= threshold:
            f0[i] = fs / (lag_min + k)
    return t, f0


def envelope(x: np.ndarray, fs: float, smooth_hz: float = 20.0):
    """Analytic (Hilbert) envelope, low-pass smoothed - the amplitude contour of the signal."""
    if x.size < 8:
        return np.abs(x)
    env = np.abs(sig.hilbert(x))
    if smooth_hz and smooth_hz < fs / 2:
        b, a = sig.butter(2, smooth_hz / (fs / 2.0), btype="low")
        env = sig.filtfilt(b, a, env)
    return np.asarray(env, dtype=np.float64)


def instantaneous_frequency(x: np.ndarray, fs: float):
    """Derivative of the analytic signal's phase - tracks an FM sweep directly."""
    if x.size < 8:
        return np.zeros(1), np.zeros(1)
    phase = np.unwrap(np.angle(sig.hilbert(x)))
    f = np.diff(phase) * fs / (2.0 * np.pi)
    return np.arange(f.size) / fs, f


def dominant_frequency(x: np.ndarray, fs: float) -> float:
    """Frequency of the largest spectral peak above DC - the headline number on the dashboard."""
    if x.size < 8:
        return float("nan")
    f, mag = spectrum(x, fs, db=False)
    lo = max(int(1.0 * x.size / fs), 1)
    if lo >= mag.size:
        return float("nan")
    return float(f[lo + int(np.argmax(mag[lo:]))])


def spectral_centroid(x: np.ndarray, fs: float) -> float:
    """Magnitude-weighted mean frequency - the usual correlate of perceived brightness."""
    f, mag = spectrum(x, fs, db=False)
    total = float(np.sum(mag))
    return float(np.sum(f * mag) / total) if total > EPS else float("nan")


def spectral_bandwidth(x: np.ndarray, fs: float) -> float:
    f, mag = spectrum(x, fs, db=False)
    total = float(np.sum(mag))
    if total <= EPS:
        return float("nan")
    centre = float(np.sum(f * mag) / total)
    return float(np.sqrt(np.sum(((f - centre) ** 2) * mag) / total))


def spectral_rolloff(x: np.ndarray, fs: float, fraction: float = 0.85) -> float:
    """Frequency below which `fraction` of the total spectral energy lies."""
    f, mag = spectrum(x, fs, db=False)
    c = np.cumsum(mag)
    if c[-1] <= EPS:
        return float("nan")
    return float(f[int(np.searchsorted(c, fraction * c[-1]))])


def spectral_flatness(x: np.ndarray, fs: float) -> float:
    """Geometric over arithmetic mean of the spectrum: ~1 for noise, ~0 for a pure tone.

    The most direct single-number answer to "is this tonal or noisy?", and a good sanity
    check that a denoiser actually removed broadband noise.
    """
    _f, mag = spectrum(x, fs, db=False)
    mag = np.maximum(mag, EPS)
    return float(np.exp(np.mean(np.log(mag))) / np.mean(mag))


def summary(x: np.ndarray, fs: float) -> dict:
    """Compact spectral description used by the dashboard tiles and the exported report."""
    return {"Dominant frequency (Hz)": dominant_frequency(x, fs),
            "Spectral centroid (Hz)": spectral_centroid(x, fs),
            "Spectral bandwidth (Hz)": spectral_bandwidth(x, fs),
            "Spectral roll-off 85% (Hz)": spectral_rolloff(x, fs),
            "Spectral flatness": spectral_flatness(x, fs)}
