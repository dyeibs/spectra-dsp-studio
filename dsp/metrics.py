"""
Objective quality metrics.

These are what turn "the plot looks cleaner" into a defensible number.  Every enhancement
page reports the same set before and after processing, so an improvement can be stated as
a delta in dB rather than an impression.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal as sig

EPS = 1e-20


def _align(a: np.ndarray, b: np.ndarray):
    """Trim two signals to a common length - processing can shift length by a few samples."""
    n = min(a.size, b.size)
    return np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    r, t = _align(reference, test)
    return float(np.mean((r - t) ** 2)) if r.size else float("nan")


def rmse(reference: np.ndarray, test: np.ndarray) -> float:
    return float(np.sqrt(mse(reference, test)))


def mae(reference: np.ndarray, test: np.ndarray) -> float:
    r, t = _align(reference, test)
    return float(np.mean(np.abs(r - t))) if r.size else float("nan")


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """SNR of `test` against the clean `reference`: signal power over error power, in dB."""
    r, t = _align(reference, test)
    if not r.size:
        return float("nan")
    p_sig = float(np.sum(r ** 2))
    p_err = float(np.sum((r - t) ** 2))
    if p_err < EPS:
        return float("inf")
    return 10.0 * np.log10(max(p_sig, EPS) / p_err)


def segmental_snr_db(reference: np.ndarray, test: np.ndarray, fs: float,
                     frame_ms: float = 20.0) -> float:
    """Frame-wise SNR averaged over frames, clamped to [-10, 35] dB.

    The standard speech measure: a global SNR is dominated by the loudest passage, while
    segSNR weights every 20 ms frame equally and so tracks perceived quality far better.
    """
    r, t = _align(reference, test)
    n = max(int(fs * frame_ms / 1000.0), 1)
    if r.size < n:
        return snr_db(r, t)
    count = r.size // n
    r_f = r[:count * n].reshape(count, n)
    t_f = t[:count * n].reshape(count, n)
    p_sig = np.sum(r_f ** 2, axis=1)
    p_err = np.sum((r_f - t_f) ** 2, axis=1)
    active = p_sig > (np.mean(p_sig) * 1e-3)          # skip silent frames
    if not np.any(active):
        return snr_db(r, t)
    ratios = 10.0 * np.log10((p_sig[active] + EPS) / (p_err[active] + EPS))
    return float(np.mean(np.clip(ratios, -10.0, 35.0)))


def psnr_db(reference: np.ndarray, test: np.ndarray, peak: float | None = None) -> float:
    """Peak SNR - the image-world metric; `peak` defaults to the reference's dynamic range."""
    r, t = _align(np.ravel(reference), np.ravel(test))
    if not r.size:
        return float("nan")
    if peak is None:
        peak = float(np.max(np.abs(r))) or 1.0
    err = float(np.mean((r - t) ** 2))
    if err < EPS:
        return float("inf")
    return 10.0 * np.log10((peak ** 2) / err)


def correlation(reference: np.ndarray, test: np.ndarray) -> float:
    """Pearson correlation - how much of the reference's shape survived processing."""
    r, t = _align(reference, test)
    if r.size < 2:
        return float("nan")
    r = r - r.mean()
    t = t - t.mean()
    denom = float(np.sqrt(np.sum(r ** 2) * np.sum(t ** 2)))
    return float(np.sum(r * t) / denom) if denom > EPS else float("nan")


def thd_percent(x: np.ndarray, fs: float, fundamental: float | None = None,
                harmonics: int = 6) -> float:
    """Total harmonic distortion: RMS of harmonics 2..N over the fundamental, as a percentage.

    The fundamental is found from the spectral peak when not supplied; each harmonic is
    collected from a small bin neighbourhood so leakage does not under-count it.
    """
    if x.size < 16:
        return float("nan")
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    if fundamental is None:
        lo = max(int(1.0 * x.size / fs), 1)
        fundamental = float(freqs[lo + int(np.argmax(spec[lo:]))])
    if fundamental <= 0:
        return float("nan")

    def amp_at(f):
        if f >= fs / 2:
            return 0.0
        k = int(round(f * x.size / fs))
        lo, hi = max(k - 2, 0), min(k + 3, spec.size)
        return float(np.max(spec[lo:hi])) if hi > lo else 0.0

    a1 = amp_at(fundamental)
    if a1 < EPS:
        return float("nan")
    rest = np.sqrt(sum(amp_at(fundamental * k) ** 2 for k in range(2, harmonics + 2)))
    return float(100.0 * rest / a1)


def crest_factor_db(x: np.ndarray) -> float:
    """Peak-to-RMS in dB - how 'peaky' a signal is; clipping drives it toward 0."""
    r = float(np.sqrt(np.mean(x ** 2)))
    p = float(np.max(np.abs(x)))
    return 20.0 * np.log10(p / r) if r > EPS and p > EPS else float("nan")


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return 20.0 * np.log10(max(r, EPS))


def dynamic_range_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    floor = float(np.percentile(np.abs(x), 5)) if x.size else 0.0
    return 20.0 * np.log10(max(peak, EPS) / max(floor, EPS))


def zero_crossing_rate(x: np.ndarray) -> float:
    """Fraction of adjacent sample pairs that change sign - a cheap voiced/unvoiced cue."""
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if x.size > 1 else 0.0


def noise_floor_db(x: np.ndarray, fs: float) -> float:
    """Median of the Welch PSD in dB - a stable estimate of the residual noise level."""
    if x.size < 64:
        return float("nan")
    nper = min(1024, x.size)
    _f, pxx = sig.welch(x, fs=fs, nperseg=nper)
    return float(10.0 * np.log10(max(float(np.median(pxx)), EPS)))


# ----------------------------------------------------------------- image metrics
def ssim(reference: np.ndarray, test: np.ndarray, data_range: float | None = None,
         sigma: float = 1.5) -> float:
    """Structural similarity (Wang et al. 2004) on a Gaussian-weighted local window.

    Implemented directly on scipy.ndimage so the project does not depend on scikit-image:
    local means, variances and covariance are Gaussian-filtered, then combined with the
    standard C1/C2 stabilisers.  Returns the mean SSIM over the image, in [-1, 1].
    """
    r = np.asarray(reference, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    if r.ndim == 3:
        r = r.mean(axis=2)
    if t.ndim == 3:
        t = t.mean(axis=2)
    h, w = min(r.shape[0], t.shape[0]), min(r.shape[1], t.shape[1])
    r, t = r[:h, :w], t[:h, :w]
    if data_range is None:
        data_range = float(r.max() - r.min()) or 255.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    flt = lambda img: ndimage.gaussian_filter(img, sigma, truncate=3.5)  # noqa: E731

    mu_r, mu_t = flt(r), flt(t)
    rr, tt, rt = flt(r * r), flt(t * t), flt(r * t)
    var_r = rr - mu_r * mu_r
    var_t = tt - mu_t * mu_t
    cov = rt - mu_r * mu_t

    num = (2 * mu_r * mu_t + c1) * (2 * cov + c2)
    den = (mu_r ** 2 + mu_t ** 2 + c1) * (var_r + var_t + c2)
    return float(np.mean(num / np.maximum(den, EPS)))


def image_report(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> dict:
    """The four numbers the Image page shows under every before/after pair."""
    return {"MSE": mse(np.ravel(reference), np.ravel(test)),
            "RMSE": rmse(np.ravel(reference), np.ravel(test)),
            "PSNR (dB)": psnr_db(reference, test, peak=data_range),
            "SSIM": ssim(reference, test, data_range=data_range)}


def is_tonal(x: np.ndarray, fs: float, harmonics: int = 6) -> bool:
    """True when most of the energy sits on one fundamental and its harmonics.

    THD is only defined for a periodic tone.  On speech or noise the "fundamental" is just
    whichever bin happens to be loudest, and the percentage that comes out is meaningless, so
    it is reported only when at least half the energy is harmonic.
    """
    if x.size < 64:
        return False
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    total = float(spec.sum())
    if total < EPS:
        return False
    lo = max(int(1.0 * x.size / fs), 1)
    k1 = lo + int(np.argmax(spec[lo:]))
    tonal = 0.0
    for h in range(1, harmonics + 2):
        k = k1 * h
        if k >= spec.size:
            break
        tonal += float(spec[max(k - 3, 0):k + 4].sum())
    return tonal / total > 0.5


def activity_levels(x: np.ndarray, fs: float):
    """(background noise level, speech level) in dB, measured from the signal alone.

    For recordings with no clean copy.  Speech is intermittent, so the quietest 32 ms frames
    are mostly noise and the loudest mostly speech: the noise level is the 10th percentile of
    frame power, the speech level the mean of the loudest half.  Frames of digital silence
    are ignored - they are the recorder starting up, not the room.

    Compared before and after a clean-up, the two numbers answer the two questions that
    matter without a reference: how much quieter did the background get, and did the voice
    survive.  They are deliberately not combined into an "SNR": a denoiser silences the
    pauses far more than it improves the noise *under* the speech, so a pause-based SNR
    would overstate the gain several times over (measured: +15 dB claimed for +5 dB real).
    Before cleaning, though, their difference does track true SNR to within about 3 dB.
    """
    hop = max(int(0.016 * fs), 1)
    size = 2 * hop
    if x.size < 4 * size:
        return float("nan"), float("nan")
    # Frame power from a running sum of x^2: one vectorised pass instead of a Python loop
    # over every frame, which was most of the cost of redrawing the results panel.
    csum = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    idx = np.arange(0, x.size - size + 1, hop)
    power = np.maximum((csum[idx + size] - csum[idx]) / size, 0.0)
    if not np.any(power > 0):
        return float("nan"), float("nan")
    live = power[power > 1e-6 * float(np.median(power[power > 0]))]
    if live.size < 8:
        return float("nan"), float("nan")
    noise = max(float(np.percentile(live, 10)), EPS)
    # The loud frames hold speech *plus* the noise under it.  Subtracting the noise share is
    # what makes this a speech level: otherwise removing noise would look like the voice
    # getting quieter (by about 1.2 dB at 5 dB SNR, with the speech itself untouched).
    loud = max(float(np.mean(np.sort(live)[live.size // 2:])) - noise, EPS)
    return 10.0 * np.log10(noise), 10.0 * np.log10(loud)


def signal_report(reference: np.ndarray | None, test: np.ndarray, fs: float) -> dict:
    """Metric set for a 1-D signal.  Reference-based rows appear only when a clean reference
    exists (i.e. the noise was added in-app and the ground truth is genuinely known)."""
    out = {"RMS (dB)": rms_db(test),
           "Peak": float(np.max(np.abs(test))) if test.size else 0.0,
           "Crest factor (dB)": crest_factor_db(test),
           "Zero-crossing rate": zero_crossing_rate(test),
           "Noise floor (dB/Hz)": noise_floor_db(test, fs),
           "THD (%)": thd_percent(test, fs) if is_tonal(test, fs) else float("nan")}
    if reference is not None and reference.size:
        out.update({"SNR (dB)": snr_db(reference, test),
                    "Segmental SNR (dB)": segmental_snr_db(reference, test, fs),
                    "PSNR (dB)": psnr_db(reference, test),
                    "MSE": mse(reference, test),
                    "Correlation": correlation(reference, test)})
    return out
