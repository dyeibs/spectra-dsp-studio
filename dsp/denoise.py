"""
Noise reduction, restoration and signal enhancement.

The methods here span the three classical families, so the trade-offs can be demonstrated
rather than asserted:

  * Spectral-domain subtraction / Wiener  - estimate the noise spectrum, then remove it.
  * Statistical / adaptive (LMS, NLMS)    - learn the noise from a correlated reference.
  * Non-linear and basis methods          - median, Savitzky-Golay, wavelet thresholding.

Each function takes and returns a float64 mono array at the given sample rate, so they can
be chained freely by the UI's processing pipeline.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy import signal as sig

from . import filters as flt

METHODS = ("Spectral subtraction", "Wiener filter (frequency domain)", "Wiener filter (adaptive local)",
           "Wavelet thresholding", "Adaptive LMS", "Adaptive NLMS", "Savitzky-Golay smoothing",
           "Median filter (de-click)", "Moving average", "Notch - mains hum removal",
           "Band-pass (speech band)", "Noise gate", "Spectral gating")

THRESHOLD_RULES = ("bayes", "universal", "sure", "minimax")
WAVELETS = ("db4", "db8", "sym5", "sym8", "coif3", "haar", "bior3.5")
EPS = 1e-12


# ----------------------------------------------------------------- helpers
def _stft(x, fs, frame, hop):
    win = np.hanning(frame)
    f, t, Z = sig.stft(x, fs=fs, window=win, nperseg=frame, noverlap=frame - hop,
                       boundary="zeros", padded=True)
    return f, t, Z, win


def _istft(Z, fs, frame, hop, length):
    win = np.hanning(frame)
    _t, y = sig.istft(Z, fs=fs, window=win, nperseg=frame, noverlap=frame - hop,
                      boundary=True)
    y = np.asarray(y, dtype=np.float64)
    if y.size < length:
        y = np.pad(y, (0, length - y.size))
    return y[:length]


def _frame_count(fs, frame, hop, seconds):
    return max(int(seconds * fs / hop), 1)


def estimate_noise_spectrum(Z: np.ndarray, mode: str = "auto", noise_frames: int = 8,
                            percentile: float = 15.0) -> np.ndarray:
    """Per-bin noise magnitude estimate, shaped (n_freq, 1).

    Three strategies, because which one is right depends entirely on the recording:

      * "auto"    - minimum statistics: a low percentile of each bin's magnitude across all
                    frames.  Speech and music are intermittent in any given bin, so the lower
                    tail of that distribution is dominated by noise even when the signal never
                    stops.  This is the only one that is safe without prior knowledge.
      * "leading" - mean of the first `noise_frames` frames; correct only when the recording
                    genuinely opens with silence, and badly wrong (over-subtraction) when not.
      * "clip"    - caller supplies a noise-only excerpt; handled by the callers.
    """
    mag = np.abs(Z)
    if mag.shape[1] == 0:
        return np.zeros((mag.shape[0], 1))
    if mode == "leading":
        k = max(min(int(noise_frames), mag.shape[1]), 1)
        return np.mean(mag[:, :k], axis=1, keepdims=True)

    # Digital silence is not noise.  A microphone recording usually opens with a few hundred
    # milliseconds of exact zeros while the device starts, and those frames sit at the very
    # bottom of every bin's distribution - so they drag a low percentile towards zero and the
    # filter concludes there is almost no noise to remove.
    energy = np.sum(mag ** 2, axis=0)
    live = energy > 1e-6 * float(np.median(energy[energy > 0])) if np.any(energy > 0) \
        else np.ones(energy.size, bool)
    if int(live.sum()) >= 4:
        mag = mag[:, live]

    # A low percentile also under-reads the noise by itself: noise magnitudes in a bin are
    # Rayleigh-distributed, and the 15th percentile of a Rayleigh is only 0.455x its mean.
    # Full compensation (x2.2) over-subtracts wherever speech lifts the low tail, so the
    # factor is a measured compromise - see NOISE_BIAS.
    return NOISE_BIAS * np.percentile(mag, float(np.clip(percentile, 1.0, 50.0)), axis=1,
                                      keepdims=True)


#: Bias correction for the minimum-statistics noise estimate.  Swept on real speech (an
#: 8-second narration excerpt, two segments) and synthetic speech, in white, pink and babble
#: noise at 0, 5 and 10 dB SNR, with and without a 0.4 s silent lead-in: the Wiener filter's
#: average SNR gain rose from +2.7 dB uncorrected to +5.2 dB at 1.5, with the worst single
#: case still improving by +1.5 dB.  Above 1.5 the average barely moves while the worst case
#: falls toward zero, which is over-subtraction starting to eat the speech.
NOISE_BIAS = 1.5


def _noise_mag(x, fs, frame, hop, noise_clip, noise_mode, noise_seconds, Z):
    """Resolve the noise magnitude spectrum from whichever evidence the caller provided."""
    if noise_clip is not None and np.size(noise_clip) >= frame:
        _f, _t, Zn, _w = _stft(np.asarray(noise_clip, dtype=np.float64), fs, frame, hop)
        return np.mean(np.abs(Zn), axis=1, keepdims=True)
    return estimate_noise_spectrum(Z, mode=noise_mode,
                                   noise_frames=_frame_count(fs, frame, hop, noise_seconds))


# ----------------------------------------------------------------- spectral methods
def spectral_subtraction(x: np.ndarray, fs: float, alpha: float = 2.0, beta: float = 0.02,
                         frame_ms: float = 32.0, overlap: float = 0.75,
                         noise_seconds: float = 0.25, noise_clip: np.ndarray | None = None,
                         noise_mode: str = "auto", smooth: bool = True) -> np.ndarray:
    """Boll's spectral subtraction.

    Subtract `alpha` times the estimated noise magnitude from every frame's magnitude, keep
    the noisy phase, and floor the result at `beta` times the noise magnitude.  The floor is
    what controls "musical noise": with beta = 0 the isolated surviving bins ring as tones,
    so a small floor trades a little residual hiss for a much less annoying artefact.
    """
    frame = int(2 ** round(np.log2(max(frame_ms * fs / 1000.0, 32))))
    hop = max(int(frame * (1.0 - overlap)), 1)
    _f, _t, Z, _w = _stft(x, fs, frame, hop)

    noise_mag = _noise_mag(x, fs, frame, hop, noise_clip, noise_mode, noise_seconds, Z)

    mag, phase = np.abs(Z), np.angle(Z)
    clean = mag - alpha * noise_mag
    clean = np.maximum(clean, beta * noise_mag)        # spectral floor
    if smooth:                                         # mild time smoothing of the gain
        gain = clean / np.maximum(mag, EPS)
        gain = sig.lfilter([0.4], [1.0, -0.6], gain, axis=1)
        clean = gain * mag
    return _istft(clean * np.exp(1j * phase), fs, frame, hop, x.size)


def wiener_frequency(x: np.ndarray, fs: float, frame_ms: float = 32.0, overlap: float = 0.75,
                     noise_seconds: float = 0.25, noise_clip: np.ndarray | None = None,
                     noise_mode: str = "auto", max_atten_db: float = 18.0) -> np.ndarray:
    """Frequency-domain Wiener filter via a decision-directed a-priori SNR estimate.

    The gain in each bin is SNR/(1+SNR), the MMSE-optimal scaling.  Unlike plain subtraction
    the gain is bounded and varies smoothly, so it produces much less musical noise; the
    `max_atten_db` floor keeps the suppression from sounding gated.
    """
    frame = int(2 ** round(np.log2(max(frame_ms * fs / 1000.0, 32))))
    hop = max(int(frame * (1.0 - overlap)), 1)
    _f, _t, Z, _w = _stft(x, fs, frame, hop)

    noise_pow = np.maximum(
        _noise_mag(x, fs, frame, hop, noise_clip, noise_mode, noise_seconds, Z) ** 2, EPS)

    mag_pow = np.abs(Z) ** 2
    floor = 10.0 ** (-abs(max_atten_db) / 20.0)
    gain = np.empty_like(mag_pow)
    prev = np.zeros((Z.shape[0], 1))
    xi_min, a_dd = 10.0 ** (-2.0), 0.98
    for i in range(Z.shape[1]):
        post = mag_pow[:, i:i + 1] / noise_pow - 1.0            # a-posteriori SNR - 1
        xi = a_dd * prev + (1.0 - a_dd) * np.maximum(post, 0.0)  # decision-directed a-priori
        xi = np.maximum(xi, xi_min)
        g = xi / (1.0 + xi)
        gain[:, i:i + 1] = np.maximum(g, floor)
        prev = (g ** 2) * mag_pow[:, i:i + 1] / noise_pow
    return _istft(Z * gain, fs, frame, hop, x.size)


def spectral_gating(x: np.ndarray, fs: float, threshold_db: float = 6.0, frame_ms: float = 32.0,
                    overlap: float = 0.75, noise_seconds: float = 0.25,
                    noise_mode: str = "auto", freq_smooth: int = 3,
                    time_smooth: int = 3) -> np.ndarray:
    """Noise gating per frequency bin: bins quieter than the noise profile plus a margin are
    attenuated.  The 2-D smoothing of the mask is what stops the gate from chattering."""
    frame = int(2 ** round(np.log2(max(frame_ms * fs / 1000.0, 32))))
    hop = max(int(frame * (1.0 - overlap)), 1)
    _f, _t, Z, _w = _stft(x, fs, frame, hop)
    mag = np.abs(Z)
    prof = _noise_mag(x, fs, frame, hop, None, noise_mode, noise_seconds, Z)
    thresh = prof * (10.0 ** (threshold_db / 20.0))
    mask = (mag > thresh).astype(np.float64)
    if freq_smooth > 1 or time_smooth > 1:
        kern = np.outer(np.hanning(max(freq_smooth, 1) + 2)[1:-1],
                        np.hanning(max(time_smooth, 1) + 2)[1:-1])
        kern /= max(kern.sum(), EPS)
        mask = sig.convolve2d(mask, kern, mode="same", boundary="symm")
    return _istft(Z * np.clip(mask, 0.05, 1.0), fs, frame, hop, x.size)


def wiener_local(x: np.ndarray, size: int = 5, noise_var: float | None = None) -> np.ndarray:
    """Lee's adaptive local Wiener filter in the time domain.

    Smooths hard where the local variance is near the noise variance and backs off where the
    signal is active, so transients survive better than under a fixed low-pass.
    """
    size = max(int(size), 3)
    y = sig.wiener(np.asarray(x, dtype=np.float64), mysize=size, noise=noise_var)
    return np.nan_to_num(np.asarray(y, dtype=np.float64))


# ----------------------------------------------------------------- adaptive filters
def lms(x: np.ndarray, reference: np.ndarray, mu: float = 0.01, order: int = 32,
        normalised: bool = False):
    """Adaptive (N)LMS filter.  Returns (error, prediction, learning curve in dB).

    `x` is the primary input and `reference` the adaptive filter's input.  Which of the two
    outputs is "the enhanced signal" depends on how the pair is wired:

      * Noise cancellation - `reference` correlates with the noise only, so the filter learns
        the noise path and the **error** converges to the clean signal.
      * Line enhancement (ALE) - `reference` is a delayed copy of `x`.  The delay decorrelates
        the broadband noise but not the periodic component, so the **prediction** is the
        enhanced signal and the error is the rejected noise.

    `mu` is the step size: too large diverges, too small never converges.  NLMS divides by the
    instantaneous input power, which makes one `mu` work across very different signal levels.
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(reference, dtype=np.float64)
    n = min(x.size, d.size)
    x, d = x[:n], d[:n]
    order = int(np.clip(order, 2, 512))
    w = np.zeros(order)
    err_sig = np.zeros(n)
    pred = np.zeros(n)
    curve = np.zeros(n)
    buf = np.zeros(order)
    for i in range(n):
        buf[1:] = buf[:-1]
        buf[0] = d[i]
        yhat = float(w @ buf)
        err = x[i] - yhat
        step = mu / (float(buf @ buf) + 1e-6) if normalised else mu
        w += 2.0 * step * err * buf
        err_sig[i] = err
        pred[i] = yhat
        curve[i] = err * err
    curve = 10.0 * np.log10(np.maximum(sig.lfilter([0.01], [1.0, -0.99], curve), EPS))
    return err_sig, pred, curve


def nlms(x, reference, mu: float = 0.5, order: int = 32):
    return lms(x, reference, mu=mu, order=order, normalised=True)


def adaptive_line_enhancer(x: np.ndarray, mu: float = 0.02, order: int = 96,
                           delay: int = 4, normalised: bool = False):
    """ALE: pull a periodic signal out of broadband noise with no external reference.

    The decorrelation `delay` is the whole trick - after a few samples white noise is
    uncorrelated with itself while a periodic component still is, so the predictor can only
    reproduce the periodic part.  Returns (enhanced, rejected noise, learning curve).
    """
    delay = max(int(delay), 1)
    ref = np.concatenate([np.zeros(delay), x[:-delay]]) if delay < x.size else np.zeros_like(x)
    err_sig, pred, curve = lms(x, ref, mu=mu, order=order, normalised=normalised)
    return pred, err_sig, curve


# ----------------------------------------------------------------- basis / non-linear
def wavelet_denoise(x: np.ndarray, wavelet: str = "db4", level: int | None = None,
                    mode: str = "soft", rule: str = "bayes",
                    threshold_scale: float = 1.0) -> np.ndarray:
    """Wavelet shrinkage (Donoho-Johnstone).

    Decompose, shrink the detail coefficients toward zero, reconstruct.  Noise spreads thinly
    over every coefficient while signal energy concentrates in a few large ones, so a single
    threshold separates them.  The noise level is estimated robustly from the finest detail
    band as sigma = MAD/0.6745, which is insensitive to the signal's own large coefficients.

    `mode` 'soft' shrinks every coefficient (smoother, slight bias); 'hard' keeps or kills
    (sharper, but can ring).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return x.copy()
    max_level = pywt.dwt_max_level(x.size, pywt.Wavelet(wavelet).dec_len)
    level = int(np.clip(level or max(max_level // 2, 1), 1, max(max_level, 1)))
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")

    # Noise level from the finest detail band: MAD/0.6745 is a robust sigma that the signal's
    # own large coefficients cannot inflate the way a plain standard deviation would.
    sigma = float(np.median(np.abs(coeffs[-1])) / 0.6745)
    n = max(x.size, 2)

    def band_threshold(c: np.ndarray) -> float:
        if rule == "bayes":
            # BayesShrink: T = sigma^2 / sigma_signal, computed per sub-band.  Adapting the
            # threshold to each band is what keeps it from flattening the coarse levels,
            # where the signal lives and the universal threshold is far too aggressive.
            var_y = float(np.mean(c ** 2))
            sigma_x = np.sqrt(max(var_y - sigma ** 2, 1e-20))
            return float(sigma ** 2 / sigma_x) if sigma_x > 1e-10 else float(np.max(np.abs(c)))
        if rule == "sure":
            return sigma * np.sqrt(2.0 * np.log(n) / max(np.log2(n), 1.0))
        if rule == "minimax":
            return sigma * (0.3936 + 0.1829 * np.log2(n))
        return sigma * np.sqrt(2.0 * np.log(n))         # universal / VisuShrink

    scale = float(threshold_scale)
    out = [coeffs[0]] + [pywt.threshold(c, max(band_threshold(c) * scale, 0.0), mode=mode)
                         for c in coeffs[1:]]
    y = pywt.waverec(out, wavelet, mode="periodization")
    return np.asarray(y[:x.size], dtype=np.float64)


def median_declick(x: np.ndarray, size: int = 5, only_outliers: bool = True,
                   sigma: float = 3.0) -> np.ndarray:
    """Impulse / click removal.

    With `only_outliers` the median is substituted *only* where a sample deviates from the
    local median by more than `sigma` robust deviations.  That keeps the rest of the waveform
    bit-exact, where a plain median filter would dull the whole signal.
    """
    med = flt.median_smooth(x, size)
    if not only_outliers:
        return med
    resid = x - med

    # The robust scale must be estimated from the samples that actually vary.  A signal with
    # real pauses is exactly zero over much of its length, so the residual is zero there too;
    # a plain MAD over everything then returns 0, the threshold collapses, and the filter
    # silently does nothing - which is precisely what happened on speech with silent gaps.
    active = resid[np.abs(resid) > EPS]
    if active.size < 8:
        return x.copy()
    scale = float(np.median(np.abs(active - np.median(active))) / 0.6745)
    if scale < EPS:
        scale = float(np.std(active))
    if scale < EPS:
        return x.copy()

    bad = np.abs(resid) > sigma * scale
    y = x.copy()
    y[bad] = med[bad]
    return y


def noise_gate(x: np.ndarray, fs: float, threshold_db: float = -40.0, attack_ms: float = 5.0,
               release_ms: float = 60.0) -> np.ndarray:
    """Classic downward expander: follow the envelope, open above the threshold, close below.

    Attack/release times are what separate a gate from a hard mute - instant switching
    produces audible clicks at every transition.
    """
    env = np.abs(sig.hilbert(x)) if x.size > 8 else np.abs(x)
    a_att = np.exp(-1.0 / max(fs * attack_ms / 1000.0, 1.0))
    a_rel = np.exp(-1.0 / max(fs * release_ms / 1000.0, 1.0))
    thr = 10.0 ** (threshold_db / 20.0) * max(float(np.max(np.abs(x))), EPS)
    gain = np.zeros_like(env)
    g = 0.0
    for i, e in enumerate(env):
        target = 1.0 if e > thr else 0.0
        coeff = a_att if target > g else a_rel
        g = coeff * g + (1.0 - coeff) * target
        gain[i] = g
    return x * gain


def remove_hum(x: np.ndarray, fs: float, base: float = 60.0, harmonics: int = 4,
               q: float = 30.0) -> np.ndarray:
    """Cascade of notch filters at the mains frequency and its harmonics.

    Hum is never a single tone - the harmonics carry much of the audible energy, so notching
    only the fundamental leaves an obvious buzz behind.
    """
    y = np.asarray(x, dtype=np.float64)
    for k in range(1, max(int(harmonics), 1) + 1):
        f = base * k
        if f >= fs / 2.0 * 0.98:
            break
        y = flt.apply_filter(flt.design_notch(f, fs, q), y, zero_phase=True)
    return y


def speech_bandpass(x: np.ndarray, fs: float, low: float = 300.0, high: float = 3400.0,
                    order: int = 6) -> np.ndarray:
    """Restrict to the telephony speech band - removes rumble and hiss outside intelligibility."""
    high = min(high, fs / 2.0 * 0.98)
    low = max(min(low, high * 0.5), 1.0)
    return flt.apply_filter(flt.design_iir("Butterworth", "Band-pass", (low, high), fs, order), x)


def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - a*x[n-1].  Boosts the high band to offset the -6 dB/octave roll-off of
    the glottal source; standard front-end step before speech feature extraction."""
    return np.append(x[:1], x[1:] - coeff * x[:-1])


def de_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return sig.lfilter([1.0], [1.0, -coeff], x)


def normalise_peak(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > EPS else x.copy()


def normalise_rms(x: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    r = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    if r < EPS:
        return x.copy()
    y = x * (10.0 ** (target_db / 20.0) / r)
    peak = float(np.max(np.abs(y)))
    return y / peak * 0.99 if peak > 0.99 else y


def dc_remove(x: np.ndarray) -> np.ndarray:
    return x - float(np.mean(x)) if x.size else x.copy()


# ----------------------------------------------------------------- dispatcher
def enhance(method: str, x: np.ndarray, fs: float, params: dict | None = None):
    """Apply one of METHODS by name.  Returns (enhanced signal, info dict for the UI).

    Centralising the dispatch keeps the UI free of per-method branching and makes the
    pipeline replayable from a saved list of (method, params) pairs.
    """
    p = dict(params or {})
    info: dict = {}

    if method == "Spectral subtraction":
        y = spectral_subtraction(x, fs, alpha=p.get("alpha", 2.0), beta=p.get("beta", 0.02),
                                 frame_ms=p.get("frame_ms", 32.0),
                                 noise_seconds=p.get("noise_seconds", 0.25),
                                 noise_clip=p.get("noise_clip"))
    elif method == "Wiener filter (frequency domain)":
        y = wiener_frequency(x, fs, frame_ms=p.get("frame_ms", 32.0),
                             noise_seconds=p.get("noise_seconds", 0.25),
                             noise_clip=p.get("noise_clip"),
                             max_atten_db=p.get("max_atten_db", 18.0))
    elif method == "Wiener filter (adaptive local)":
        y = wiener_local(x, size=int(p.get("size", 5)))
    elif method == "Wavelet thresholding":
        y = wavelet_denoise(x, wavelet=p.get("wavelet", "db4"), level=p.get("level"),
                            mode=p.get("mode", "soft"), rule=p.get("rule", "bayes"),
                            threshold_scale=p.get("threshold_scale", 1.0))
    elif method in ("Adaptive LMS", "Adaptive NLMS"):
        norm = method == "Adaptive NLMS"
        mu = p.get("mu", 0.05 if norm else 0.02)
        order = int(p.get("order", 96))
        ref = p.get("reference")
        if ref is None:
            # No noise reference available: fall back to line enhancement, where a delayed
            # copy of the input stands in and the *prediction* is the enhanced signal.
            y, rejected, curve = adaptive_line_enhancer(
                x, mu=mu, order=order, delay=int(p.get("delay", 4)), normalised=norm)
            info["note"] = ("No external noise reference - ran as an adaptive line enhancer "
                            f"with a {int(p.get('delay', 4))}-sample decorrelation delay.")
            info["rejected"] = rejected
        else:
            y, _pred, curve = lms(x, np.asarray(ref, dtype=np.float64), mu=mu, order=order,
                                  normalised=norm)
            info["note"] = "Noise cancellation against the supplied reference."
        info["learning_curve"] = curve
    elif method == "Savitzky-Golay smoothing":
        y = flt.savitzky_golay(x, window=int(p.get("window", 7)), poly=int(p.get("poly", 3)))
    elif method == "Median filter (de-click)":
        y = median_declick(x, size=int(p.get("size", 5)),
                           only_outliers=p.get("only_outliers", True),
                           sigma=p.get("sigma", 3.0))
    elif method == "Moving average":
        y = flt.moving_average(x, size=int(p.get("size", 3)))
    elif method == "Notch - mains hum removal":
        y = remove_hum(x, fs, base=p.get("base", 60.0), harmonics=int(p.get("harmonics", 4)),
                       q=p.get("q", 30.0))
    elif method == "Band-pass (speech band)":
        y = speech_bandpass(x, fs, low=p.get("low", 300.0), high=p.get("high", 3400.0),
                            order=int(p.get("order", 6)))
    elif method == "Noise gate":
        y = noise_gate(x, fs, threshold_db=p.get("threshold_db", -40.0),
                       attack_ms=p.get("attack_ms", 5.0), release_ms=p.get("release_ms", 60.0))
    elif method == "Spectral gating":
        y = spectral_gating(x, fs, threshold_db=p.get("threshold_db", 6.0),
                            frame_ms=p.get("frame_ms", 32.0),
                            noise_seconds=p.get("noise_seconds", 0.25))
    else:
        raise ValueError("Unknown enhancement method: " + str(method))

    y = np.nan_to_num(np.asarray(y, dtype=np.float64))
    if p.get("normalise", False):
        y = normalise_peak(y, 0.95)
    return y, info
