"""
Signal synthesis and controlled degradation.

Every generator returns a float64 mono array in roughly [-1, 1] so the rest of the pipeline
(filters, enhancement, metrics, playback) can assume a single convention.  The degradation
helpers are the other half of the story: they inject *known* noise at a *known* SNR, so the
enhancement pages can report an honest before/after improvement instead of a guess.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig

TWO_PI = 2.0 * np.pi

WAVEFORMS = ("Sine", "Cosine", "Square", "Sawtooth", "Triangle", "Chirp (linear)",
             "Chirp (logarithmic)", "Impulse train", "Unit step", "Gaussian pulse",
             "Damped sine", "AM (amplitude mod.)", "FM (frequency mod.)", "Multi-tone",
             "White noise", "Pink noise", "Brown noise", "ECG-like", "Speech-like formants")

NOISE_KINDS = ("White Gaussian (AWGN)", "Pink (1/f)", "Brown (1/f^2)", "Impulse / click",
               "Mains hum (50 Hz)", "Mains hum (60 Hz)", "Narrowband tone", "Uniform")


def time_axis(duration: float, fs: float) -> np.ndarray:
    """Sample instants for `duration` seconds at `fs` Hz - the n/fs sampled time base."""
    return np.arange(int(round(duration * fs)), dtype=np.float64) / float(fs)


def _normalise(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > 1e-12 else x


# ----------------------------------------------------------------- coloured noise
def white_noise(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.standard_normal(n)


def pink_noise(n: int, rng=None) -> np.ndarray:
    """1/f noise by shaping a white spectrum - amplitude falls as 1/sqrt(f), -3 dB/octave."""
    rng = rng or np.random.default_rng()
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    return _normalise(np.fft.irfft(spec / np.sqrt(f), n), 1.0)


def brown_noise(n: int, rng=None) -> np.ndarray:
    """1/f^2 (random-walk) noise - integrated white noise with the DC drift removed."""
    rng = rng or np.random.default_rng()
    x = np.cumsum(rng.standard_normal(n))
    return _normalise(x - x.mean(), 1.0)


# ----------------------------------------------------------------- generators
def generate(kind: str, duration: float = 2.0, fs: float = 8000.0, freq: float = 440.0,
             amplitude: float = 0.8, freq2: float = 1200.0, mod_index: float = 5.0,
             duty: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Synthesize one of WAVEFORMS.  `freq2` is the second frequency (chirp stop, modulator,
    partner tone) and `mod_index` the AM depth / FM deviation index."""
    rng = np.random.default_rng(seed)
    t = time_axis(duration, fs)
    n = t.size
    if n == 0:
        return np.zeros(0)

    if kind == "Sine":
        x = np.sin(TWO_PI * freq * t)
    elif kind == "Cosine":
        x = np.cos(TWO_PI * freq * t)
    elif kind == "Square":
        x = sig.square(TWO_PI * freq * t, duty=duty)
    elif kind == "Sawtooth":
        x = sig.sawtooth(TWO_PI * freq * t)
    elif kind == "Triangle":
        x = sig.sawtooth(TWO_PI * freq * t, width=0.5)
    elif kind == "Chirp (linear)":
        x = sig.chirp(t, f0=freq, f1=freq2, t1=max(t[-1], 1e-9), method="linear")
    elif kind == "Chirp (logarithmic)":
        x = sig.chirp(t, f0=max(freq, 1.0), f1=max(freq2, 1.0), t1=max(t[-1], 1e-9),
                      method="logarithmic")
    elif kind == "Impulse train":
        x = np.zeros(n)
        x[::max(int(fs / max(freq, 1.0)), 1)] = 1.0
    elif kind == "Unit step":
        x = np.where(t >= duration / 2.0, 1.0, 0.0)
    elif kind == "Gaussian pulse":
        x = sig.gausspulse(t - duration / 2.0, fc=max(freq, 1.0), bw=0.4)
    elif kind == "Damped sine":
        x = np.exp(-3.0 * t / max(duration, 1e-9)) * np.sin(TWO_PI * freq * t)
    elif kind == "AM (amplitude mod.)":
        depth = float(np.clip(mod_index / 10.0, 0.0, 1.0))
        x = (1.0 + depth * np.sin(TWO_PI * freq2 * t)) * np.sin(TWO_PI * freq * t)
    elif kind == "FM (frequency mod.)":
        x = np.sin(TWO_PI * freq * t + mod_index * np.sin(TWO_PI * freq2 * t))
    elif kind == "Multi-tone":
        x = sum(np.sin(TWO_PI * freq * k * t) / k for k in (1, 2, 3, 5))
    elif kind == "White noise":
        x = white_noise(n, rng)
    elif kind == "Pink noise":
        x = pink_noise(n, rng)
    elif kind == "Brown noise":
        x = brown_noise(n, rng)
    elif kind == "ECG-like":
        x = _ecg(t, bpm=freq if 30.0 <= freq <= 220.0 else 72.0)
    elif kind == "Speech-like formants":
        x = _speech_like(t, f0=float(np.clip(freq, 80.0, 320.0)), rng=rng)
    else:
        raise ValueError("Unknown waveform: " + str(kind))

    return _normalise(np.asarray(x, dtype=np.float64), amplitude)


def _ecg(t: np.ndarray, bpm: float = 72.0) -> np.ndarray:
    """Synthetic ECG: P, Q, R, S and T deflections as Gaussians placed on each beat period."""
    period = 60.0 / bpm
    phase = np.mod(t, period) / period
    waves = ((0.12, 0.16, 0.030), (-0.14, 0.36, 0.011), (1.00, 0.40, 0.009),
             (-0.22, 0.44, 0.011), (0.30, 0.62, 0.045))
    return sum(a * np.exp(-((phase - mu) ** 2) / (2 * s ** 2)) for a, mu, s in waves)


def _speech_like(t: np.ndarray, f0: float = 120.0, rng=None) -> np.ndarray:
    """A voiced vowel: a glottal pulse train shaped by three formant resonances, gated into
    syllable-like bursts separated by pauses.

    The pauses are not decoration.  Every noise-estimation method that works without a
    hand-marked noise region (minimum statistics, the leading-frames heuristic, VAD-driven
    updates) depends on the signal falling silent sometimes.  A continuously voiced tone
    complex is pathological for all of them and would make the enhancement pages look broken
    for reasons that have nothing to do with the enhancement itself.
    """
    rng = rng or np.random.default_rng()
    fs = 1.0 / (t[1] - t[0]) if t.size > 1 else 8000.0
    excitation = np.zeros(t.size)
    excitation[::max(int(fs / f0), 1)] = 1.0
    excitation += 0.02 * rng.standard_normal(t.size)
    out = np.zeros_like(excitation)
    for fc, bw, gain in ((700.0, 90.0, 1.0), (1220.0, 110.0, 0.55), (2600.0, 160.0, 0.28)):
        if fc >= fs / 2:
            continue
        r = np.exp(-np.pi * bw / fs)
        theta = TWO_PI * fc / fs
        out += gain * sig.lfilter([1.0 - r], [1.0, -2.0 * r * np.cos(theta), r * r], excitation)

    # Syllable gate: ~280 ms of speech, ~170 ms of silence, with cosine-tapered edges so the
    # transitions do not themselves look like impulses to the analysis.
    gate = np.zeros(t.size)
    period = 0.45
    taper = max(int(0.02 * fs), 1)
    for start in np.arange(0.05, t[-1] if t.size else 0.0, period):
        a = int(start * fs)
        b = min(a + int(0.28 * fs), t.size)
        if b <= a:
            continue
        gate[a:b] = 1.0
        ramp = np.linspace(0.0, 1.0, min(taper, (b - a) // 2) or 1)
        gate[a:a + ramp.size] = ramp
        gate[b - ramp.size:b] = ramp[::-1]
    envelope = gate * (0.65 + 0.35 * np.sin(TWO_PI * 3.1 * t) ** 2)
    return out * envelope


# ----------------------------------------------------------------- degradation
def _scale_to_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so that 10*log10(Px/Pn) == snr_db exactly, then add it to `x`."""
    p_sig = float(np.mean(x ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise < 1e-20 or p_sig < 1e-20:
        return x + noise
    target = p_sig / (10.0 ** (snr_db / 10.0))
    return x + noise * np.sqrt(target / p_noise)


def awgn(x: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Additive white Gaussian noise scaled so the result sits at exactly `snr_db`."""
    rng = rng or np.random.default_rng()
    return _scale_to_snr(x, rng.standard_normal(x.size), snr_db)


def add_noise(x: np.ndarray, fs: float, kind: str = "White Gaussian (AWGN)", snr_db: float = 10.0,
              density: float = 0.02, tone_hz: float = 1000.0, seed: int | None = None) -> np.ndarray:
    """Corrupt `x` with one of NOISE_KINDS.  Additive kinds honour `snr_db`; impulse noise uses
    `density` (the fraction of samples hit) because clicks are not well described by an SNR."""
    rng = np.random.default_rng(seed)
    n = x.size

    if kind == "White Gaussian (AWGN)":
        return _scale_to_snr(x, rng.standard_normal(n), snr_db)
    if kind == "Pink (1/f)":
        return _scale_to_snr(x, pink_noise(n, rng), snr_db)
    if kind == "Brown (1/f^2)":
        return _scale_to_snr(x, brown_noise(n, rng), snr_db)
    if kind == "Uniform":
        return _scale_to_snr(x, rng.uniform(-1.0, 1.0, n), snr_db)
    if kind == "Impulse / click":
        y = x.copy()
        count = max(int(density * n), 1)
        idx = rng.choice(n, size=count, replace=False)
        peak = max(float(np.max(np.abs(x))), 1e-6)
        y[idx] = rng.choice([-1.0, 1.0], size=count) * peak * rng.uniform(1.2, 2.5, count)
        return y
    if kind in ("Mains hum (50 Hz)", "Mains hum (60 Hz)"):
        base = 50.0 if "50" in kind else 60.0
        t = np.arange(n) / fs
        hum = sum(np.sin(TWO_PI * base * k * t) / k for k in (1, 2, 3) if base * k < fs / 2)
        return _scale_to_snr(x, np.asarray(hum, dtype=np.float64), snr_db)
    if kind == "Narrowband tone":
        t = np.arange(n) / fs
        return _scale_to_snr(x, np.sin(TWO_PI * min(tone_hz, fs / 2 - 1.0) * t), snr_db)
    raise ValueError("Unknown noise kind: " + str(kind))


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Uniform mid-tread quantisation to `bits` - the amplitude half of A/D conversion."""
    levels = 2 ** int(np.clip(bits, 1, 24))
    peak = max(float(np.max(np.abs(x))), 1e-12)
    step = 2.0 * peak / levels
    return np.round(x / step) * step


def downsample_demo(x: np.ndarray, fs: float, factor: int, anti_alias: bool = True):
    """Decimate by `factor`, optionally *without* the anti-alias filter so folding becomes
    audible and visible.  Returns (signal, new rate) - the sampling-theorem demonstration."""
    factor = max(int(factor), 1)
    if factor == 1:
        return x.copy(), fs
    y = sig.decimate(x, factor, ftype="fir", zero_phase=True) if anti_alias else x[::factor]
    return np.asarray(y, dtype=np.float64), fs / factor


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase rational resampling - used when a file's rate differs from the session rate."""
    if abs(fs_in - fs_out) < 1e-6:
        return x.copy()
    from math import gcd
    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    return np.asarray(sig.resample_poly(x, a // g, b // g), dtype=np.float64)
