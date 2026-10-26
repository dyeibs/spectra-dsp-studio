"""
Digital filter design and application.

Two families are offered side by side so the difference can be demonstrated:

  * FIR - designed by the window method or by Parks-McClellan (remez).  Always stable, and
    exactly linear phase, at the cost of a high order.
  * IIR - Butterworth / Chebyshev I / Chebyshev II / Elliptic / Bessel, designed as
    second-order sections (SOS) for numerical stability at high order.  Far cheaper for the
    same selectivity, but the phase is non-linear.

Everything is returned in a single `FilterSpec` so the UI can plot magnitude, phase, group
delay, impulse/step response and the pole-zero map from one object.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sig

FIR_WINDOWS = ("hamming", "hann", "blackman", "blackmanharris", "bartlett", "boxcar",
               "flattop", "nuttall", "kaiser")
IIR_FAMILIES = ("Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic", "Bessel")
BAND_TYPES = ("Low-pass", "High-pass", "Band-pass", "Band-stop")
FIR_METHODS = ("Window method", "Parks-McClellan (remez)", "Least-squares (firls)")

_SCIPY_BAND = {"Low-pass": "lowpass", "High-pass": "highpass",
               "Band-pass": "bandpass", "Band-stop": "bandstop"}
_IIR_FTYPE = {"Butterworth": "butter", "Chebyshev I": "cheby1", "Chebyshev II": "cheby2",
              "Elliptic": "ellip", "Bessel": "bessel"}


class FilterError(ValueError):
    """Raised for a specification scipy cannot realise (bad band edges, order, ripple)."""


@dataclass
class FilterSpec:
    """A designed filter plus the parameters that produced it.

    FIR filters carry `b` (and a == [1]); IIR filters carry `sos`.  `describe()` is what the
    UI prints in the summary strip and what lands in the exported report.
    """
    kind: str                                  # "FIR" | "IIR"
    band: str
    fs: float
    cutoff: tuple
    order: int
    b: np.ndarray | None = None
    a: np.ndarray | None = None
    sos: np.ndarray | None = None
    family: str = ""
    window: str = ""
    method: str = ""
    ripple_db: float = 1.0
    atten_db: float = 40.0

    def describe(self) -> str:
        cut = " - ".join(f"{c:,.1f}" for c in self.cutoff) + " Hz"
        if self.kind == "FIR":
            detail = self.window if self.method == "Window method" else self.method
            return f"FIR {self.band} | {cut} | {self.order} taps | {detail}"
        return f"IIR {self.family} {self.band} | {cut} | order {self.order}"

    @property
    def taps(self) -> int:
        return int(self.b.size) if self.b is not None else 0


# ----------------------------------------------------------------- validation
def _norm_cutoff(cutoff, fs: float, band: str):
    """Normalise cutoffs to Nyquist units and sanity-check them against the band type."""
    nyq = fs / 2.0
    cut = tuple(float(c) for c in (cutoff if isinstance(cutoff, (list, tuple, np.ndarray))
                                   else (cutoff,)))
    need = 2 if band in ("Band-pass", "Band-stop") else 1
    if len(cut) != need:
        raise FilterError(f"{band} needs {need} cutoff frequency(ies), got {len(cut)}.")
    if need == 2 and cut[0] >= cut[1]:
        raise FilterError("The low cutoff must be below the high cutoff.")
    for c in cut:
        if not (0.0 < c < nyq):
            raise FilterError(f"Cutoff {c:,.1f} Hz must be between 0 and Nyquist ({nyq:,.1f} Hz).")
    return cut, tuple(c / nyq for c in cut)


# ----------------------------------------------------------------- design
def design_fir(band: str, cutoff, fs: float, numtaps: int = 101,
               window: str = "hamming", method: str = "Window method",
               kaiser_beta: float = 8.6, transition_hz: float | None = None) -> FilterSpec:
    """Design an FIR filter.  `numtaps` is forced odd for high-pass/band-stop, which need a
    Type-I (symmetric, odd-length) response to be able to pass Nyquist."""
    cut, wn = _norm_cutoff(cutoff, fs, band)
    numtaps = max(int(numtaps), 3)
    if band in ("High-pass", "Band-stop") and numtaps % 2 == 0:
        numtaps += 1

    try:
        if method == "Parks-McClellan (remez)":
            b = _design_remez(band, cut, fs, numtaps, transition_hz)
        elif method == "Least-squares (firls)":
            b = _design_firls(band, cut, fs, numtaps if numtaps % 2 else numtaps + 1, transition_hz)
        else:
            win = ("kaiser", kaiser_beta) if window == "kaiser" else window
            pass_zero = band in ("Low-pass", "Band-stop")
            b = sig.firwin(numtaps, wn, window=win, pass_zero=pass_zero)
    except FilterError:
        raise
    except Exception as exc:
        raise FilterError(f"FIR design failed: {exc}") from exc

    return FilterSpec(kind="FIR", band=band, fs=fs, cutoff=cut, order=int(np.size(b)),
                      b=np.asarray(b, dtype=np.float64), a=np.array([1.0]),
                      window=window, method=method)


def _edges(band: str, cut, fs: float, trans: float):
    """Band edges for the optimal-design routines, as (frequency list, desired gain list).

    Each transition is clipped to 90% of the room actually available on its side, so a wide
    requested width near DC or Nyquist shrinks instead of producing a degenerate band that
    remez/firls would reject.
    """
    nyq = fs / 2.0
    t = float(np.clip(trans, fs * 0.002, nyq * 0.45))
    below = lambda f: f - min(t, f * 0.9)              # noqa: E731  - room down to DC
    above = lambda f: f + min(t, (nyq - f) * 0.9)      # noqa: E731  - room up to Nyquist

    if band == "Low-pass":
        return [0, cut[0], above(cut[0]), nyq], [1, 0]
    if band == "High-pass":
        return [0, below(cut[0]), cut[0], nyq], [0, 1]
    if band == "Band-pass":
        return [0, below(cut[0]), cut[0], cut[1], above(cut[1]), nyq], [0, 1, 0]
    return [0, below(cut[0]), cut[0], cut[1], above(cut[1]), nyq], [1, 0, 1]


def _design_remez(band, cut, fs, numtaps, transition_hz):
    trans = transition_hz if transition_hz else fs / 2.0 * 0.08
    edges, gains = _edges(band, cut, fs, trans)
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise FilterError("Transition width is too wide for these cutoffs - reduce it.")
    return sig.remez(numtaps, edges, gains, fs=fs)


def _design_firls(band, cut, fs, numtaps, transition_hz):
    trans = transition_hz if transition_hz else fs / 2.0 * 0.08
    edges, gains = _edges(band, cut, fs, trans)
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise FilterError("Transition width is too wide for these cutoffs - reduce it.")
    desired = [g for g in gains for _ in (0, 1)]
    return sig.firls(numtaps, edges, desired, fs=fs)


def design_iir(family: str, band: str, cutoff, fs: float, order: int = 4,
               ripple_db: float = 1.0, atten_db: float = 40.0) -> FilterSpec:
    """Design an IIR filter as second-order sections.

    SOS rather than transfer-function coefficients: above about order 8 the direct-form
    polynomial loses accuracy badly, and cascaded biquads stay well-conditioned.
    """
    cut, wn = _norm_cutoff(cutoff, fs, band)
    order = int(np.clip(order, 1, 20))
    ftype = _IIR_FTYPE.get(family)
    if ftype is None:
        raise FilterError(f"Unknown IIR family: {family}")
    kw = {}
    if ftype in ("cheby1", "ellip"):
        kw["rp"] = max(float(ripple_db), 0.01)
    if ftype in ("cheby2", "ellip"):
        kw["rs"] = max(float(atten_db), 1.0)
    try:
        sos = sig.iirfilter(order, wn, btype=_SCIPY_BAND[band], ftype=ftype, output="sos", **kw)
    except Exception as exc:
        raise FilterError(f"IIR design failed: {exc}") from exc
    return FilterSpec(kind="IIR", band=band, fs=fs, cutoff=cut, order=order,
                      sos=np.asarray(sos, dtype=np.float64), family=family,
                      ripple_db=ripple_db, atten_db=atten_db)


def design_notch(freq: float, fs: float, q: float = 30.0) -> FilterSpec:
    """A single second-order notch - the tool for mains hum and other fixed interference."""
    if not (0.0 < freq < fs / 2.0):
        raise FilterError(f"Notch frequency must be between 0 and {fs / 2:,.1f} Hz.")
    b, a = sig.iirnotch(freq, max(q, 0.1), fs)
    return FilterSpec(kind="IIR", band="Band-stop", fs=fs, cutoff=(freq,), order=2,
                      sos=np.asarray(sig.tf2sos(b, a), dtype=np.float64), family="Notch")


# ----------------------------------------------------------------- application
def apply_filter(spec: FilterSpec, x: np.ndarray, zero_phase: bool = True) -> np.ndarray:
    """Run `x` through the filter.

    `zero_phase` uses filtfilt / sosfiltfilt: the signal is filtered forwards then backwards,
    which cancels the phase response entirely (no group-delay smearing) and squares the
    magnitude response, so the effective order is doubled.  Turn it off to see real,
    causal filter behaviour including the phase distortion an IIR filter introduces.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    try:
        if spec.sos is not None:
            if zero_phase and x.size > 3 * spec.sos.shape[0] * 2:
                return sig.sosfiltfilt(spec.sos, x)
            return sig.sosfilt(spec.sos, x)
        if zero_phase and x.size > 3 * spec.b.size:
            return sig.filtfilt(spec.b, spec.a, x)
        return sig.lfilter(spec.b, spec.a, x)
    except Exception as exc:
        raise FilterError(f"Could not apply the filter: {exc}") from exc


# ----------------------------------------------------------------- characterisation
def frequency_response(spec: FilterSpec, n: int = 4096):
    """(frequencies in Hz, magnitude in dB, phase in degrees) for the magnitude/phase plots."""
    if spec.sos is not None:
        w, h = sig.sosfreqz(spec.sos, worN=n, fs=spec.fs)
    else:
        w, h = sig.freqz(spec.b, spec.a, worN=n, fs=spec.fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
    phase = np.degrees(np.unwrap(np.angle(h)))
    return w, mag, phase


def group_delay(spec: FilterSpec, n: int = 2048):
    """Group delay in samples: -d(phase)/d(omega).  Flat for a linear-phase FIR, which is
    exactly the property that makes FIR the right choice when waveform shape matters."""
    import warnings
    b, a = _as_tf(spec)
    try:
        # A band-stop or notch has a near-zero denominator at the notch, where group delay is
        # genuinely undefined; scipy warns about the singularity. The warning is expected for
        # these designs and would otherwise be printed on every redraw.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w, gd = sig.group_delay((b, a), w=n, fs=spec.fs)
        return w, np.nan_to_num(gd, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.linspace(0, spec.fs / 2, n), np.zeros(n)


def pole_zero(spec: FilterSpec):
    """(zeros, poles) in the z-plane.  Stability is readable at a glance: every pole must sit
    strictly inside the unit circle, which an FIR filter guarantees by having none."""
    if spec.sos is not None:
        z, p, _k = sig.sos2zpk(spec.sos)
    else:
        z, p, _k = sig.tf2zpk(spec.b, spec.a)
    return np.asarray(z), np.asarray(p)


def impulse_response(spec: FilterSpec, n: int = 256):
    imp = np.zeros(int(n))
    imp[0] = 1.0
    return apply_filter(spec, imp, zero_phase=False)


def step_response(spec: FilterSpec, n: int = 256):
    return apply_filter(spec, np.ones(int(n)), zero_phase=False)


def is_stable(spec: FilterSpec) -> bool:
    if spec.sos is None:
        return True                                   # FIR: all poles at the origin
    _z, p = pole_zero(spec)
    return bool(np.all(np.abs(p) < 1.0))


def _as_tf(spec: FilterSpec):
    if spec.sos is not None:
        b, a = sig.sos2tf(spec.sos)
        return np.asarray(b), np.asarray(a)
    return spec.b, spec.a


def estimate_attenuation(spec: FilterSpec) -> dict:
    """Realised pass-band ripple and stop-band attenuation, measured from the response.

    Useful in defence: the designed order and the achieved attenuation rarely match the
    textbook estimate, and this shows the actual figures for the filter on screen.
    """
    w, mag, _ph = frequency_response(spec)
    nyq = spec.fs / 2.0
    cut = spec.cutoff
    if len(cut) == 1 and spec.band in ("Band-pass", "Band-stop"):
        # A notch is a band-stop described by a single centre frequency; give it a nominal
        # width so the pass/stop masks below have two edges to work with.
        centre = cut[0]
        width = max(centre * 0.1, nyq * 0.01)
        cut = (max(centre - width, nyq * 1e-4), min(centre + width, nyq * 0.999))
    if spec.band == "Low-pass":
        pb, sb = w <= cut[0] * 0.8, w >= min(cut[0] * 1.5, nyq * 0.99)
    elif spec.band == "High-pass":
        pb, sb = w >= min(cut[0] * 1.25, nyq * 0.99), w <= cut[0] * 0.6
    elif spec.band == "Band-pass":
        pb = (w >= cut[0] * 1.1) & (w <= cut[1] * 0.9)
        sb = (w <= cut[0] * 0.6) | (w >= min(cut[1] * 1.6, nyq))
    else:
        pb = (w <= cut[0] * 0.6) | (w >= min(cut[1] * 1.6, nyq))
        sb = (w >= cut[0] * 1.1) & (w <= cut[1] * 0.9)
    out = {"Pass-band ripple (dB)": float("nan"), "Stop-band attenuation (dB)": float("nan")}
    if np.any(pb):
        out["Pass-band ripple (dB)"] = float(np.ptp(mag[pb]))
    if np.any(sb):
        out["Stop-band attenuation (dB)"] = float(-np.max(mag[sb]))
    return out


# ----------------------------------------------------------------- simple time-domain filters
def moving_average(x: np.ndarray, size: int = 3) -> np.ndarray:
    """Boxcar smoother - the simplest FIR low-pass, shown for comparison with a designed one."""
    size = max(int(size), 1)
    kernel = np.ones(size) / size
    return np.convolve(x, kernel, mode="same")


def median_smooth(x: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter - non-linear, so it removes impulses without blurring edges the way a
    linear low-pass would."""
    size = max(int(size), 1)
    if size % 2 == 0:
        size += 1
    return sig.medfilt(np.asarray(x, dtype=np.float64), kernel_size=size)


def savitzky_golay(x: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    """Local polynomial least-squares smoothing: suppresses noise while preserving peak height
    and width far better than a moving average of the same length."""
    window = max(int(window), 3)
    if window % 2 == 0:
        window += 1
    window = min(window, x.size if x.size % 2 else x.size - 1)
    poly = int(np.clip(poly, 1, window - 1))
    if window < 3:
        return x.copy()
    return sig.savgol_filter(x, window, poly)


def convolve(x: np.ndarray, h: np.ndarray, mode: str = "full", method: str = "auto"):
    """Linear convolution - the defining operation of an LTI system, y[n] = x[n] * h[n]."""
    return sig.convolve(x, h, mode=mode, method=method)


def correlate(x: np.ndarray, y: np.ndarray, mode: str = "full"):
    """Cross-correlation - matched filtering and delay estimation."""
    return sig.correlate(x, y, mode=mode)
