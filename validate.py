"""
SPECTRA validation suite - evidence that the DSP actually works.

Run this before a demonstration, or during the defence when asked "how do you know it is
correct?".  It checks the algorithms against ground truth rather than merely checking that
they run:

  * generators produce the frequency they claim,
  * added noise lands at the SNR it was asked for,
  * filters attenuate what they are supposed to attenuate,
  * every denoiser improves the SNR of the noise it is designed for,
  * image restoration recovers a known blur,
  * metrics agree with their definitions on cases with known answers.

    python validate.py            run everything
    python validate.py --quick    skip the slower image and benchmark sections
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

from dsp import analysis as A, denoise as D, filters as F, imaging as I, metrics as M, signals as S

PASS, FAIL = 0, 0
FS = 8000.0

# The Windows console defaults to cp1252, which cannot encode the arrows and dashes used
# below; without this the suite dies on its own output rather than on a real failure.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [ok]   {label}" + (f"  — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {label}" + (f"  — {detail}" if detail else ""))


def section(name: str) -> None:
    print(f"\n{name}\n" + "-" * len(name))


# ----------------------------------------------------------------- 1. generation
def test_generation():
    section("1. Signal generation — does a generator produce the frequency it claims?")
    for freq in (100.0, 440.0, 1000.0, 2500.0):
        x = S.generate("Sine", 1.0, FS, freq)
        measured = A.dominant_frequency(x, FS)
        check(f"sine at {freq:,.0f} Hz", abs(measured - freq) < 2.0,
              f"measured {measured:,.1f} Hz")

    x = S.generate("Square", 1.0, FS, 200.0)
    f, mag = A.spectrum(x, FS, db=False)
    # A square wave has only odd harmonics; the 2nd should be far below the 3rd.
    def amp(target):
        return float(mag[int(round(target * x.size / FS))])
    check("square wave has odd harmonics only", amp(600.0) > 8 * amp(400.0),
          f"3rd/2nd harmonic ratio {amp(600.0) / max(amp(400.0), 1e-12):,.0f}x")

    x = S.generate("Chirp (linear)", 2.0, FS, 200.0, freq2=2000.0)
    t, f0 = A.instantaneous_frequency(x, FS)
    mid = f0[len(f0) // 2]
    check("linear chirp sweeps through its midpoint", abs(mid - 1100.0) < 150.0,
          f"instantaneous {mid:,.0f} Hz at the halfway point (expected ~1,100)")

    for kind in S.WAVEFORMS:
        y = S.generate(kind, 0.3, FS, 300.0)
        if not (np.isfinite(y).all() and y.size == int(0.3 * FS)):
            check(f"generator {kind}", False)
            break
    else:
        check(f"all {len(S.WAVEFORMS)} generators produce finite, correctly sized output", True)


# ----------------------------------------------------------------- 2. noise
def test_noise():
    section("2. Noise injection — does the requested SNR actually land?")
    clean = S.generate("Speech-like formants", 2.0, FS, 120.0)
    for target in (-5.0, 0.0, 5.0, 15.0, 25.0):
        noisy = S.add_noise(clean, FS, "White Gaussian (AWGN)", target, seed=1)
        measured = M.snr_db(clean, noisy)
        check(f"AWGN at {target:+.0f} dB", abs(measured - target) < 0.1,
              f"measured {measured:+.3f} dB")

    for kind in ("Pink (1/f)", "Brown (1/f^2)", "Uniform", "Mains hum (60 Hz)",
                 "Narrowband tone"):
        measured = M.snr_db(clean, S.add_noise(clean, FS, kind, 10.0, seed=2))
        check(f"{kind} at 10 dB", abs(measured - 10.0) < 0.1, f"measured {measured:.3f} dB")

    hum = S.add_noise(clean, FS, "Mains hum (60 Hz)", 0.0, seed=3)
    f, mag = A.spectrum(hum - clean, FS, db=False)
    peak = float(f[int(np.argmax(mag))])
    check("mains hum peaks at 60 Hz", abs(peak - 60.0) < 2.0, f"peak at {peak:.1f} Hz")


# ----------------------------------------------------------------- 3. sampling
def test_sampling():
    section("3. Sampling and quantisation — the theorems, measured")
    # Quantisation SNR should approach the 6.02B + 1.76 dB rule for a full-scale sine.
    x = S.generate("Sine", 1.0, FS, 300.0, amplitude=1.0)
    for bits in (4, 8, 12, 16):
        q = S.quantize(x, bits)
        snr = M.snr_db(x, q)
        theory = 6.02 * bits + 1.76
        check(f"{bits}-bit quantisation SNR near 6.02B+1.76",
              abs(snr - theory) < 4.0, f"measured {snr:.1f} dB, theory {theory:.1f} dB")

    # Aliasing: a 3 kHz tone decimated by 4 (new Nyquist 1 kHz) must fold to 1 kHz.
    tone = S.generate("Sine", 1.0, FS, 3000.0)
    aliased, new_fs = S.downsample_demo(tone, FS, 4, anti_alias=False)
    folded = A.dominant_frequency(aliased, new_fs)
    check("decimating without an anti-alias filter folds 3 kHz to 1 kHz",
          abs(folded - 1000.0) < 25.0, f"appears at {folded:,.0f} Hz (new Nyquist "
                                       f"{new_fs / 2:,.0f} Hz)")
    clean_dec, _fs2 = S.downsample_demo(tone, FS, 4, anti_alias=True)
    energy = float(np.mean(clean_dec ** 2)) / max(float(np.mean(tone ** 2)), 1e-20)
    check("with the anti-alias filter the out-of-band tone is removed instead", energy < 0.02,
          f"residual energy {energy:.4f} of the original")


# ----------------------------------------------------------------- 4. filters
def test_filters():
    section("4. Filter design — does a filter attenuate what it should?")
    noise = np.random.default_rng(0).standard_normal(int(FS * 2))

    def band_energy(sig, lo, hi):
        f, mag = A.spectrum(sig, FS, db=False)
        band = (f >= lo) & (f <= hi)
        return float(np.sum(mag[band] ** 2))

    for kind, spec in (("FIR", F.design_fir("Low-pass", 1000.0, FS, 201)),
                       ("IIR", F.design_iir("Butterworth", "Low-pass", 1000.0, FS, 6))):
        y = F.apply_filter(spec, noise)
        stop_before = band_energy(noise, 2000.0, 3800.0)
        stop_after = band_energy(y, 2000.0, 3800.0)
        pass_ratio = band_energy(y, 100.0, 800.0) / max(band_energy(noise, 100.0, 800.0), 1e-20)
        check(f"{kind} low-pass rejects the stop band",
              stop_after < stop_before * 1e-3,
              f"stop-band energy cut to {stop_after / max(stop_before, 1e-20):.2e}")
        check(f"{kind} low-pass preserves the pass band", 0.7 < pass_ratio < 1.4,
              f"pass-band energy ratio {pass_ratio:.3f}")

    for band, cut in (("Low-pass", 1200.0), ("High-pass", 800.0),
                      ("Band-pass", (400.0, 2000.0)), ("Band-stop", (900.0, 1300.0))):
        for fam in F.IIR_FAMILIES:
            spec = F.design_iir(fam, band, cut, FS, 5)
            if not F.is_stable(spec):
                check(f"{fam} {band} stability", False)
                break
        else:
            continue
        break
    else:
        check(f"all {len(F.IIR_FAMILIES)} IIR families stable across all 4 band types", True)

    fir = F.design_fir("Band-pass", (500.0, 2000.0), FS, 101)
    w, gd = F.group_delay(fir)
    inband = (w > 600) & (w < 1900)
    check("linear-phase FIR has flat group delay",
          float(np.ptp(gd[inband])) < 0.5,
          f"varies by {float(np.ptp(gd[inband])):.3f} samples; expected (N-1)/2 = "
          f"{(fir.taps - 1) / 2:.0f}")
    check("FIR has no poles outside the origin (cannot be unstable)",
          F.pole_zero(fir)[1].size == 0 or np.all(np.abs(F.pole_zero(fir)[1]) < 1e-9))

    # A filter's coefficients are tied to the sample rate it was designed for.  Reusing an
    # 8 kHz design on a 16 kHz signal silently doubles every cutoff, so the UI re-designs
    # against the signal's own rate on every apply; this guards that.
    for rate in (8000.0, 16000.0, 44100.0):
        spec = F.design_iir("Butterworth", "Low-pass", 2000.0, rate, 4)
        w, mag, _ph = F.frequency_response(spec, 8192)
        below = np.where(mag < -3.0)[0]
        edge = float(w[below[0]]) if below.size else float("nan")
        check(f"a 2 kHz low-pass designed at {rate:,.0f} Hz really cuts at 2 kHz",
              abs(edge - 2000.0) < 60.0,
              f"-3 dB at {edge:,.0f} Hz (reusing an 8 kHz design here would give "
              f"{2000.0 * rate / 8000.0:,.0f} Hz)")

    # A notch must kill its target and leave a nearby tone alone.
    t = np.arange(int(FS)) / FS
    mixed = np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 440 * t)
    out = F.apply_filter(F.design_notch(60.0, FS, 30.0), mixed)
    check("60 Hz notch removes the hum but keeps 440 Hz",
          band_energy(out, 55, 65) < band_energy(mixed, 55, 65) * 0.01
          and band_energy(out, 420, 460) > band_energy(mixed, 420, 460) * 0.8,
          "hum cut by >20 dB, tone retained")


# ----------------------------------------------------------------- 5. enhancement
def test_enhancement():
    section("5. Noise reduction — does each method improve its own kind of noise?")
    clean = S.generate("Speech-like formants", 2.0, FS, 120.0)

    cases = [("White Gaussian (AWGN)", 5.0,
              ["Spectral subtraction", "Wiener filter (frequency domain)",
               "Wavelet thresholding", "Spectral gating", "Wiener filter (adaptive local)"]),
             ("Mains hum (60 Hz)", 3.0, ["Notch - mains hum removal"]),
             ("Impulse / click", 8.0, ["Median filter (de-click)"]),
             ("Pink (1/f)", 5.0, ["Spectral subtraction", "Wiener filter (frequency domain)"])]

    for noise_kind, snr, methods in cases:
        noisy = S.add_noise(clean, FS, noise_kind, snr, density=0.01, seed=5)
        base = M.snr_db(clean, noisy)
        for method in methods:
            y, _info = D.enhance(method, noisy, FS)
            n = min(clean.size, y.size)
            gain = M.snr_db(clean[:n], y[:n]) - base
            check(f"{method} on {noise_kind}", gain > 0.5,
                  f"{base:.2f} → {base + gain:.2f} dB ({gain:+.2f} dB)")

    # Every method must at least run and return a finite, correctly-sized result.
    noisy = S.add_noise(clean, FS, "White Gaussian (AWGN)", 5.0, seed=6)
    broken = [m for m in D.METHODS
              if not _runs_clean(m, noisy)]
    check(f"all {len(D.METHODS)} enhancement methods run and return finite output",
          not broken, f"failed: {broken}" if broken else "")


def _runs_clean(method, x):
    try:
        y, _i = D.enhance(method, x, FS)
        return np.isfinite(y).all() and abs(y.size - x.size) <= 2
    except Exception:
        return False


# ----------------------------------------------------------------- 6. analysis
def test_analysis():
    section("6. Spectral and speech analysis")
    t = np.arange(int(FS)) / FS
    two_tone = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t)
    f, mag = A.spectrum(two_tone, FS, db=False)

    def amp(target):
        k = int(round(target * two_tone.size / FS))
        return float(np.max(mag[k - 2:k + 3]))

    check("DFT amplitude is window-compensated", abs(amp(440.0) - 1.0) < 0.05,
          f"a unit-amplitude sine reads {amp(440.0):.3f}")
    check("relative amplitudes preserved", abs(amp(1200.0) / amp(440.0) - 0.5) < 0.05,
          f"ratio {amp(1200.0) / amp(440.0):.3f}, expected 0.5")

    check("spectral flatness: noise ≈ 1, tone ≈ 0",
          A.spectral_flatness(np.random.default_rng(1).standard_normal(4096), FS) > 0.4
          and A.spectral_flatness(np.sin(2 * np.pi * 440 * t), FS) < 0.05,
          f"noise {A.spectral_flatness(np.random.default_rng(1).standard_normal(4096), FS):.2f}"
          f" vs tone {A.spectral_flatness(np.sin(2 * np.pi * 440 * t), FS):.4f}")

    for f0 in (100.0, 150.0, 220.0):
        speech = S.generate("Speech-like formants", 1.5, FS, f0)
        _t, track = A.pitch_track(speech, FS)
        median = float(np.nanmedian(track))
        check(f"pitch tracker finds F0 = {f0:.0f} Hz", abs(median - f0) < f0 * 0.06,
              f"median {median:.1f} Hz")

    speech = S.generate("Speech-like formants", 1.0, FS, 120.0)
    _tm, coeffs = A.mfcc(speech, FS, 13, 26)
    check("MFCC matrix has the requested shape", coeffs.shape[0] == 13 and coeffs.shape[1] > 10,
          f"{coeffs.shape[0]} × {coeffs.shape[1]}")
    check("mel filterbank rows are normalised triangles",
          np.allclose(A.mel_filterbank(26, 512, FS).max(axis=1), 1.0, atol=0.01))

    t2 = np.arange(int(FS * 0.5)) / FS
    check("spectral centroid rises with brightness",
          A.spectral_centroid(np.sin(2 * np.pi * 2000 * t2), FS) >
          A.spectral_centroid(np.sin(2 * np.pi * 300 * t2), FS))


# ----------------------------------------------------------------- 7. metrics
def test_metrics():
    section("7. Metrics — do they agree with their definitions?")
    x = S.generate("Sine", 1.0, FS, 440.0)
    check("SNR of a signal against itself is infinite", np.isinf(M.snr_db(x, x)))
    check("MSE of a signal against itself is zero", M.mse(x, x) < 1e-20)
    check("correlation of a signal against itself is 1", abs(M.correlation(x, x) - 1.0) < 1e-9)
    check("correlation against its negation is -1", abs(M.correlation(x, -x) + 1.0) < 1e-9)

    # Halving the error power must raise SNR by exactly 3.01 dB.
    noisy1 = S.add_noise(x, FS, "White Gaussian (AWGN)", 10.0, seed=1)
    err = noisy1 - x
    noisy2 = x + err / np.sqrt(2.0)
    delta = M.snr_db(x, noisy2) - M.snr_db(x, noisy1)
    check("halving the error power raises SNR by 3.01 dB", abs(delta - 3.0103) < 0.01,
          f"measured {delta:.4f} dB")

    pure = S.generate("Sine", 1.0, FS, 500.0)
    check("THD of a pure sine is near zero", M.thd_percent(pure, FS) < 1.0,
          f"{M.thd_percent(pure, FS):.3f}%")
    square = S.generate("Square", 1.0, FS, 500.0)
    check("THD of a square wave is large", M.thd_percent(square, FS) > 20.0,
          f"{M.thd_percent(square, FS):.1f}% (theory ~48%)")

    img = np.random.default_rng(0).random((96, 96)) * 255
    check("SSIM of an image against itself is 1", abs(M.ssim(img, img, 255.0) - 1.0) < 1e-6)
    check("PSNR of an image against itself is infinite", np.isinf(M.psnr_db(img, img, 255.0)))
    blurred = I.spatial_filter(img, "Gaussian", 7, 2.0)
    check("SSIM drops when an image is degraded", M.ssim(img, blurred, 255.0) < 0.9,
          f"SSIM {M.ssim(img, blurred, 255.0):.3f} after blurring")


# ----------------------------------------------------------------- 8. imaging
def test_imaging():
    section("8. Image processing and restoration")
    rng = np.random.default_rng(0)
    h = w = 192
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3))
    img[..., 0] = 120 + 100 * np.sin(2 * np.pi * xx / 40)
    img[..., 1] = np.where((xx > 60) & (xx < 140) & (yy > 50) & (yy < 150), 200, 60)
    img[..., 2] = np.clip(yy * 255 / h, 0, 255)
    grey = I.to_gray(img)

    noisy = I.add_noise(img, "Gaussian", 25.0, seed=1)
    base = M.psnr_db(grey, I.to_gray(noisy), 255.0)
    for kind in ("Mean (box)", "Gaussian", "Median", "Bilateral"):
        out = I.spatial_filter(noisy, kind, 5)
        gain = M.psnr_db(grey, I.to_gray(out), 255.0) - base
        check(f"{kind} denoises Gaussian noise", gain > 1.0, f"PSNR {gain:+.2f} dB")

    sp = I.add_noise(img, "Salt & pepper", density=0.06, seed=2)
    sp_base = M.psnr_db(grey, I.to_gray(sp), 255.0)
    med = M.psnr_db(grey, I.to_gray(I.spatial_filter(sp, "Median", 3)), 255.0) - sp_base
    mean = M.psnr_db(grey, I.to_gray(I.spatial_filter(sp, "Mean (box)", 3)), 255.0) - sp_base
    check("median beats mean on salt & pepper (the non-linear advantage)", med > mean + 3.0,
          f"median {med:+.2f} dB vs mean {mean:+.2f} dB")

    for blur_kind in I.BLUR_KINDS:
        psf = I.make_psf(blur_kind, 15, 30.0, 3.0)
        degraded = I.add_noise(I.blur(img, psf), "Gaussian", 2.0, seed=3)
        before = M.psnr_db(grey, I.to_gray(degraded), 255.0)
        for method in ("Wiener deconvolution", "Richardson-Lucy", "Constrained least squares"):
            # Each method's regularisation parameter is a different quantity, so each gets
            # its own default rather than a single shared value.
            out = I.restore(degraded, method, psf,
                            k=I.DEFAULT_REGULARISATION.get(method, 0.01), iterations=15)
            gain = M.psnr_db(grey, I.to_gray(out), 255.0) - before
            check(f"{method} recovers {blur_kind}", gain > 0.5, f"PSNR {gain:+.2f} dB")

    periodic = I.add_noise(img, "Periodic (sinusoidal)", 40.0, freq=0.12)
    peaks = I.detect_periodic_peaks(periodic, 2)
    check("periodic interference is detected as off-centre spectral peaks", len(peaks) == 2,
          f"peaks at {peaks}")
    cleaned, _mask = I.frequency_filter(periodic, "Band-stop (notch)", 40.0, 2, 12.0, peaks)
    gain = (M.psnr_db(grey, I.to_gray(cleaned), 255.0)
            - M.psnr_db(grey, I.to_gray(periodic), 255.0))
    check("notching those peaks removes the interference", gain > 1.0, f"PSNR {gain:+.2f} dB")

    eq = I.histogram_op(img, "Histogram equalisation")
    check("histogram equalisation widens the intensity spread",
          float(np.std(I.to_gray(eq))) > float(np.std(grey)),
          f"std {np.std(grey):.1f} → {np.std(I.to_gray(eq)):.1f}")

    for op in I.HISTOGRAM_OPS + I.MORPH_OPS:
        fn = I.histogram_op if op in I.HISTOGRAM_OPS else I.morphology
        if not np.isfinite(fn(img, op)).all():
            check(f"operation {op}", False)
            break
    else:
        check("all histogram and morphology operations produce finite output", True)


# ----------------------------------------------------------------- 9. video
def test_video():
    section("9. Video processing")
    import os
    import tempfile

    import cv2
    from dsp import video as V

    path = os.path.join(tempfile.gettempdir(), "spectra_validate_clip.mp4")
    w = h = 128
    fps, frames = 20, 30
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for i in range(frames):
            frame = np.full((h, w, 3), 60.0)
            cx = int(16 + (w - 48) * i / frames)
            cv2.rectangle(frame, (cx, 50), (cx + 24, 80), (220, 180, 60), -1)
            frame += rng.normal(0, 14, frame.shape)
            writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        writer.release()

    src = V.VideoSource(path)
    info = src.summary()
    src.close()
    check("a clip opens and reports its properties",
          info["Frames"] >= frames - 2 and abs(info["Frame rate (fps)"] - fps) < 1,
          f"{info['Frames']} frames @ {info['Frame rate (fps)']} fps")

    failed = []
    for op in V.OPERATIONS:
        proc = V.FrameProcessor(op)
        try:
            out = None
            for _i, frame in V.VideoSource(path).frames(limit=5):
                out = proc.process(frame)
            if out is None or not np.isfinite(out).all():
                failed.append(op)
        except Exception:
            failed.append(op)
    check(f"all {len(V.OPERATIONS)} frame operations run", not failed,
          f"failed: {failed}" if failed else "")

    # Temporal averaging over a static background must reduce noise frame to frame.
    proc = V.FrameProcessor("Temporal averaging", {"window": 8})
    raw_var, smooth_var = [], []
    for _i, frame in V.VideoSource(path).frames(limit=12):
        out = proc.process(frame)
        raw_var.append(float(np.var(I.to_gray(frame)[0:20, 0:20])))
        smooth_var.append(float(np.var(I.to_gray(out)[0:20, 0:20])))
    check("temporal averaging reduces noise in a static region",
          np.mean(smooth_var[4:]) < np.mean(raw_var[4:]) * 0.6,
          f"background variance {np.mean(raw_var[4:]):.1f} → {np.mean(smooth_var[4:]):.1f}")

    times, energy = V.motion_energy(path, 25)
    check("motion energy is produced for a moving target", energy.size > 5 and energy.mean() > 0,
          f"{energy.size} points, mean {energy.mean():.2f}")

    out_path = os.path.join(tempfile.gettempdir(), "spectra_validate_out.mp4")
    stats = V.process_video(path, out_path, "Gaussian denoise", {"size": 5}, max_frames=15)
    check("a processed clip is written back out", stats["Frames written"] >= 14,
          f"{stats['Frames written']} frames to {os.path.basename(out_path)}")
    for p in (path, out_path):
        try:
            os.remove(p)
        except OSError:
            pass


# ----------------------------------------------------------------- 10. presets
def test_presets():
    section("10. Simple presets — the one-dial layer used by the interface")
    from dsp import presets as P

    clean = S.generate("Speech-like formants", 2.0, FS, 120.0)

    # --- does Auto work out what is actually wrong with the signal?
    for kind, snr, expected in (("White Gaussian (AWGN)", 5.0, "Background noise"),
                                ("Mains hum (60 Hz)", 3.0, "Mains hum"),
                                ("Mains hum (50 Hz)", 12.0, "Mains hum"),
                                ("Impulse / click", 8.0, "Clicks & pops"),
                                ("Pink (1/f)", 5.0, "Background noise")):
        noisy = S.add_noise(clean, FS, kind, snr, density=0.01, seed=5)
        verdict = P.diagnose_audio(noisy, FS)["verdict"]
        check(f"Auto diagnoses {kind} as '{expected}'", verdict == expected,
              f"said '{verdict}'")

    # A clean voice must not be mistaken for hum: a 120 Hz pitch puts energy on every
    # harmonic of 60 Hz, which is exactly the trap the persistence test exists to avoid.
    check("a clean voice is not mistaken for mains hum",
          P.diagnose_audio(clean, FS)["hum"] < 0.2,
          f"hum score {P.diagnose_audio(clean, FS)['hum']:.2f}")

    # --- does Auto then improve the signal, at both ends of the dial?
    for kind, snr in (("White Gaussian (AWGN)", 5.0), ("Mains hum (60 Hz)", 3.0),
                      ("Impulse / click", 8.0)):
        noisy = S.add_noise(clean, FS, kind, snr, density=0.01, seed=5)
        base = M.snr_db(clean, noisy)
        for strength in (0.0, 0.5, 1.0):
            y, _desc = P.clean_audio(noisy, FS, P.AUTO, strength)
            n = min(clean.size, y.size)
            gain = M.snr_db(clean[:n], y[:n]) - base
            check(f"Auto cleanup of {kind} at {strength:.0%}", gain > 0.5,
                  f"{gain:+.2f} dB")

    # --- recordings: no noise clip, often a silent lead-in while the microphone starts
    noisy = S.add_noise(clean, FS, "White Gaussian (AWGN)", 5.0, seed=8)
    lead = np.zeros(int(0.4 * FS))
    gains = []
    for x in (noisy, np.concatenate([lead, noisy])):
        ref = clean if x.size == clean.size else np.concatenate([lead, clean])
        y, _d = P.clean_audio(x, FS, "Background noise", 1.0)
        gains.append(M.snr_db(ref, y) - M.snr_db(ref, x))
    check("Background noise cleanup works without a noise sample", gains[0] > 3.0,
          f"{gains[0]:+.2f} dB from the signal alone")
    check("a silent lead-in does not blind the noise estimate", abs(gains[1] - gains[0]) < 0.5,
          f"{gains[0]:+.2f} dB without, {gains[1]:+.2f} dB with 0.4 s of digital silence")

    # --- reference-free measures, for a recording with no clean copy
    n0, s0 = M.activity_levels(noisy, FS)
    y, _d = P.clean_audio(noisy, FS, "Background noise", 1.0)
    n1, s1 = M.activity_levels(y, FS)
    check("without a reference, the cleanup is seen to lower the background",
          n1 < n0 - 6.0, f"background {n0:.1f} → {n1:.1f} dB")
    check("…while the voice level survives", abs(s1 - s0) < 1.5, f"voice {s1 - s0:+.2f} dB")
    check("THD is reported for a tone but not for speech",
          M.is_tonal(S.generate("Sine", 1.0, FS, 440.0), FS) and not M.is_tonal(clean, FS))

    broken = []
    noisy = S.add_noise(clean, FS, "White Gaussian (AWGN)", 5.0, seed=6)
    for preset in P.CLEANUP_PRESETS:
        try:
            y, _d = P.clean_audio(noisy, FS, preset, 0.5)
            if not (np.isfinite(y).all() and abs(y.size - noisy.size) <= 2):
                broken.append(preset)
        except Exception:
            broken.append(preset)
    check(f"all {len(P.CLEANUP_PRESETS)} cleanup presets run", not broken,
          f"failed: {broken}" if broken else "")

    # --- image: does the sharpness measure survive noise?
    img = _chart()
    noisy_img = I.add_noise(img, "Gaussian", 25.0, seed=1)
    check("the sharpness measure is not fooled by noise",
          P.estimate_sharpness(noisy_img) < P.estimate_sharpness(img) * 1.5,
          f"clean {P.estimate_sharpness(img):.2f} vs noisy "
          f"{P.estimate_sharpness(noisy_img):.2f} — the plain Laplacian variance rises about "
          f"tenfold on this same pair")
    blurred, _psf = P.degrade_image(img, 0.6, 0.0)
    check("blurring lowers the sharpness measure",
          P.estimate_sharpness(blurred) < P.estimate_sharpness(img),
          f"{P.estimate_sharpness(img):.2f} -> {P.estimate_sharpness(blurred):.2f}")
    for sigma_true in (5.0, 18.0, 30.0):
        estimated = P.estimate_noise(I.add_noise(img, "Gaussian", sigma_true, seed=2))
        check(f"the noise estimate matches {sigma_true:.0f} grey levels of added noise",
              abs(estimated - sigma_true) < 1.5,
              f"estimated {estimated:.1f}")

    # --- image: blurry and noisy really does become clear again
    for blur, noise, label in ((0.5, 0.0, "blur only"), (0.0, 0.4, "noise only"),
                               (0.5, 0.3, "blur + noise"), (0.8, 0.15, "heavy blur")):
        damaged, psf = P.degrade_image(img, blur, noise, seed=3)
        before = I.compare(img, damaged)
        after = I.compare(img, P.auto_enhance(damaged, psf)[0])
        check(f"Auto enhance repairs {label} — PSNR",
              after["PSNR (dB)"] > before["PSNR (dB)"],
              f"{before['PSNR (dB)']:.2f} -> {after['PSNR (dB)']:.2f} dB "
              f"({after['PSNR (dB)'] - before['PSNR (dB)']:+.2f})")
        check(f"Auto enhance repairs {label} — SSIM", after["SSIM"] > before["SSIM"],
              f"{before['SSIM']:.3f} -> {after['SSIM']:.3f}")

    # With the kernel known and no noise, the deconvolution should be close to exact.
    for blur in (0.3, 0.6):
        damaged, psf = P.degrade_image(img, blur, 0.0)
        gain = (I.compare(img, P.auto_enhance(damaged, psf)[0])["PSNR (dB)"]
                - I.compare(img, damaged)["PSNR (dB)"])
        check(f"Auto deblurs a clean {blur:.0%} blur almost exactly (K follows the noise)",
              gain > 4.0, f"PSNR {gain:+.2f} dB")

    # Auto must leave a good image alone rather than processing it for the sake of it.
    _out, steps = P.auto_enhance(img, None)
    check("Auto leaves an undamaged image essentially alone", len(steps) <= 2,
          f"did: {steps}")

    # --- video: temporal averaging is the thing a video can do that a still cannot
    cleaner = P.VideoCleaner(denoise=0.8)
    check("a denoise setting engages temporal filtering", cleaner.window > 1,
          f"{cleaner.window}-frame settling window")
    check("denoise off leaves temporal filtering off", P.VideoCleaner(denoise=0.0).window == 1)

    # The motion-adaptive filter must clean the static scene AND leave the moving ball sharp;
    # a plain frame average does the first and fails the second.
    from app.pages.video import _test_frame
    rng = np.random.default_rng(9)
    clean_frames = [_test_frame(i) for i in range(30)]
    noisy_frames = [f + rng.normal(0, 14, f.shape) for f in clean_frames]
    adaptive = P.VideoCleaner(denoise=0.6)
    outs = [adaptive.process(f) for f in noisy_frames]
    averaged = [np.mean(noisy_frames[max(i - 2, 0):i + 1], axis=0) for i in range(30)]

    def region_psnr(frames, i, moving):
        g0, g1 = I.to_gray(clean_frames[i]), I.to_gray(frames[i])
        mask = np.abs(g0 - I.to_gray(clean_frames[(i + 15) % 30])) > 25
        mask = mask if moving else ~mask
        return 10 * np.log10(255.0 ** 2 / np.mean((g0[mask] - g1[mask]) ** 2))

    idx = range(12, 30)
    static_gain = np.mean([region_psnr(outs, i, False) - region_psnr(noisy_frames, i, False)
                           for i in idx])
    ball_adaptive = np.mean([region_psnr(outs, i, True) for i in idx])
    ball_average = np.mean([region_psnr(averaged, i, True) for i in idx])
    ball_noisy = np.mean([region_psnr(noisy_frames, i, True) for i in idx])
    check("temporal filtering cleans the static scene", static_gain > 5.0,
          f"PSNR {static_gain:+.2f} dB")
    check("the moving object is not smeared (motion-adaptive beats a plain average)",
          ball_adaptive > ball_noisy and ball_adaptive > ball_average + 5.0,
          f"ball PSNR: noisy {ball_noisy:.1f}, plain 3-frame average {ball_average:.1f}, "
          f"motion-adaptive {ball_adaptive:.1f} dB")
    frame = I.add_noise(img, "Gaussian", 20.0, seed=4)
    auto = P.auto_video_settings(frame)
    check("Auto video settings react to a noisy frame", auto.denoise > 0.3,
          f"denoise {auto.denoise:.2f}, {auto.describe()}")
    check("Auto video settings leave deconvolution off", auto.deblur == 0.0,
          "no blur kernel is known for a loaded clip")


# ----------------------------------------------------------------- 11. edges and patterns
def test_edges():
    section("11. Edge detection and periodic interference")
    from dsp import presets as P

    # A bright square on a dark ground: the true edge is its one-pixel outline, known exactly.
    h = w = 160
    square = np.full((h, w), 50.0)
    square[40:120, 40:120] = 200.0
    truth = np.zeros((h, w))
    truth[40:120, 40] = truth[40:120, 119] = truth[40, 40:120] = truth[119, 40:120] = 255.0

    for op in P.EDGE_OPERATORS:
        found = P.find_edges(square, op, 0.5)
        m = P.edge_match(truth, found, tolerance=2)
        check(f"{op} finds the square's outline and nothing else",
              m["Precision"] > 0.9 and m["Recall"] > 0.9,
              f"precision {m['Precision']:.1%}, recall {m['Recall']:.1%}")

    # The Sobel response across a vertical step must peak at the step, not beside it.
    step = np.zeros((40, 80))
    step[:, 40:] = 255.0
    profile = I.edge_detect(step, "Sobel")[20]
    peak = int(np.argmax(profile))
    check("Sobel gradient peaks on the step", peak in (39, 40), f"peak at column {peak}")

    # Raising the sensitivity must never find fewer edges.
    textured = I.add_noise(_chart(), "Gaussian", 8.0, seed=11)
    monotone = [op for op in P.EDGE_OPERATORS
                if P.edge_density(P.find_edges(textured, op, 0.9))
                < P.edge_density(P.find_edges(textured, op, 0.1))]
    check("higher sensitivity finds at least as many edges, for every detector", not monotone,
          f"failed: {monotone}" if monotone else "")

    # A derivative is a high-pass filter: on a noisy image, smoothing first is what keeps the
    # grain from being reported as edges.
    noisy = I.add_noise(square, "Gaussian", 20.0, seed=12)
    for op in ("Sobel", "Canny", "Prewitt"):
        raw = P.edge_match(truth, P.find_edges(noisy, op, 0.5, smooth=False), 2)["Precision"]
        smooth = P.edge_match(truth, P.find_edges(noisy, op, 0.5, smooth=True), 2)["Precision"]
        check(f"{op}: smoothing first removes false edges caused by noise", smooth > raw + 0.1,
              f"precision {raw:.1%} → {smooth:.1%}")

    check("edge agreement of a map with itself is perfect",
          abs(P.edge_match(truth, truth)["F1"] - 1.0) < 1e-12)

    # Restoration should give back edges, not just PSNR.
    from app.pages.image import _demo_scene
    scene = _demo_scene(240)
    damaged, psf = P.degrade_image(scene, 0.5, 0.3, seed=7)
    restored = P.auto_enhance(damaged, psf)[0]
    ref = P.find_edges(scene, "Canny", 0.5)
    f_before = P.edge_match(ref, P.find_edges(damaged, "Canny", 0.5))["F1"]
    f_after = P.edge_match(ref, P.find_edges(restored, "Canny", 0.5))["F1"]
    check("restoration recovers the original's edges", f_after > f_before + 0.1,
          f"Canny F1 {f_before:.2f} → {f_after:.2f}")

    frame = P.EdgeSettings("Sobel", 0.5, True, "Overlay").render(scene)
    check("edge rendering returns a displayable RGB frame",
          frame.shape == scene.shape and np.isfinite(frame).all())

    # --- periodic interference: two points in the spectrum, removable by a notch
    check("a clean scene's own texture is not mistaken for interference",
          not P.find_periodic_peaks(scene),
          f"found {len(P.find_periodic_peaks(scene))} peaks")
    striped, _ = P.degrade_image(scene, 0.0, 0.0, pattern=0.5)
    peaks = P.find_periodic_peaks(striped)
    check("added stripes appear as a conjugate pair of spectral peaks", len(peaks) == 2,
          f"peaks at {[(p[0], p[1]) for p in peaks]}")
    cleaned, _ = P.remove_pattern(striped)
    gain = M.psnr_db(I.to_gray(scene), I.to_gray(cleaned), 255.0) - \
        M.psnr_db(I.to_gray(scene), I.to_gray(striped), 255.0)
    check("notching the peaks removes the stripes", gain > 6.0, f"PSNR {gain:+.2f} dB")
    _out, steps = P.auto_enhance(striped)
    check("Auto enhance detects and notches the stripes", any("notch" in s for s in steps),
          f"did: {steps}")


def test_photo_like():
    section("12. Real-photo behaviour — texture, not just flat shapes")
    from dsp import presets as P

    # The other image checks use flat-shaded pictures, where a bilateral filter shines.  Real
    # photographs also have fine texture, and there the bilateral filter scored below doing
    # nothing.  This image has both: a smooth sky, a 1/f-textured ground and hard edges.
    img = _photo_like()
    check("the noise estimate is not fooled by texture on a clean image",
          P.estimate_noise(img) < 1.0, f"estimated {P.estimate_noise(img):.2f}")
    for sigma in (7.0, 12.0, 18.5):
        noisy = I.add_noise(img, "Gaussian", sigma, seed=1)
        est = P.estimate_noise(noisy)
        check(f"noise estimate on a textured image at sigma {sigma:g}",
              abs(est - sigma) / sigma < 0.1, f"estimated {est:.2f}")

        def mse(out):
            return float(np.mean((np.clip(out, 0, 255) - img) ** 2))

        chosen, how = P.auto_denoise(noisy, est)
        strength = float(np.clip((est - 2.5) / 45.0, 0.2, 0.55))
        best = min(mse(P.denoise_image(noisy, strength)), mse(P.wavelet_denoise_image(noisy, est)))
        check(f"SURE picks the better denoiser without a reference (sigma {sigma:g})",
              mse(chosen) <= best * 1.02, f"chose {how}: MSE {mse(chosen):.1f}, best {best:.1f}")

    noisy = I.add_noise(img, "Gaussian", 12.0, seed=1)
    before, after = I.compare(img, noisy), I.compare(img, P.auto_enhance(noisy)[0])
    check("Auto enhance improves a textured image — PSNR and SSIM",
          after["PSNR (dB)"] > before["PSNR (dB)"] and after["SSIM"] > before["SSIM"],
          f"PSNR {before['PSNR (dB)']:.2f} → {after['PSNR (dB)']:.2f}, "
          f"SSIM {before['SSIM']:.3f} → {after['SSIM']:.3f}")


def _photo_like(n: int = 256, seed: int = 5):
    """Smooth sky above, 1/f-textured ground below (the power spectrum of natural images),
    and two hard-edged blocks - the mix of flat and textured regions a real photo has."""
    rng = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]
    rho = np.maximum(np.hypot(fy, fx), 1.0 / n)
    tex = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) / rho))
    tex = 30.0 * (tex - tex.mean()) / tex.std()
    yy = np.mgrid[0:n, 0:n][0]
    img = np.zeros((n, n, 3))
    for c, (sky, ground) in enumerate(((150, 90), (180, 110), (220, 60))):
        img[..., c] = np.where(yy < n // 2, sky - 40.0 * yy / n, ground + tex)
    img[n // 4:n // 2 + 20, 30:90] = (200, 80, 70)
    img[n // 3:n // 2 + 10, 150:220] = (60, 70, 160)
    return np.clip(img, 0, 255)


def _chart():
    """The synthetic test image used by the image checks."""
    h = w = 192
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3))
    img[..., 0] = 120 + 100 * np.sin(2 * np.pi * xx / 40)
    img[..., 1] = np.where((xx > 60) & (xx < 140) & (yy > 50) & (yy < 150), 200, 60)
    img[..., 2] = np.clip(yy * 255 / h, 0, 255)
    return img


# ----------------------------------------------------------------- runner
def main() -> int:
    quick = "--quick" in sys.argv
    started = time.time()
    print("=" * 74)
    print("SPECTRA — DSP validation suite")
    print("=" * 74)

    tests = [test_generation, test_noise, test_sampling, test_filters, test_enhancement,
             test_analysis, test_metrics]
    tests += [test_presets, test_edges]
    if not quick:
        tests += [test_imaging, test_video, test_photo_like]

    for test in tests:
        try:
            test()
        except Exception as exc:
            global FAIL
            FAIL += 1
            print(f"  [FAIL] {test.__name__} raised {type(exc).__name__}: {exc}")

    total = PASS + FAIL
    print("\n" + "=" * 74)
    print(f"{PASS}/{total} checks passed in {time.time() - started:.1f}s"
          + (f"  —  {FAIL} FAILED" if FAIL else "  —  all good"))
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
