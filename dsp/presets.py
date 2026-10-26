"""
The simple layer: one decision instead of twenty knobs.

The modules underneath expose every parameter of every algorithm, which is right for a DSP
engine and wrong for the person using it.  This module sits on top and offers the three
things anyone actually wants - *what is wrong with this signal*, *fix it*, and *how much* -
mapping a single 0-1 strength onto whichever parameters the chosen method really needs.

Nothing here invents new signal processing.  Every function delegates to `denoise`,
`filters` or `imaging`; the value it adds is the diagnosis and the parameter mapping, so the
interface can be a dropdown and a slider instead of a control panel.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal as sig

from . import analysis as A, denoise as D, filters as F, imaging as I, metrics as M

# ----------------------------------------------------------------- audio
AUTO = "Auto — detect and fix"
CLEANUP_PRESETS = (AUTO, "Background noise", "Hiss / static", "Mains hum",
                   "Clicks & pops", "Smooth the waveform")

#: What each preset is for, shown under the chooser.
PRESET_HELP = {
    AUTO: "Looks at the signal, decides what is wrong with it, and applies the matching fix.",
    "Background noise": "Steady broadband noise behind speech — room tone, air conditioning, "
                        "traffic. Uses a Wiener filter driven by an estimate of the noise "
                        "spectrum.",
    "Hiss / static": "Fine high-frequency hiss. Uses wavelet shrinkage, which removes small "
                     "coefficients everywhere while leaving the large ones that carry the "
                     "signal.",
    "Mains hum": "A low buzz at 50 or 60 Hz and its harmonics, picked up from mains wiring. "
                 "Uses a cascade of narrow notch filters.",
    "Clicks & pops": "Short impulsive defects — vinyl crackle, dropouts, switching noise. "
                     "Uses a median filter that replaces only the samples that are outliers.",
    "Smooth the waveform": "General smoothing with Savitzky-Golay, which suppresses noise "
                           "while preserving the height and width of peaks.",
}


def _lerp(low: float, high: float, t: float) -> float:
    return low + (high - low) * float(np.clip(t, 0.0, 1.0))


def _hum_score(x: np.ndarray, fs: float):
    """Detect mains hum.  Returns (score 0-1, the mains frequency it matched).

    Two properties together identify hum, and neither is sufficient alone:

      * it is *narrowband* - a sharp peak at 50 or 60 Hz standing above its neighbours, and
      * it is *persistent* - present in essentially every frame.

    Persistence is what stops a voice being mistaken for hum.  A 120 Hz voice puts strong
    energy at harmonics of 60 Hz, so a purely spectral test flags clean speech as badly
    hummed; but pitch comes and goes with every pause, while hum never does.  Only the
    fundamental is tested, for the same reason - the harmonics are where voices collide.

    Long (~0.5 s) frames are used deliberately: the resolution has to be fine enough to
    separate a 60 Hz tone from its 6 Hz neighbourhood.
    """
    nper = min(int(2 ** np.ceil(np.log2(max(fs * 0.5, 64)))), x.size)
    if nper < 64:
        return 0.0, 60.0
    freqs, _t, Z = sig.stft(x, fs=fs, nperseg=nper, noverlap=nper // 2)
    power = np.abs(Z) ** 2

    best_score, best_base = 0.0, 60.0
    for base in (50.0, 60.0):
        peak_band = (freqs > base - 2) & (freqs < base + 2)
        side_band = (((freqs > base - 25) & (freqs < base - 6)) |
                     ((freqs > base + 6) & (freqs < base + 25)))
        if not np.any(peak_band) or not np.any(side_band):
            continue
        ratio = power[peak_band].max(axis=0) / (np.median(power[side_band], axis=0) + 1e-20)
        persistence = float(np.mean(ratio > 4.0))
        median_ratio = float(np.median(ratio))
        # 8x over the neighbourhood counts as nothing, 250x as certain, scaled on a log axis
        # because the ratio spans four orders of magnitude between faint and obvious hum.
        strength = float(np.clip((np.log10(max(median_ratio, 1.0)) - 0.9) / 1.3, 0.0, 1.0))
        score = strength * persistence
        if score > best_score:
            best_score, best_base = score, base
    return best_score, best_base


def diagnose_audio(x: np.ndarray, fs: float) -> dict:
    """Decide what is wrong with a signal.

    Returns scores in roughly 0-1 for each defect plus `verdict`, the preset that best
    matches.  The three tests look for genuinely different signatures:

      * hum    - a narrow peak at 50 or 60 Hz standing far above its spectral neighbours,
      * clicks - isolated samples far outside the local distribution,
      * noise  - a high spectral flatness in the quietest frames, which is what broadband
                 noise looks like when the signal itself is absent.
    """
    out = {"hum": 0.0, "clicks": 0.0, "noise": 0.0, "verdict": "Background noise"}
    if x.size < 256:
        return out

    out["hum"], out["hum_base"] = _hum_score(x, fs)

    # --- clicks: samples far from the local median, measured robustly
    med = sig.medfilt(x, kernel_size=5)
    resid = x - med
    active = resid[np.abs(resid) > 1e-12]
    if active.size > 16:
        scale = float(np.median(np.abs(active - np.median(active))) / 0.6745)
        if scale > 1e-12:
            rate = float(np.mean(np.abs(resid) > 8.0 * scale))
            out["clicks"] = float(np.clip(rate / 0.01, 0.0, 1.0))

    # --- broadband noise: how noise-like are the quietest 20% of frames?
    _t, frames = A.frame_signal(x, fs, 32.0, 16.0)
    if frames.shape[0] >= 5:
        energy = np.mean(frames ** 2, axis=1)
        quiet = frames[energy <= np.percentile(energy, 20)]
        if quiet.size:
            flat = A.spectral_flatness(quiet.ravel(), fs)
            floor = float(np.sqrt(np.mean(quiet ** 2)))
            peak = float(np.sqrt(np.mean(x ** 2))) or 1e-12
            # Loud *and* noise-like floor means real background noise, not just silence.
            out["noise"] = float(np.clip(flat * 1.6, 0, 1) * np.clip(floor / peak / 0.1, 0, 1))

    ranked = max(("hum", "clicks", "noise"), key=lambda k: out[k])
    if out[ranked] < 0.12:
        out["verdict"] = "Background noise"        # nothing stands out; the safe default
    else:
        out["verdict"] = {"hum": "Mains hum", "clicks": "Clicks & pops",
                          "noise": "Background noise"}[ranked]
    return out


def clean_audio(x: np.ndarray, fs: float, preset: str = AUTO, strength: float = 0.5,
                noise_clip: np.ndarray | None = None):
    """Apply a cleanup preset.  Returns (cleaned signal, human-readable description).

    `strength` is a single 0-1 dial.  Each preset maps it onto the parameter that actually
    controls how hard that algorithm works, over a range chosen to stay useful at both ends -
    at 0 the effect is gentle but real, at 1 it is aggressive but not yet destructive.
    """
    x = np.asarray(x, dtype=np.float64)
    chosen = preset
    note = ""

    if preset == AUTO:
        diag = diagnose_audio(x, fs)
        chosen = diag["verdict"]
        note = f"detected {chosen.lower()}"
        if chosen == "Mains hum":
            base = diag.get("hum_base", 60.0)
            y = D.remove_hum(x, fs, base=base, harmonics=int(_lerp(2, 5, strength)), q=30.0)
            return y, f"Auto: {note} at {base:.0f} Hz — notched it and its harmonics"

    if chosen == "Background noise":
        atten = _lerp(6.0, 26.0, strength)
        y = D.wiener_frequency(x, fs, noise_clip=noise_clip, max_atten_db=atten)
        desc = f"Wiener filter, up to {atten:.0f} dB of suppression"
    elif chosen == "Hiss / static":
        scale = _lerp(0.5, 2.0, strength)
        y = D.wavelet_denoise(x, wavelet="db4", rule="bayes", mode="soft",
                              threshold_scale=scale)
        desc = f"wavelet shrinkage, threshold x{scale:.2f}"
    elif chosen == "Mains hum":
        base = 60.0
        diag = diagnose_audio(x, fs)
        base = diag.get("hum_base", 60.0)
        harmonics = int(_lerp(2, 5, strength))
        y = D.remove_hum(x, fs, base=base, harmonics=harmonics, q=30.0)
        desc = f"{base:.0f} Hz notch plus {harmonics - 1} harmonics"
    elif chosen == "Clicks & pops":
        sigma = _lerp(5.0, 2.0, strength)           # lower sigma catches more samples
        y = D.median_declick(x, size=5, only_outliers=True, sigma=sigma)
        desc = f"median replacement of samples beyond {sigma:.1f} sigma"
    elif chosen == "Smooth the waveform":
        window = int(_lerp(5, 25, strength)) | 1
        y = F.savitzky_golay(x, window=window, poly=3)
        desc = f"Savitzky-Golay, {window}-sample window"
    else:
        raise ValueError("Unknown cleanup preset: " + str(preset))

    if note:
        desc = f"Auto: {note} — {desc}"
    return np.nan_to_num(y), desc


# ----------------------------------------------------------------- image
def estimate_sharpness(img: np.ndarray) -> float:
    """A blur measure that noise does not fool.  Higher is sharper; below ~2 looks soft.

    The textbook measure is the plain variance of the Laplacian, but it conflates blur with
    noise badly enough to be useless here: adding noise to this project's test chart raises
    it from 147 to 1399, reporting a *blurrier* image as ten times sharper, because noise is
    all high-frequency content and that is exactly what the Laplacian responds to.

    Two corrections fix it.  A Gaussian first removes the noise the Laplacian would otherwise
    measure, and dividing by the image's own variance normalises out overall contrast, so a
    dark flat photograph and a bright punchy one land on the same scale.  The scaling factor
    only puts typical values in a readable 0-15 range.

    The pre-smoothing width is a measured compromise.  At sigma 1.0 adding noise still
    inflated the result fourfold; at 2.0 it inflates it by 1.25x while a blurred image still
    reads at a third of its sharp value, which is enough separation to act on.  Even so this
    is a *relative* measure - absolute values depend on image content, so the before/after
    comparison is trustworthy in a way that a single reading is not.
    """
    grey = I.to_gray(img)
    smoothed = ndimage.gaussian_filter(grey, 2.0)
    variance = float(np.var(grey))
    return float(ndimage.laplace(smoothed).var() / max(variance, 1e-9) * 10000.0)


def estimate_noise(img: np.ndarray) -> float:
    """Immerkaer's noise estimate: the standard deviation of additive noise, in grey levels.

    Convolving with a kernel that is blind to smooth gradients and to linear edges leaves
    mostly noise behind, so its mean absolute response scales with sigma.  This is what lets
    the auto mode decide whether denoising is needed at all.

    Colour images are measured one channel at a time and the estimates averaged.  Running it
    on the luminance instead would read low by a fixed factor - the BT.601 weights give
    sqrt(0.299^2 + 0.587^2 + 0.114^2) = 0.669 - because averaging three independent noise
    channels partly cancels the noise, and the result would understate what the viewer sees
    by a third.

    The kernel is only blind to *linear* structure, though.  Fine texture (foliage, roof
    tiles, rippling water) leaks through it and reads as noise, so on real photographs the
    plain estimate ran high: a clean, detailed landscape measured sigma 8.8, and the app
    "denoised" photos that had nothing wrong with them.  Following Tai & Yang (2008), the
    response is averaged only over the flattest 30% of pixels, ranked by Sobel gradient
    magnitude, where texture cannot hide.  Over 11 test images and four noise levels that
    cut the mean error from 10.7% to 4.7%, and the worst from 70% to 31%, while the flat
    synthetic scene still reads exactly.
    """
    img = np.asarray(img, dtype=np.float64)
    if I.is_colour(img):
        return float(np.mean([estimate_noise(img[..., c]) for c in range(img.shape[2])]))
    if min(img.shape[:2]) < 4:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = np.abs(ndimage.convolve(img, kernel, mode="reflect"))[1:-1, 1:-1]
    gradient = np.hypot(ndimage.sobel(img, 0), ndimage.sobel(img, 1))[1:-1, 1:-1]
    flat = gradient <= np.percentile(gradient, NOISE_FLAT_FRACTION * 100.0)
    return float(np.mean(response[flat]) * np.sqrt(0.5 * np.pi) / 6.0)


#: Share of pixels, flattest first, that the image noise estimate is measured over.
NOISE_FLAT_FRACTION = 0.3


def analyse_image(img: np.ndarray) -> dict:
    """Describe an image's condition in terms a person can act on."""
    g = I.to_gray(img)
    sharp = estimate_sharpness(img)
    noise = estimate_noise(img)
    contrast = float(np.std(g))
    brightness = float(np.mean(g))
    return {"Sharpness": sharp,
            "Sharpness verdict": ("very blurry" if sharp < 2.0 else
                                  "soft" if sharp < 5.0 else
                                  "acceptable" if sharp < 10.0 else "sharp"),
            "Noise (sigma)": noise,
            "Noise verdict": ("clean" if noise < 2 else
                              "slightly noisy" if noise < 6 else
                              "noisy" if noise < 14 else "very noisy"),
            "Contrast (std)": contrast,
            "Contrast verdict": ("flat" if contrast < 35 else
                                 "low" if contrast < 55 else "good"),
            "Brightness": brightness,
            "Brightness verdict": ("dark" if brightness < 85 else
                                   "bright" if brightness > 175 else "balanced")}


def denoise_image(img: np.ndarray, strength: float = 0.5,
                  salt_pepper: bool = False) -> np.ndarray:
    """Edge-preserving denoise.  `strength` is 0-1.

    Bilateral filtering is the default because it averages only over pixels of *similar
    intensity*, so it smooths flat areas without softening edges the way a Gaussian would.
    Impulse noise is a different problem and needs a median instead - averaging a spike with
    its neighbours spreads it rather than removing it.
    """
    t = float(np.clip(strength, 0.0, 1.0))
    if t <= 0.01:
        return np.asarray(img, dtype=np.float64)
    if salt_pepper:
        return I.spatial_filter(img, "Median", size=int(_lerp(3, 7, t)) | 1)
    return I.spatial_filter(img, "Bilateral", size=int(_lerp(5, 11, t)) | 1,
                            sigma=_lerp(8, 40, t), sigma_colour=_lerp(25, 110, t))


def wavelet_denoise_image(img: np.ndarray, sigma: float) -> np.ndarray:
    """Cycle-spun BayesShrink: the image counterpart of the audio wavelet denoiser.

    Each colour channel is decomposed three levels deep with db4, every detail band is soft-
    thresholded at T = sigma^2 / sigma_signal (so bands rich in real detail are thresholded
    gently and noise-only bands hard), and the result is averaged over two shifts.  The shift
    averaging ("cycle spinning") removes the blocky artefacts a single decimated transform
    leaves at its grid positions; four shifts measured no better (32.942 dB against 32.940
    over 9 photos at three noise levels) at twice the cost.  Unlike a bilateral filter it does
    not flatten fine texture, which is why it wins on real photographs.
    """
    import pywt
    x = np.asarray(img, dtype=np.float64)
    channels = x[..., None] if x.ndim == 2 else x
    s2 = float(sigma) ** 2
    out = np.zeros_like(channels)
    for dy, dx in ((0, 0), (4, 4)):
        shifted = np.roll(channels, (dy, dx), (0, 1))
        for c in range(channels.shape[2]):
            coeffs = pywt.wavedec2(shifted[..., c], "db4", level=3, mode="periodization")
            kept = [coeffs[0]]
            for band in coeffs[1:]:
                kept.append(tuple(
                    pywt.threshold(b, s2 / np.sqrt(max(float(np.mean(b ** 2)) - s2, 1e-6)),
                                   "soft") for b in band))
            rec = pywt.waverec2(kept, "db4", mode="periodization")
            out[..., c] += np.roll(rec[:channels.shape[0], :channels.shape[1]], (-dy, -dx),
                                   (0, 1))
    out /= 2.0
    return out[..., 0] if x.ndim == 2 else out


def _sure_mse(denoise, y: np.ndarray, sigma: float, rng) -> tuple:
    """Monte-Carlo SURE: an unbiased estimate of a denoiser's mean squared error, computed
    from the noisy image alone.

        MSE ≈ |y - f(y)|² / N  -  sigma²  +  2 sigma² div f / N

    The divergence (how strongly the output follows small changes in the input) has no closed
    form for a bilateral or wavelet filter, so it is measured by probing: div ≈ bᵀ(f(y + εb) -
    f(y)) / ε for a random ±1 image b.  Returns (estimated MSE, f(y)).
    """
    fy = denoise(y)
    eps = max(float(sigma), 1.0) * 0.1
    b = rng.choice([-1.0, 1.0], size=y.shape)
    div = float(np.sum(b * (denoise(y + eps * b) - fy))) / eps
    n = y.size
    return (float(np.sum((y - fy) ** 2)) / n - sigma ** 2 + 2.0 * sigma ** 2 * div / n), fy


def auto_denoise(img: np.ndarray, sigma: float):
    """Pick the best denoiser for *this* image, without a clean reference, by SURE.

    No single method wins everywhere.  Measured over 9 Kodak photographs and 2 synthetic
    images at three noise levels: the bilateral filter is best on clean-edged, flat-shaded
    pictures but flattens the texture of real photos so badly it scored *below doing
    nothing* there (29.2 dB against 30.5); wavelet shrinkage is the reverse.  SURE estimates
    each candidate's true error from the noisy image itself and keeps the lowest: 32.75 dB on
    the photos (oracle 32.76) and 40.8 dB on the synthetic images (oracle 40.8).

    Returns (denoised image, name of the method chosen).
    """
    y = np.asarray(img, dtype=np.float64)
    strength = float(np.clip((sigma - 2.5) / 45.0, 0.2, 0.55))
    # Two candidates, not more.  A third ("gentle bilateral") won SURE's vote on textured
    # photos at high noise while scoring worse than wavelet on brightness, because SURE
    # minimises colour error and the eye - and PSNR/SSIM - mostly see brightness.  With the
    # two, SURE matched the oracle on photos (32.75 dB against 32.76).
    candidates = {
        "bilateral filter": lambda v: denoise_image(v, strength),
        "wavelet shrinkage": lambda v: wavelet_denoise_image(v, sigma),
    }
    # The vote is taken on a central 256x256 crop - plenty of pixels for an unbiased error
    # estimate - and only the winner is run on the full image.  Same choice, a fraction of
    # the time (the full-size vote cost 2.4 s on a 768x512 photo).
    h, w = y.shape[:2]
    top, left = max((h - 256) // 2, 0), max((w - 256) // 2, 0)
    crop = y[top:top + 256, left:left + 256]
    rng = np.random.default_rng(0)                      # the same choice every run
    risks = {name: _sure_mse(fn, crop, sigma, rng)[0] for name, fn in candidates.items()}
    choice = min(risks, key=risks.get)
    return candidates[choice](y), choice


def deblur_image(img: np.ndarray, strength: float = 0.5,
                 psf: np.ndarray | None = None) -> np.ndarray:
    """Recover detail lost to blur.  `strength` is 0-1.

    When the blur kernel is known - because it was applied inside the application - the
    result is a true Wiener deconvolution against that exact PSF.  When it is not, the blur
    is assumed Gaussian with a width scaled by `strength`; that is an approximation, but
    deconvolving against an approximately right kernel still recovers far more detail than
    sharpening alone, which only boosts what survived.
    """
    t = float(np.clip(strength, 0.0, 1.0))
    if t <= 0.01:
        return np.asarray(img, dtype=np.float64)
    if psf is not None:
        # K is a noise-to-signal ratio, so it is swept on a log scale: 0.05 (cautious) down to
        # 0.0005 (a near-exact inverse, right for a clean image whose kernel is known).
        return I.wiener_deconvolution(img, psf, k=0.05 * 0.01 ** t)
    kernel = I.gaussian_psf(size=9, sigma=_lerp(0.8, 2.6, t))
    # More regularisation at low strength keeps a cautious setting from amplifying noise.
    k = _lerp(0.05, 0.004, t)
    return I.wiener_deconvolution(img, kernel, k=k)


def wiener_k(noise_sigma: float) -> float:
    """Wiener regularisation for a known kernel, from the image's measured noise.

    K stands for the noise-to-signal power ratio, so it should grow with sigma squared.  The
    constants come from a sweep over 24 blur/noise combinations on two test images: with no
    noise the best K was 0.0005 (anything larger just leaves blur behind, which is what made
    the old fixed K = 0.02 look soft), and 3e-5 * sigma^2 tracked the optimum as noise rose.
    """
    return 0.0005 + 3e-5 * float(noise_sigma) ** 2


def sharpen_image(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Unsharp mask: add back a scaled copy of the detail a blur would remove."""
    t = float(np.clip(amount, 0.0, 1.0))
    if t <= 0.01:
        return np.asarray(img, dtype=np.float64)
    return I.spatial_filter(img, "Unsharp mask", size=5, sigma=1.2, strength=_lerp(0.2, 1.8, t))


def fix_contrast(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Local contrast enhancement by CLAHE, blended with the original by `amount`.

    CLAHE equalises within tiles rather than globally, so it lifts detail out of shadows
    without blowing out areas that were already correctly exposed; the clip limit is what
    stops it amplifying noise in flat regions.  Blending keeps the effect controllable.
    """
    t = float(np.clip(amount, 0.0, 1.0))
    if t <= 0.01:
        return np.asarray(img, dtype=np.float64)
    strong = I.histogram_op(img, "CLAHE (adaptive)", clip_limit=_lerp(1.2, 3.5, t), tile=8)
    return np.asarray(img, dtype=np.float64) * (1.0 - t) + strong * t


def auto_enhance(img: np.ndarray, psf: np.ndarray | None = None):
    """Diagnose an image and apply only the corrections it actually needs.

    Order matters: denoise first, because deconvolution amplifies whatever noise is present;
    then deblur; then contrast, so the contrast decision is made on the restored image rather
    than the degraded one.  Returns (restored image, list of what was done).

    The deblur step is deliberately split.  When the blur kernel is known - the image was
    blurred inside this application, so the PSF was recorded - a real Wiener deconvolution
    runs and genuinely recovers detail.  When it is not, no attempt is made to guess the
    kernel: blind blur estimation is unreliable in the presence of noise, and deconvolving
    against a wrong kernel adds ringing rather than detail, so the fallback is a modest
    unsharp mask, which only makes existing detail more visible and cannot invent artefacts.

    Each step is applied only when its own measurement says it is needed, so a clean image
    is left alone instead of being processed for the sake of it.
    """
    out = np.asarray(img, dtype=np.float64)
    steps = []

    # Periodic interference goes first: the noise estimate would otherwise read the stripes
    # as grain and denoise the whole picture to get rid of a pattern that a notch removes
    # almost for free.
    out, peaks = remove_pattern(out)
    if peaks:
        steps.append(f"notched {len(peaks)} interference peak"
                     f"{'s' if len(peaks) != 1 else ''} out of the 2-D spectrum")

    noise = estimate_noise(out)
    # 3.0, not lower: scanned film photos carry their own grain at sigma 2.5-3, and
    # 'denoising' an undamaged photo only strips its texture.
    if noise > 3.0:
        # Which denoiser is chosen per image by SURE - see auto_denoise for why one fixed
        # method is wrong for either photographs or flat-shaded pictures.
        out, method = auto_denoise(out, noise)
        steps.append(f"denoised with {method} (picked by SURE) — noise sigma was {noise:.1f}")

    if psf is not None:
        k = wiener_k(noise)
        out = I.wiener_deconvolution(out, psf, k)
        steps.append(f"deblurred by Wiener deconvolution against the measured blur kernel "
                     f"(K = {k:.4f}, set from the noise)")
        if noise > 2.5:
            # Deconvolution boosts exactly the high frequencies the denoiser left a little
            # noise in, which shows up as mottling.  A light second pass removes it; the sweep
            # put this at +1.6 dB PSNR and higher SSIM on average over deconvolution alone.
            out = denoise_image(out, 0.2)
            steps.append("light clean-up of the noise the deconvolution amplified")
    else:
        sharp = estimate_sharpness(out)
        if sharp < 5.0:
            amount = float(np.clip((5.0 - sharp) / 5.0, 0.25, 0.8))
            out = sharpen_image(out, amount)
            steps.append(f"sharpened — looked soft (sharpness {sharp:.1f}); the blur kernel "
                         f"is unknown, so no deconvolution was attempted")

    # Contrast is deliberately *not* part of auto.  Local contrast enhancement reliably makes
    # a flat image look better and reliably makes its PSNR worse - a 40% CLAHE blend cost
    # 10 dB in testing - because it moves every pixel away from the reference.  Fixing a
    # measurable defect and restyling an image are different jobs, so contrast stays a manual
    # dial the user chooses to turn.
    if not steps:
        steps.append("already in good condition — nothing needed changing")
    return np.clip(out, 0, 255), steps


def restore_image(img: np.ndarray, denoise: float = 0.0, deblur: float = 0.0,
                  sharpen: float = 0.0, contrast: float = 0.0,
                  psf: np.ndarray | None = None, salt_pepper: bool = False,
                  notch: bool = False):
    """Run the manual restoration chain in the order that works.

    Pattern removal first (it is exact and cheap, and the stripes would otherwise be treated
    as noise), denoise before deblur (deconvolution amplifies noise), deblur before sharpen (no
    point boosting detail that deconvolution is about to recover properly), contrast last.
    """
    out = np.asarray(img, dtype=np.float64)
    steps = []
    if notch:
        out, peaks = remove_pattern(out)
        steps.append(f"notch {len(peaks)} peak{'s' if len(peaks) != 1 else ''}"
                     if peaks else "notch (no pattern found)")
    if denoise > 0.01:
        out = denoise_image(out, denoise, salt_pepper)
        steps.append(f"denoise {denoise:.0%}")
    if deblur > 0.01:
        out = deblur_image(out, deblur, psf)
        steps.append(f"deblur {deblur:.0%}")
    if sharpen > 0.01:
        out = sharpen_image(out, sharpen)
        steps.append(f"sharpen {sharpen:.0%}")
    if contrast > 0.01:
        out = fix_contrast(out, contrast)
        steps.append(f"contrast {contrast:.0%}")
    return np.clip(out, 0, 255), (", ".join(steps) if steps else "no change")


def degrade_image(img: np.ndarray, blur: float = 0.0, noise: float = 0.0,
                  seed: int | None = None, pattern: float = 0.0):
    """Deliberately damage an image so a restoration can be measured against the original.

    Returns (degraded image, the PSF used) - keeping the PSF is what allows the deblur step
    to be a genuine inverse rather than a guess, which is the honest way to demonstrate
    deconvolution.

    `pattern` adds sinusoidal interference - the scan-line or mains-pickup pattern that is
    hopeless to remove spatially but collapses to two bright points in the 2-D spectrum.
    """
    out = np.asarray(img, dtype=np.float64)
    psf = None
    if blur > 0.01:
        psf = I.gaussian_psf(size=15, sigma=_lerp(0.6, 4.0, blur))
        out = I.blur(out, psf)
    if noise > 0.01:
        out = I.add_noise(out, "Gaussian", amount=_lerp(2.0, 35.0, noise), seed=seed)
    if pattern > 0.01:
        out = I.add_noise(out, "Periodic (sinusoidal)", amount=_lerp(15.0, 45.0, pattern),
                          freq=0.12)
    return np.clip(out, 0, 255), psf


# ----------------------------------------------------------------- periodic interference
#: How far a spectral peak must stand above its neighbourhood, in dB, to count as
#: interference.  Measured on the demo scene and the test chart: genuine content peaks at
#: 23 dB at most (the scene's ground texture), while the weakest pattern the Damage dial can
#: add stands at 34 dB or more.  30 sits between the two, so a real repeating texture is left
#: alone and added interference is caught.
PATTERN_PROMINENCE_DB = 30.0


def find_periodic_peaks(img: np.ndarray, max_peaks: int = 6,
                        threshold_db: float = PATTERN_PROMINENCE_DB):
    """Locate periodic interference as isolated peaks in the centred 2-D spectrum.

    `imaging.detect_periodic_peaks` simply takes the brightest off-centre points, which on a
    real photograph are just its strongest genuine content.  Measuring *prominence* - height
    above a 15x15 local mean of the log spectrum - instead asks whether a point sticks out of
    its surroundings, which is what a pure sinusoid does and natural content does not.

    The DC neighbourhood and the two axis lines through it are excluded: the image's own
    borders (the FFT treats the picture as periodic) put a bright cross through the centre
    that is not interference.  Returns [(row, col, prominence_db), ...].
    """
    spec = I.fft2_spectrum(img, log=True)
    h, w = spec.shape
    # A 15x15 mean rather than median: a lone peak moves a 225-pixel mean by ~0.2 dB, so it
    # measures the background just as well, at a seventh of the cost (0.1 s against 0.74 s).
    prom = spec - ndimage.uniform_filter(spec, size=15, mode="reflect")
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = max(6, int(0.04 * min(h, w)))
    prom[((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2] = -np.inf
    prom[max(cy - 2, 0):cy + 3, :] = -np.inf
    prom[:, max(cx - 2, 0):cx + 3] = -np.inf

    peaks = []
    for _ in range(max(int(max_peaks), 0)):
        idx = int(np.argmax(prom))
        py, px = divmod(idx, w)
        if not np.isfinite(prom[py, px]) or prom[py, px] < threshold_db:
            break
        peaks.append((py, px, float(prom[py, px])))
        prom[max(py - 4, 0):py + 5, max(px - 4, 0):px + 5] = -np.inf
    return peaks


def remove_pattern(img: np.ndarray, peaks=None):
    """Remove periodic interference found as peaks in the 2-D spectrum.

    Returns (cleaned image, peaks that were removed).  A sinusoid occupies a single point in
    the spectrum, so removing it costs almost nothing of the picture, whereas any spatial
    filter strong enough to flatten the stripes would blur everything else as well.

    The removal is a *parametric notch*.  A fixed band-stop notch only works when the stripe
    frequency lands exactly on a DFT bin.  Usually it does not (0.12 cycles/pixel across 420
    pixels is bin 50.4), so its energy leaks into the neighbouring bins, and a narrow notch
    leaves visible stripes along the borders while a wide one starts removing real detail.
    Instead, each peak's frequency is refined to a fraction of a bin, and that exact sinusoid
    is fitted by least squares and subtracted.  It is gone everywhere, borders included, and
    nothing else in the spectrum is touched.
    """
    if peaks is None:
        peaks = find_periodic_peaks(img)
    x = np.asarray(img, dtype=np.float64)
    if not peaks:
        return x, []
    h, w = x.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    grey = I.to_gray(x)
    grey = grey - grey.mean()

    out = x.copy()
    done = []
    for py, px, *_ in peaks:
        fy, fx = (py - h // 2) / h, (px - w // 2) / w      # cycles per pixel
        if any(abs(fy + gy) < 1.5 / h and abs(fx + gx) < 1.5 / w for gy, gx in done):
            continue                                        # the conjugate of one already done
        fy, fx = _refine_frequency(grey, fy, fx)
        done.append((fy, fx))
        phase = 2.0 * np.pi * (fy * yy + fx * xx)
        basis = np.stack([np.cos(phase).ravel(), np.sin(phase).ravel(),
                          np.ones(h * w)], axis=1)
        channels = out[..., None] if out.ndim == 2 else out
        for c in range(channels.shape[2]):
            coef, *_ = np.linalg.lstsq(basis, channels[..., c].ravel(), rcond=None)
            channels[..., c] -= (coef[0] * basis[:, 0] + coef[1] * basis[:, 1]).reshape(h, w)
    return np.clip(out, 0, 255), peaks


def _refine_frequency(grey, fy, fx, rounds: int = 2):
    """Sub-bin frequency estimate: maximise |sum g * exp(-j2pi (fy*y + fx*x))| near the peak.

    Searched one axis at a time over +/-0.6 bin, twice.  The kernel is separable,
    exp(-j2pi fy y) * exp(-j2pi fx x), so with one frequency held fixed the image collapses to
    a single vector once, and every candidate for the other costs one short dot product.
    """
    h, w = grey.shape
    ys, xs = np.arange(h), np.arange(w)
    offsets = np.linspace(-0.6, 0.6, 25)

    for _ in range(rounds):
        cols = grey @ np.exp(-2j * np.pi * fx * xs)                 # fixed fx -> length h
        cand = fy + offsets / h
        fy = cand[np.argmax(np.abs(np.exp(-2j * np.pi * np.outer(cand, ys)) @ cols))]
        rows = np.exp(-2j * np.pi * fy * ys) @ grey                 # fixed fy -> length w
        cand = fx + offsets / w
        fx = cand[np.argmax(np.abs(np.exp(-2j * np.pi * np.outer(cand, xs)) @ rows))]
    return float(fy), float(fx)


# ----------------------------------------------------------------- edges
EDGE_OPERATORS = ("Canny", "Sobel", "Prewitt", "Scharr", "Roberts cross", "Laplacian",
                  "Laplacian of Gaussian")
EDGE_VIEWS = ("Edge map", "Overlay")

#: One line per operator, shown under the chooser so the method can be explained on screen.
EDGE_HELP = {
    "Canny": "Gaussian smoothing → Sobel gradient → thin to 1-pixel ridges (non-maximum "
             "suppression) → hysteresis with two thresholds. The reference edge detector.",
    "Sobel": "First derivative. Kernels [-1 0 1; -2 0 2; -1 0 1] and its transpose give Gx and "
             "Gy; edge strength = √(Gx² + Gy²). The centre weight of 2 adds a little smoothing.",
    "Prewitt": "First derivative like Sobel but with equal weights [-1 0 1] in every row — "
               "simpler, slightly more sensitive to noise.",
    "Scharr": "A 3×3 derivative tuned for rotational symmetry ([-3 0 3; -10 0 10; -3 0 3]), so "
              "diagonal edges are measured as accurately as straight ones.",
    "Roberts cross": "The smallest gradient operator: two 2×2 diagonal differences. Very sharp "
                     "localisation, very noise-sensitive.",
    "Laplacian": "Second derivative ∇²f, kernel [0 1 0; 1 -4 1; 0 1 0]. It swings + then − "
                 "across a step, so the edge is where it crosses zero. Very noise-sensitive.",
    "Laplacian of Gaussian": "Marr–Hildreth: Gaussian (σ = 1.5) and Laplacian in one kernel, "
                             "edges at the zero-crossings. Smoothing is built in.",
}


def _edge_strength(img: np.ndarray, operator: str, smooth: bool) -> np.ndarray:
    grey = I.to_gray(img)
    if smooth and operator not in ("Canny", "Laplacian of Gaussian"):
        # A derivative is a high-pass filter, so it amplifies noise; a Gaussian low-pass
        # first is what keeps grain from turning into a field of false edges.
        grey = ndimage.gaussian_filter(grey, 1.2)
    return I.edge_detect(grey, operator, ksize=3, sigma=1.5)


def find_edges(img: np.ndarray, operator: str = "Canny", sensitivity: float = 0.5,
               smooth: bool = True) -> np.ndarray:
    """Edge map in [0, 255]: zero where there is no edge, edge strength where there is.

    One `sensitivity` dial (0-1) drives whichever threshold the operator needs.  For Canny it
    sets the hysteresis pair, high = 220 → 40 with low at 40% of it.  For the gradient and
    Laplacian operators it is a fraction of the 99.5th-percentile response, so the setting
    means the same thing on a dim image as on a bright one.
    """
    t = float(np.clip(sensitivity, 0.0, 1.0))
    if operator == "Canny":
        grey = I.to_gray(img)
        if smooth:
            # OpenCV's Canny skips the Gaussian stage of the textbook algorithm.
            grey = ndimage.gaussian_filter(grey, 1.4)
        high = _lerp(220.0, 40.0, t)
        return I.edge_detect(grey, "Canny", low=0.4 * high, high=high)

    if operator in ("Laplacian", "Laplacian of Gaussian"):
        return _zero_crossing_edges(img, operator, t, smooth)

    strength = _edge_strength(img, operator, smooth)
    scale = max(float(np.percentile(strength, 99.5)), 1e-9)
    norm = strength / scale
    keep = norm >= _lerp(0.6, 0.08, t)
    return np.where(keep, np.clip(norm, 0.0, 1.0) * 255.0, 0.0)


def _zero_crossing_edges(img: np.ndarray, operator: str, t: float, smooth: bool) -> np.ndarray:
    """Second-derivative edges, located where the Laplacian changes sign.

    Across a step the second derivative swings positive on one side and negative on the
    other, so thresholding its *magnitude* marks both sides and draws every outline twice.
    The edge itself is the zero-crossing between them (Marr-Hildreth).  Each crossing is
    marked on the pixel nearer to zero, and weighted by the size of the swing, so the
    sensitivity dial can drop the weak crossings that flat-region noise produces.
    """
    grey = I.to_gray(img)
    if operator == "Laplacian of Gaussian":
        lap = ndimage.gaussian_laplace(grey, 1.5)
    else:
        if smooth:
            grey = ndimage.gaussian_filter(grey, 1.2)
        lap = ndimage.laplace(grey)

    strength = np.zeros_like(lap)
    for axis in (0, 1):
        a = lap[:-1, :] if axis == 0 else lap[:, :-1]
        b = lap[1:, :] if axis == 0 else lap[:, 1:]
        swing = np.where(a * b < 0, np.abs(a - b), 0.0)
        mark_a = swing * (np.abs(a) <= np.abs(b))
        mark_b = swing * (np.abs(a) > np.abs(b))
        if axis == 0:
            strength[:-1, :] = np.maximum(strength[:-1, :], mark_a)
            strength[1:, :] = np.maximum(strength[1:, :], mark_b)
        else:
            strength[:, :-1] = np.maximum(strength[:, :-1], mark_a)
            strength[:, 1:] = np.maximum(strength[:, 1:], mark_b)

    crossings = strength[strength > 0]
    if crossings.size == 0:
        return np.zeros_like(strength)
    norm = strength / max(float(np.percentile(crossings, 99.0)), 1e-9)
    keep = norm >= _lerp(0.6, 0.08, t)
    return np.where(keep, np.clip(norm, 0.0, 1.0) * 255.0, 0.0)


def overlay_edges(img: np.ndarray, edges: np.ndarray,
                  colour=(229, 50, 62)) -> np.ndarray:
    """Paint the edges onto a dimmed copy of the image, so they can be checked against the
    picture they came from."""
    base = np.asarray(img, dtype=np.float64)
    if not I.is_colour(base):
        base = np.stack([base] * 3, axis=2)
    out = base * 0.55
    mask = edges > 0
    out[mask] = colour
    return out


def edge_density(edges: np.ndarray) -> float:
    """Fraction of pixels marked as edge, in percent."""
    return float(np.mean(edges > 0) * 100.0)


def edge_match(reference: np.ndarray, test: np.ndarray, tolerance: int = 1) -> dict:
    """How well one edge map agrees with another, allowing `tolerance` pixels of misplacement.

    Precision is the share of detected edges that are real (lie near a reference edge) - noise
    drags it down by adding false edges.  Recall is the share of real edges that were found -
    blur drags it down by washing edges out.  F1 is their harmonic mean, so it only scores
    well when both are good.
    """
    ref, tst = reference > 0, test > 0
    size = 2 * max(int(tolerance), 0) + 1
    near_ref = ndimage.binary_dilation(ref, np.ones((size, size), bool))
    near_tst = ndimage.binary_dilation(tst, np.ones((size, size), bool))
    precision = float(np.sum(tst & near_ref) / max(np.sum(tst), 1))
    recall = float(np.sum(ref & near_tst) / max(np.sum(ref), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"Precision": precision, "Recall": recall, "F1": f1}


class EdgeSettings:
    """The edge-detection choices as one object, so the image page, the video preview and the
    video export all render edges identically."""

    def __init__(self, operator: str = "Canny", sensitivity: float = 0.5,
                 smooth: bool = True, view: str = "Edge map"):
        self.operator = operator
        self.sensitivity = float(sensitivity)
        self.smooth = bool(smooth)
        self.view = view

    def describe(self) -> str:
        return (f"{self.operator} edges, sensitivity {self.sensitivity:.0%}"
                + (", pre-smoothed" if self.smooth else ""))

    def detect(self, img: np.ndarray) -> np.ndarray:
        return find_edges(img, self.operator, self.sensitivity, self.smooth)

    def render(self, img: np.ndarray, edges: np.ndarray | None = None) -> np.ndarray:
        """The image to display: the edge map itself, or the edges over the picture."""
        if edges is None:
            edges = self.detect(img)
        if self.view == "Overlay":
            return overlay_edges(img, edges)
        return np.stack([edges] * 3, axis=2)


def image_quality(reference: np.ndarray, test: np.ndarray) -> dict:
    """PSNR and SSIM against a reference, plus the sharpness each one has."""
    out = I.compare(reference, test)
    out["Sharpness before"] = estimate_sharpness(reference)
    out["Sharpness after"] = estimate_sharpness(test)
    return out


# ----------------------------------------------------------------- video
class VideoCleaner:
    """Per-frame restoration with the same dials as the image page, plus temporal filtering.

    Temporal filtering is the one tool a video has that a still image does not: noise is
    independent between frames while the scene is not, so averaging over time removes noise
    without costing any spatial detail.  A plain average of the last few frames smears
    anything that moves, so this uses a *motion-adaptive recursive* filter instead:

        out[n] = out[n-1] + a * (frame[n] - out[n-1])

    with the blend factor `a` chosen per pixel.  Where the frame agrees with the running
    estimate to within the noise, nothing moved, `a` is small and the pixel keeps averaging
    (a = 0.2 averages roughly the last 9 frames, cutting noise by about 3x).  Where they
    disagree by much more than the noise could explain, something moved, and `a` goes to 1
    so the new frame is taken as it is.  Static areas get heavy averaging and moving
    objects stay sharp.
    """

    def __init__(self, denoise: float = 0.0, deblur: float = 0.0, sharpen: float = 0.0,
                 contrast: float = 0.0, temporal: bool = True):
        self.denoise = float(denoise)
        self.deblur = float(deblur)
        self.sharpen = float(sharpen)
        self.contrast = float(contrast)
        self.temporal = temporal
        self.history: list = []
        self._estimate = None
        self._sigma = None

    def reset(self) -> None:
        self.history = []
        self._estimate = None
        self._sigma = None

    @property
    def window(self) -> int:
        """How many frames the temporal filter needs to settle; 1 means it is off.

        The preview runs this many frames ahead of the one it shows, so what is displayed is
        the filter's steady state rather than its start-up.
        """
        return 12 if (self.temporal and self.denoise > 0.05) else 1

    @property
    def blend(self) -> float:
        """Blend factor for a static pixel: 0.5 at the lowest setting, 0.12 at the highest."""
        return _lerp(0.5, 0.12, (self.denoise - 0.05) / 0.95)

    def describe(self) -> str:
        bits = []
        if self.denoise > 0.01:
            bits.append(f"denoise {self.denoise:.0%}"
                        + (" (+motion-adaptive temporal)" if self.window > 1 else ""))
        if self.deblur > 0.01:
            bits.append(f"deblur {self.deblur:.0%}")
        if self.sharpen > 0.01:
            bits.append(f"sharpen {self.sharpen:.0%}")
        if self.contrast > 0.01:
            bits.append(f"contrast {self.contrast:.0%}")
        return ", ".join(bits) if bits else "no change"

    def process(self, frame: np.ndarray) -> np.ndarray:
        out = np.asarray(frame, dtype=np.float64)

        if self.window > 1:
            out = self._temporal(out)

        if self.denoise > 0.01:
            # Spatial denoise is scaled back when temporal filtering already did the work.
            spatial = self.denoise * (TEMPORAL_SPATIAL_SHARE if self.window > 1 else 1.0)
            out = denoise_image(out, spatial)
        if self.deblur > 0.01:
            out = deblur_image(out, self.deblur)
        if self.sharpen > 0.01:
            out = sharpen_image(out, self.sharpen)
        if self.contrast > 0.01:
            out = fix_contrast(out, self.contrast)
        return np.clip(out, 0, 255)

    def _temporal(self, frame: np.ndarray) -> np.ndarray:
        if self._estimate is None or self._estimate.shape != frame.shape:
            self._estimate = frame.copy()
            self._sigma = max(estimate_noise(frame), 1.0)
            return frame
        # Motion test on a smoothed luminance difference: smoothing averages the noise in the
        # difference down, so a real change stands out from grain instead of hiding in it.
        diff = ndimage.gaussian_filter(np.abs(I.to_gray(frame) - I.to_gray(self._estimate)),
                                       TEMPORAL_DIFF_SMOOTH)
        low, high = TEMPORAL_MOTION_LOW * self._sigma, TEMPORAL_MOTION_HIGH * self._sigma
        a = np.clip((diff - low) / max(high - low, 1e-9), 0.0, 1.0)
        a = self.blend + (1.0 - self.blend) * a
        if frame.ndim == 3:
            a = a[..., None]
        self._estimate = self._estimate + a * (frame - self._estimate)
        return self._estimate.copy()


#: Motion-adaptive temporal filter tuning, in units of the measured noise sigma.  Below LOW a
#: pixel counts as static and is averaged; above HIGH it counts as moving and is passed
#: through; in between it is blended.  Set by sweeping against the clean test clip, where it
#: scored 37.6 dB against 34.5 dB for the old plain 3-frame average - and 33.9 dB against 20.9
#: dB on the moving ball, which the plain average smeared.  The spatial share is small because
#: the temporal filter has already done most of the work, and any more just softens detail.
TEMPORAL_MOTION_LOW = 0.6
TEMPORAL_MOTION_HIGH = 1.6
TEMPORAL_DIFF_SMOOTH = 1.5
TEMPORAL_SPATIAL_SHARE = 0.1


def auto_video_settings(frame: np.ndarray) -> "VideoCleaner":
    """Choose the dials from one representative frame.

    Only the noise estimate drives a strength, because it is the one measurement that is
    reliable on arbitrary content.  Sharpening is a fixed modest amount and deconvolution is
    left off entirely: there is no known blur kernel for a loaded clip, and the blur metric
    is content-dependent enough that acting on it automatically does more harm than good.
    """
    return VideoCleaner(
        denoise=float(np.clip((estimate_noise(frame) - 2.5) / 15.0, 0.0, 1.0)),
        deblur=0.0,
        sharpen=0.3,
        contrast=0.0)
