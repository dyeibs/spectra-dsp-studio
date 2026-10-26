# SPECTRA — Defence Guide

How each rubric criterion is met, where to show it in the app, which code implements it, and
the questions most likely to come up. Every member should be able to answer every section.

Before the defence: run `python validate.py` (expect **157/157**) and keep `exports/` empty
so the PDF you generate live is the one you show.

---

## Rubric → where to show it

| # | Criterion (pts) | Show this | Code |
|---|---|---|---|
| 1 | System functionality (25) | Run the whole demo script in the README end to end; `validate.py` 157/157 | `validate.py` |
| 2 | DSP implementation (20) | Filter design, Wiener, deconvolution, notch, edge operators — explain the formulas below | `dsp/` |
| 3 | Signal processing application (15) | **Audio** (Audio Studio), **speech** (Audio Analysis → Speech), **image** (Image Restore), **video** (Video Restore) | `dsp/presets.py`, `dsp/imaging.py`, `dsp/video.py` |
| 4 | Filtering & enhancement (15) | Noise reduction: Clean up, Remove noise. Filtering: Filter tool, notch. Restoration: Wiener deconvolution, de-click. Enhancement: sharpen, contrast (CLAHE), edges | `dsp/denoise.py`, `dsp/filters.py`, `dsp/imaging.py` |
| 5 | Output & visualisation (10) | Before/after waveform, spectrum, spectrogram; before/after image, histogram, 2-D spectrum, edge maps; video frame pair, histogram, motion energy; **PDF report** | `app/pages/*`, `app/pages/report.py` |
| 6 | Design & UI (5) | Sidebar pages, one job per page, Auto buttons, dials named after the problem not the algorithm | `app/` |
| 7 | Demo & defence (10) | This document | — |

---

## Audio and speech

**Sampling & quantisation** — `dsp/signals.py: quantize, downsample_demo`
- Quantisation SNR ≈ **6.02·B + 1.76 dB**. Each bit adds about 6 dB. The app shows measured vs theoretical.
- Aliasing: a 3 kHz tone decimated by 4 (8 kHz → 2 kHz, Nyquist 1 kHz) folds to |3000 − 2000| = **1 kHz**. With the anti-alias low-pass on, it is removed instead.
- *Q: Why must the anti-alias filter come before downsampling?* Once a frequency has folded, it is indistinguishable from a real one at the new rate, so it can't be removed afterwards.

**FIR vs IIR filters** — `dsp/filters.py: design_fir, design_iir, apply_filter`
- FIR: finite impulse response, always stable (poles only at the origin), and **linear phase** when symmetric (constant group delay). Needs many taps for a sharp cut.
- IIR: feedback, so much sharper for the same order, but non-linear phase, and stability depends on the poles staying inside the unit circle. Designed as **second-order sections (SOS)** for numerical stability.
- Families: Butterworth (maximally flat), Chebyshev I (passband ripple), Chebyshev II (stopband ripple), Elliptic (both; sharpest), Bessel (best phase).
- `apply_filter` uses **filtfilt** (forward then backward), which gives zero phase and squares the magnitude response.
- *Q: Why redesign the filter when the sample rate changes?* Cutoffs are normalised to fs. An 8 kHz design reused at 16 kHz doubles every cutoff.

**Noise reduction** — `dsp/denoise.py`
- **Wiener filter** (`wiener_frequency`): STFT, then per-bin gain G = ξ/(1+ξ), where ξ is the a-priori SNR, estimated **decision-directed** (98% of the previous frame's clean estimate, 2% of the current measurement), then inverse STFT. A gain floor (−18 dB) prevents "musical noise".
- **Spectral subtraction**: |Y|² − α·|N|². Simpler, but produces musical noise.
- **Wavelet shrinkage (BayesShrink)**: noise spreads thinly over all coefficients while the signal concentrates in a few large ones. The noise σ is estimated as MAD/0.6745 from the finest band, and the per-band threshold is T = σ²/σₓ.
- **Hum**: cascaded IIR notches at 50/60 Hz and its harmonics. **Clicks**: median replacement of outlier samples only.
- *Q: How does Auto know it's hum and not a 120 Hz voice?* Hum is present in **every** frame. A voice's pitch stops in the pauses (persistence test).
- *Q: How does the Wiener filter know the noise on a recording?* **Minimum statistics**: speech comes and goes in each frequency bin, so the low tail of a bin's magnitude over time is mostly noise. Take the 15th percentile, then correct its bias: noise magnitude is Rayleigh-distributed, and its 15th percentile is only 0.455× the mean. The correction used is ×1.5, not the full ×2.2, because speech lifts the low tail and full correction over-subtracts. Frames of digital silence (the mic starting up) are excluded. Measured result: +2.7 → +5.2 dB average SNR gain.
- *Q: How do you show improvement on a recording with no clean copy?* Background level (quietest frames) and voice level (loudest frames minus the noise share), before and after. Not an "SNR", because a denoiser silences the pauses much more than it cleans under the speech, and that would overstate the gain about 3×.
- *Q: Why is THD "n/a" for speech?* THD is defined for a periodic tone. It is shown only when at least half the energy is on one fundamental and its harmonics.

**Speech analysis** — `dsp/analysis.py`
- **STFT / spectrogram**: the frame-length slider is the time–frequency uncertainty trade-off. Short frames sharpen time; long frames sharpen frequency.
- **Pitch (F0)**: autocorrelation of each frame. The peak lag within [fs/400, fs/60] gives F0 = fs/lag. Frames with a weak peak are unvoiced.
- **MFCC**: pre-emphasis (y[n] = x[n] − 0.97x[n−1]) → Hamming window → |FFT|² → 26 mel filters → log → **DCT-II** → 13 coefficients. The DCT decorrelates the log energies.
- **Pre-emphasis** boosts high frequencies, which speech loses at about −6 dB/octave.

**Metrics** — `dsp/metrics.py`: SNR = 10·log₁₀(Σs² / Σ(s−ŝ)²), segmental SNR (frame-averaged), THD.

---

## Image

**Damage → repair, measured** — `dsp/presets.py: degrade_image, auto_enhance`
- "Damage it" records the blur kernel (PSF). That makes deblurring a true **Wiener deconvolution**: F̂ = G·H* / (|H|² + K).
- *Q: Why not just divide by H (inverse filter)?* H is near zero at high frequencies, so dividing amplifies noise without limit. K (the noise-to-signal ratio) bounds the division.
- Images are padded symmetrically before the FFT, because the FFT assumes the image wraps around, and that causes ringing at the borders.
- **Denoise**: two candidates. A **bilateral filter** weights neighbours by distance *and* intensity similarity, so it smooths flat areas but not edges. **Wavelet shrinkage** (BayesShrink, cycle-spun) keeps fine texture. A median filter is used for salt-and-pepper noise.
- **Auto picks between them with SURE** (Stein's Unbiased Risk Estimate): MSE ≈ |y − f(y)|²/N − σ² + 2σ²·div f/N. It estimates each denoiser's true error from the noisy image alone. The divergence is measured by probing: div ≈ bᵀ(f(y+εb) − f(y))/ε for a random ±1 image b. On 9 real photos this reached 32.75 dB against 32.76 for an oracle that knows the original.
- *Q: Why not just use one denoiser?* Bilateral was best on flat-shaded images but scored *below doing nothing* on real photos (29.2 vs 30.5 dB), because it flattens texture. Wavelet is the reverse. No single method wins.
- **Noise estimate** (Immerkaer): convolve with [1 −2 1; −2 4 −2; 1 −2 1], which cancels smooth gradients and straight edges, so what remains is mostly noise; σ = √(π/2)·mean|response|/6. It is measured only over the **flattest 30% of pixels** (Tai & Yang 2008), because texture leaks through the kernel and reads as noise: a clean landscape measured σ = 8.8 before this.
- **Contrast**: CLAHE equalises the histogram per tile, with a clip limit so noise isn't amplified.
- Order matters: denoise **before** deblur, because deconvolution amplifies noise.
- **K is set from the noise**: K = 0.0005 + 3×10⁻⁵·σ². K is a noise-to-signal ratio, so it scales with σ². With no noise, K ≈ 0 and the blur is inverted almost exactly. A light bilateral pass afterwards removes the noise the deconvolution amplified (otherwise it shows as mottling).
- **PSNR** = 10·log₁₀(255²/MSE). **SSIM** compares local mean, variance and covariance (structure), which matches perception better than PSNR.

**2-D spectrum and stripe removal** — `dsp/presets.py: find_periodic_peaks, remove_pattern`
- The 2-D DFT is shown centred (fftshift) on a log scale. Low frequencies are in the middle and fine detail is at the edge.
- A sinusoidal stripe pattern is **two conjugate points** in the spectrum. Remove them and the stripes vanish, with almost no loss to the picture.
- Peaks are detected by **prominence**: height above a 15×15 mean of the log spectrum. Real content sits at ≤23 dB; added stripes at ≥34 dB; the threshold is 30 dB.
- *Q: Why not a simple band-stop notch?* The stripe frequency (0.12 cycles/px × 420 px = bin 50.4) is not on a DFT bin, so its energy **leaks** into neighbouring bins. A narrow notch leaves stripes at the borders; a wide one removes real detail. So the frequency is refined to a fraction of a bin, and the exact sinusoid is fitted by least squares and subtracted (a parametric notch): +25.7 dB, against +15.5 dB for the fixed notch.
- *Q: Why not remove stripes with a blur?* A low-pass strong enough to flatten them blurs everything else as well. In the frequency domain the interference is isolated.

**Edge detection** — `dsp/presets.py: find_edges`, `dsp/imaging.py: edge_detect`
- An edge is a rapid change in brightness, so edge detectors are **derivatives**, which are **high-pass filters** and therefore amplify noise. That is why "Smooth first" (a Gaussian low-pass) matters: Canny's precision on a noisy square goes from 5% to 100%.
- **Sobel**: Gx = [−1 0 1; −2 0 2; −1 0 1], Gy = its transpose, magnitude √(Gx²+Gy²). The centre weight of 2 adds smoothing. **Prewitt**: equal weights. **Scharr**: tuned for rotational accuracy. **Roberts**: 2×2 diagonal differences.
- **Laplacian** ∇²f (second derivative): the edge is at the **zero-crossing**, not the peak. Thresholding the magnitude would draw every outline twice. **LoG** (Marr–Hildreth) = Gaussian then Laplacian.
- **Canny**: Gaussian → Sobel gradient → non-maximum suppression (thin to 1 px) → hysteresis with two thresholds (strong edges kept; weak edges kept only if connected to strong ones).
- *Q: Why doesn't Canny outline the buildings in the demo scene?* The grey buildings and the
  blue sky have almost the same **luminance** (129 vs 118 grey levels across the edge). They
  differ in colour, not brightness, and every detector here runs on luminance (Y = 0.299R +
  0.587G + 0.114B). Lowering the threshold doesn't help; it lets noise in instead (F1 drops
  from 0.93 to 0.59). A colour-gradient detector would be the fix.
- Edge accuracy is scored against the original's edges: **precision** (are found edges real? Noise lowers this), **recall** (were real edges found? Blur lowers this), and **F1**.

---

## Video

`app/pages/video.py`, `dsp/presets.py: VideoCleaner`, `dsp/video.py`
- A video is a 3-D signal (x, y, t). Averaging N frames cuts noise by √N with no spatial blur, because noise is independent between frames while the scene is not. A plain average smears anything that moves.
- **Motion-adaptive recursive filter**: out[n] = out[n−1] + a·(frame[n] − out[n−1]), a first-order IIR filter along time, with `a` chosen per pixel. If |frame − estimate| (smoothed) < 0.6σ the pixel is static, so a is small (0.12–0.5) and it keeps averaging. If > 1.6σ it moved, so a = 1 and the new frame passes through. On the test clip the moving ball scores 33.9 dB, against 20.9 dB for a plain 3-frame average.
- *Q: Why recursive instead of storing N frames?* It is an IIR filter: one stored frame gives an exponentially weighted average over many frames (a = 0.2 ≈ the last 9).
- Remove noise (temporal filter plus a light bilateral), Remove blur (Wiener with an assumed Gaussian PSF), Sharpen (unsharp mask: x + k·(x − blur(x))), Contrast (CLAHE).
- **Edge detection runs on the cleaned frame**: on raw grain, a derivative finds the grain.
- **Motion energy** = mean |frame − previous frame| over time, which turns the clip into a 1-D signal.
- Frames are streamed from disk (a generator), never loaded whole, so memory stays flat.

---

## Questions about the system as a whole

- *How do you know it's correct?* `validate.py` checks against **known answers**, not just "it runs". Examples: tone frequency, SNR within 0.1 dB, 6.02B + 1.76, aliasing to exactly 1 kHz, filter stability, every edge detector tracing a known outline, the notch recovering more than 6 dB.
- *Why does the hum notch score negative in Compare all?* That run is broadband noise, not hum. The notch removes a little signal and none of the noise, so a negative number is the correct answer.
- *Why does the UI stay responsive?* Heavy work runs on worker threads, which never touch a widget. Results come back through a queue.
- *What does the PDF contain?* Every metric, the processing pipeline in order, and figures for audio (waveform, spectrum, spectrograms), image (before/after, edges, 2-D spectra), and the last previewed video frame (original/processed, edges, histograms).
