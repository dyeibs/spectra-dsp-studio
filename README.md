# SPECTRA — DSP Studio

**Group 6** · CPE07A · DIGISIG · Engr. Rodante Santos

Elideros, John Ver D. (Leader) · Banal, Ryan P. · Batac, Jayson V. · Manuyag, Darwin F.


**S**ignal **P**rocessing, **E**nhancement, **C**haracterisation, **T**ransform & **R**estoration **A**pplication

An interactive digital signal processing workbench for **audio, speech, images and video**,
written in Python. Every page does one job: take a signal that has something wrong with it,
fix it, and measure how much better it got.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Two things to try first:

1. **Dashboard → Noisy speech**, then **Audio Studio → Clean up → Clean it up**. Auto works
   out that the problem is background noise and removes it; the improvement is shown in dB.
2. **Image Restore → Demo scene → Damage it → Auto enhance**. A blurry, grainy photo comes
   back sharp, with the PSNR and SSIM recovery measured against the untouched original.

To check the signal processing itself is correct:

```bash
python validate.py
```

---

## How it is organised

Seven pages, each with a handful of controls rather than a control panel:

| Page | What it is for |
|---|---|
| **Dashboard** | Session overview, quick-start demos, before/after at a glance |
| **Audio Studio** | Everything you *do* to audio: get it, filter it, clean it up |
| **Audio Analysis** | Everything you *look at*: frequency content and speech features |
| **Image Restore** | Blurry/noisy/striped image → clear, with PSNR and SSIM; 2-D spectrum; edge detection |
| **Video Restore** | Clean up a grainy clip frame by frame, detect edges, and export it |
| **Report & Export** | CSV, PDF, WAV and PNG output |
| **About Us** | The team, and what the project covers |

**Audio Studio** holds the three audio toolsets behind one chooser — Get audio, Filter,
Clean up — sharing a single set of before/after plots. They used to be three separate tabs,
which hid the fact that they all act on the same signal: you had to take it on trust that
the filter you designed on one tab had changed the waveform on another. Now every action
shows up on the same plots, immediately, wherever it came from.

### The design idea: one decision, not twenty knobs

The engine in `dsp/` exposes every parameter of every algorithm — thirteen denoising methods,
three FIR design methods, nine window functions, four deconvolution algorithms. That is right
for a DSP engine and wrong for the person using it.

`dsp/presets.py` sits on top and offers three things instead: *what is wrong with this
signal*, *fix it*, and *how much*. A single 0–100% dial maps onto whichever parameters the
chosen method actually needs. So **Clean Up** is one chooser and one slider, and
**Image Restore** is one Auto button and four dials — while everything underneath is still
there, still tested, and still available to talk about during the defence.

The choosers name the **problem**, not the algorithm: "Mains hum", not "cascaded IIR notch
filter at 50/60 Hz and harmonics".

---

## What it does

### Audio Studio · Get audio
Six test signals (a synthetic voice with real pauses, pure tone, harmonic tone, chirp, square
wave, ECG), plus loading any audio file or recording from the microphone.

Then the two halves of A/D conversion, both measurable:
- **Quantisation** from 1 to 16 bits, with the measured SNR compared against the
  6.02·B + 1.76 dB rule.
- **Decimation with the anti-alias filter switchable off**, so aliasing can be seen and heard
  rather than just described — a 3 kHz tone decimated by 4 reappears at exactly 1 kHz.

### Audio Studio · Filter
FIR or IIR, four shapes (low-pass, high-pass, band-pass, band-stop), one cutoff and one
"sharpness" dial that maps to taps for an FIR and order for an IIR. Five IIR families
(Butterworth, Chebyshev I/II, Elliptic, Bessel), designed as second-order sections for
numerical stability.

The filter's shape is drawn straight onto the signal's own spectrum, so what it keeps and
what it removes is visible on the same axes as the audio.

**A filter is re-designed for the signal's sample rate every time it is applied.** A digital
filter's coefficients are tied to the rate they were made for: reusing an 8 kHz design on a
16 kHz signal doubles every cutoff, and on a 44.1 kHz signal a 2 kHz low-pass would actually
cut at 11 kHz — silently. Three checks in `validate.py` guard against that returning.

### Audio Studio · Clean up
Five named problems plus Auto, which diagnoses the signal itself:

| Problem | Method underneath |
|---|---|
| Background noise | Wiener filter with a decision-directed SNR estimate |
| Hiss / static | Wavelet shrinkage (BayesShrink) |
| Mains hum | Cascaded notch filters at 50/60 Hz and harmonics |
| Clicks & pops | Median replacement of outlier samples only |
| Smooth the waveform | Savitzky–Golay |

**Auto** detects hum by combining a narrowband spectral test with a *persistence* test —
hum is present in every frame, while a voice's pitch comes and goes with the pauses. Without
that, a 120 Hz voice is flagged as badly hummed, because its harmonics land exactly on
multiples of 60 Hz.

**Compare every method** runs all thirteen denoisers in the engine on the same noisy input and
ranks them by measured SNR recovery — turning "which denoiser is best here" into a table.

**Cleaning a real recording.** When noise is added in-app, the exact noise sample is known and
handed to the filter. A recording has no such sample, so the noise spectrum is estimated from
the signal itself (minimum statistics: a low percentile of each frequency bin over time). Two
corrections made that estimate trustworthy, tested on real narrated speech and on synthetic
speech, in white, pink and babble noise at 0–10 dB SNR:
- **Digital silence is ignored.** A microphone recording opens with a few hundred ms of exact
  zeros while the device starts; those frames dragged the estimate towards zero, so the filter
  thought there was no noise to remove. A 0.4 s lead-in cost 1 dB of cleanup; now it costs none.
- **Bias correction ×1.5.** Noise magnitudes are Rayleigh-distributed, and the 15th percentile
  of a Rayleigh is only 0.455× its mean, so the raw estimate read the noise ~7 dB low. Full
  correction (×2.2) over-subtracts where speech lifts the low tail; ×1.5 was the measured
  optimum. Together the average SNR gain went from **+2.7 dB to +5.2 dB**, with every one of
  54 test cases still improving.

**Measuring a recording with no clean copy.** SNR needs a reference, so for a recording the
panel reports the two things that can be measured from the signal alone: the **background
level** (from the pauses) and the **voice level** (from the loud frames, with the noise share
subtracted), before and after. They are deliberately not combined into an "SNR": a denoiser
silences the pauses far more than it improves the noise *under* the speech, so a pause-based
SNR overstated a real +5 dB gain as +15 dB.

THD is shown only when the signal is genuinely tonal (at least half its energy on one
fundamental and its harmonics). On speech the "fundamental" is just the loudest bin and the
percentage is meaningless, so it reads n/a.

**Before and after are always distinguishable.** The waveform is drawn as two stacked panels on
one amplitude scale, and the original spectrum is dashed on top of the processed one. Overlaid
on one axis, a cleaned signal sat exactly on top of the original and the comparison showed a
single red shape.

### Image Restore — blurry to clear
- **Auto enhance** measures the image and fixes only what is wrong with it.
- Manual dials: remove noise, remove blur, sharpen, contrast.
- **Damage it** blurs, adds noise and (optionally) a stripe pattern on purpose *and records
  the blur kernel*, which turns the repair into a true Wiener deconvolution and makes the
  PSNR/SSIM scores real.
- **2-D frequency spectrum** of the image before and after, on one shared colour scale.
- **Stripe removal by parametric notch.** A sinusoidal pattern is hopeless to remove spatially
  but is just two bright points in the 2-D spectrum. Peaks are found by *prominence* (height
  above a 15×15 local mean of the log spectrum) and circled on the spectrum. Genuine content
  peaks at 23 dB at most; the weakest added pattern at 34 dB; the 30 dB threshold sits between,
  so Auto removes interference but leaves real texture alone.

  A fixed band-stop notch left stripes along the borders, because the stripe frequency rarely
  lands on a DFT bin (0.12 cycles/px × 420 px = bin 50.4) and its energy leaks into the
  neighbouring bins. So each peak's frequency is refined to a fraction of a bin (maximising the
  projection onto e^{-j2π(f_y·y + f_x·x)}, using separability to keep it fast), and that exact
  sinusoid is fitted by least squares and subtracted. Measured: **+25.7 dB PSNR**, against
  +15.5 dB for the fixed notch.
- **The denoiser is chosen per image, by SURE.** No single method wins everywhere. Tested on
  9 real photographs (the Kodak set) as well as the synthetic images, the bilateral filter was
  best on flat-shaded pictures but flattened the fine texture of real photos so badly it scored
  *below doing nothing* (29.2 dB against 30.5); wavelet shrinkage was the reverse. Stein's
  Unbiased Risk Estimate measures each candidate's mean squared error from the noisy image
  alone (no clean copy needed), and Auto keeps the lower one: **32.75 dB on the photos, against
  32.76 for an oracle that knows the original**. The vote runs on a 256×256 crop, so it costs
  well under a second.
- **A noise estimate that texture can't fool.** The Immerkaer estimator reads fine texture
  (foliage, roof tiles) as noise: a clean, detailed landscape measured σ = 8.8, so Auto
  "denoised" photos with nothing wrong with them. Measuring only over the flattest 30% of
  pixels (Tai & Yang, 2008) cut the mean error from 10.7% to 4.7%. On the sample photo, a
  noise-only repair that used to make it *worse* (30.3 → 26.9 dB) now improves it (→ 31.3 dB).
- **Deblurring with K set from the noise.** Wiener's K is the noise-to-signal ratio, so Auto
  sets K = 0.0005 + 3×10⁻⁵·σ² from the measured noise, and follows the deconvolution with a
  light edge-preserving pass that removes the noise it amplified. A sweep over 24 blur/noise
  combinations put this at +1.6 dB PSNR on average over the old fixed K = 0.02, and +2.5 to
  +3.7 dB on noise-free blur, where the old setting left the image visibly soft.
- **Edge detection** on both the before and after images, with seven detectors (Canny, Sobel,
  Prewitt, Scharr, Roberts cross, Laplacian, Laplacian of Gaussian), one sensitivity dial, an
  optional Gaussian pre-smooth, and an edge-map or overlay view. Each detector's kernel is
  explained on screen. The edges are **scored against the original's edges** (precision,
  recall, F1 with one pixel of tolerance): on the damaged demo scene Canny's F1 goes from
  14% to 93% after Auto enhance — noise had been producing false edges (low precision), blur
  had been erasing real ones (low recall).

Measured recovery on the demo scene: **+1.9 to +9.1 dB PSNR**, with SSIM rising from 0.44 to
0.93 on a noisy image.

Three measurement details that mattered:
- The Laplacian and LoG detectors mark **zero-crossings** of the second derivative, not its
  magnitude. A second derivative swings + then − across a step, so thresholding its magnitude
  draws every outline twice — the validation suite caught exactly that (83% precision on a
  plain square).
- The usual blur metric (variance of the Laplacian) is fooled by noise — adding noise made a
  blurrier image score ten times *sharper*. Pre-smoothing and normalising by image variance
  cuts that to 1.25×.
- The noise estimate runs per colour channel. Measuring on luminance reads low by exactly
  `√(0.299² + 0.587² + 0.114²) = 0.669`, because averaging three independent noise channels
  partly cancels the noise.

### Video Restore
Remove noise, remove blur, sharpen and contrast, plus the one tool a video has that a still
image does not: **temporal filtering**. Noise is independent between frames while the scene
is not, so averaging over time removes noise at no cost in spatial detail.

A plain average of the last few frames smears anything that moves. This uses a
**motion-adaptive recursive filter** instead: out[n] = out[n−1] + a·(frame[n] − out[n−1]),
with `a` chosen per pixel. Where the frame differs from the running estimate by less than the
noise could explain (0.6σ), the pixel is static and keeps averaging (a ≈ 0.12–0.5). Above 1.6σ
something moved, a = 1, and the new frame passes straight through. Measured on the test clip
against its clean frames: 37.6 dB overall against 34.5 dB for a plain 3-frame average, and
**33.9 dB against 20.9 dB on the moving ball**, which the plain average smeared worse than
the noise itself.

The test clip is the demo city scene with a ball bouncing across it, so the static/moving
contrast is easy to see. Frame preview, batch export with progress and a working cancel.

**Edge detection** uses the same seven detectors and runs on the *cleaned* frame — a
derivative is a high-pass filter, so on raw grain it finds the grain. The output switch
(Cleaned / Edge map / Overlay) applies to the preview and to the exported clip.

Two graphs: the previewed frame's histogram before and after, and the clip's **motion
energy** — mean |frame − previous frame| over time, which reduces the whole video to a 1-D
signal with the previewed frame marked.

### Report & Export
CSV of every metric (including edge precision/recall/F1 and the previewed video frame's
noise, sharpness and edge figures), a multi-page PDF with a figure page for each signal type —
audio waveform/spectrum/spectrograms; image before/after, edge maps and 2-D spectra; video
frame original/processed, edges and histograms — processed audio as WAV, before/after images
as PNG.

---

## Validation

`validate.py` checks the processing against ground truth rather than merely checking that it
runs — **157 checks**, including:

- Generators produce the frequency they claim; noise lands at the requested SNR within 0.1 dB
- Quantisation SNR matches 6.02·B + 1.76 dB; aliasing folds to exactly the predicted frequency
- All five IIR families stable across all four band types; linear-phase FIR group delay flat
  to 0.000 samples
- Every denoiser improves the noise it is designed for (hum notch +13 dB, de-click +17 dB)
- Auto correctly diagnoses five different noise types, and does not mistake a voice for hum
- The noise estimate matches the noise actually added to within 0.3 grey levels
- Auto enhance improves both PSNR *and* SSIM on four kinds of image damage, and leaves an
  undamaged image alone
- Every edge detector traces a known square outline with >90% precision and recall; Sobel
  peaks exactly on a step; pre-smoothing removes noise-induced false edges; restoration raises
  edge F1 against the original
- Stripe interference is found as exactly one conjugate pair of spectral peaks, real texture
  is not, and the notch recovers more than 6 dB
- On a photo-like image (smooth sky, 1/f-textured ground, hard edges) the noise estimate is
  within 10% at three noise levels, SURE picks the truly better denoiser each time, and Auto
  improves both PSNR and SSIM
- A known blur with no noise is deconvolved almost exactly (+4 dB or more); the video
  temporal filter cleans the static scene by more than 5 dB and beats a plain frame average
  on the moving object by more than 5 dB

---

## Project layout

```
.
├── main.py                 Entry point, dependency check, logging
├── validate.py             157 checks against ground truth
├── requirements.txt
├── DEFENSE.md              Rubric map, concepts, formulas and likely questions
├── GUIDE.md                Plain-words tour of every feature, for the whole group
├── dsp/                    Pure DSP engine — no UI, fully testable
│   ├── presets.py            The simple layer: diagnose, fix, one strength dial;
│   │                         edge detection, stripe-pattern notch
│   ├── signals.py            Generation, noise injection, quantisation, resampling
│   ├── filters.py            FIR/IIR design, application, characterisation
│   ├── denoise.py            13 noise-reduction and restoration methods
│   ├── analysis.py           DFT, PSD, STFT, cepstrum, MFCC, pitch
│   ├── metrics.py            SNR, segmental SNR, PSNR, SSIM, THD
│   ├── imaging.py            2-D filtering, frequency domain, deconvolution
│   ├── video.py              Frame streaming, temporal filters, export
│   └── audio_io.py           File I/O, playback, recording
├── app/                    User interface
│   ├── main_window.py        Shell, navigation, threading
│   ├── theme.py              Palette, fonts, widget factories
│   ├── widgets.py            Metric cards, plot cards, image cards, controls
│   ├── state.py              Shared session
│   ├── compat.py             One upstream bug workaround (see below)
│   └── pages/                One module per page
├── exports/                Default output folder
└── logs/                   spectra.log
```

### Engineering notes

**The DSP engine knows nothing about the UI.** Everything in `dsp/` takes and returns numpy
arrays, which is why `validate.py` can test it directly and why the same functions serve the
interactive pages, the batch video export and the report.

**Three-signal model.** The session holds the `original` (as loaded), the `reference` (clean
ground truth, when one genuinely exists) and the `current` working signal. When noise is added
inside the application the clean copy is kept, so the improvement is a real measurement. When
a file is loaded from disk there is no ground truth — reference-based metrics report `n/a`
rather than inventing a baseline.

**Threading.** Heavy processing runs on worker threads that never touch a Tk widget; they post
results to a queue the main loop drains every 120 ms. Video export publishes progress to a
plain attribute the main thread polls, because Tk's `after()` is not thread-safe.

**Lazy pages.** Each page is built the first time it is opened, and only the *visible* page
redraws when the session changes. Building all nine up front, and redrawing all nine on every
change, cost about twenty seconds of frozen UI.

**One upstream patch.** `app/compat.py` removes an `update_idletasks()` call from
CustomTkinter's scrollbar redraw. Without it that call re-enters the event loop from inside a
redraw, matplotlib's pending `idle_draw` runs there, the layout changes, the scrollbar is told
to update again, and the two recurse until the process locks up — `python main.py` hangs
before the window appears.

---

## Demonstration script

1. **About Us.** Opens on the team and a one-paragraph statement of what the project is.
2. **Dashboard → Noisy speech.** Point out that a clean copy exists, so the numbers are real.
3. **Audio Studio → Get audio.** Generate a pure tone at 3 kHz; decimate by 4 with the filter
   off — it reappears at 1 kHz. Turn the filter on and repeat. Then quantise to 4 bits and
   compare the measured SNR against 6.02·B + 1.76.
4. **Audio Studio → Filter.** Design an FIR band-pass; the filter's shape is drawn on the
   signal's own spectrum, and the waveform below changes as soon as it is applied. Switch to
   IIR Elliptic at the same sharpness: far fewer coefficients for a steeper cut, but the
   phase is no longer linear.
5. **Audio Studio → Clean up.** Add 5 dB of noise, press Clean it up, then **Compare all**.
   Discuss why the spectral methods win on broadband noise and why the hum notch loses here —
   the negative number is the correct answer, not a bug.
6. **Audio Analysis → Frequency.** Drag the frame-length slider: short frames sharpen time,
   long frames sharpen frequency, and you cannot have both.
7. **Audio Analysis → Speech.** Pitch contour tracking F0, and the MFCC matrix a recogniser
   would consume.
8. **Image Restore.** Demo scene → set Stripe pattern to 50% → Damage it → Auto enhance.
   Read out the PSNR and SSIM recovery, and explain why the deblur is a true deconvolution
   (the kernel was recorded). Scroll to the 2-D spectrum: the stripes are the two circled
   points, and the notch removed them.
9. **Image Restore → Find the edges.** Canny on the damaged image catches the stripes and
   grain (low precision); on the repaired one it traces the windows, sun and bars (F1 ≈ 93%).
   The building outlines are faint by design: grey building and blue sky are almost the same
   *brightness*, and edge detectors work on luminance — a good point to make. Untick
   Smooth first to show why a derivative needs a low-pass in front of it, then switch to
   Laplacian to show second-derivative zero-crossings.
10. **Video Restore.** Make the test clip and press Auto clean: the city comes out clean
    while the bouncing ball stays sharp — the temporal filter averages only pixels that are
    not moving (explain the 0.6σ/1.6σ motion test). Switch the
    output to Overlay to run edge detection on the cleaned frames; the motion-energy graph
    shows the whole clip as a 1-D signal.
11. **Report & Export.** Export the PDF.

---

## Requirements

Python 3.10+ with: numpy, scipy, matplotlib, customtkinter, Pillow, opencv-python, sounddevice,
soundfile, PyWavelets, reportlab.

Audio playback and recording need a working sound device; everything else runs without one,
and the sidebar reports which is the case.
