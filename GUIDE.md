# SPECTRA — Simple Feature Guide

A plain-words tour of what the app does and how to use it. No maths here. For the
formulas and the panel's likely questions, see `DEFENSE.md`.

**To start the app:** open a terminal in this folder and run `python main.py`.
**To prove it works:** run `python validate.py`. It should say **157/157 checks passed**.

---

## The big idea

Every page does the same three things:

1. **Get something damaged**: a noisy sound, a blurry photo, a grainy video.
2. **Fix it**, usually with one **Auto** button, or with a few simple sliders.
3. **Show how much better it got**, with before/after pictures, graphs and numbers.

The sidebar on the left lists the pages. Here is each one.

---

## 1. Dashboard — the home screen

A quick overview of what you're working on.

- **Quick start buttons** load a ready-made problem sound in one click:
  - *Noisy speech*: a voice with hiss behind it.
  - *Buzzing hum*: a voice with electrical buzz.
  - *Crackly clicks*: a voice with pops, like an old record.
  - *Frequency sweep*: a tone that slides from low to high.
- Shows the sound **before (top)** and **after (bottom)**, plus its frequencies.
- The **Activity log** lists everything you've done.

---

## 2. Audio Studio — work on sound

Three tabs at the top. They all work on the **same sound**, and every change shows up in
the graphs below right away.

### Get audio tab — bring in a sound
- **Load audio…** opens a sound file. **● Record** records from your microphone.
  **Save…** saves the current sound.
- **Test signal** makes a sound for you (a voice, a pure tone, a musical note, a sweep, a
  square wave or a heartbeat). Pick one, set **Pitch** and **Length**, then press **Generate**.
  **▶** plays it; **■** stops it.
- **Quantise** shows what happens when you store sound with fewer bits. Set **Bits per
  sample** low (like 4) and it sounds rough and grainy. The app shows how much quality was lost.
- **Decimate** throws away samples to make the file smaller. Untick **Filter first** and
  high sounds turn into *wrong*, lower sounds (this is called **aliasing**). Tick it, and the
  problem is prevented.

### Filter tab — keep some frequencies, remove others
- **FIR** or **IIR**: two kinds of filter. FIR is safer and keeps the sound's shape. IIR is
  sharper and cheaper, but delays some frequencies a little.
- **Shape**:
  - *Low-pass* keeps the low notes (removes hiss).
  - *High-pass* keeps the high notes (removes rumble).
  - *Band-pass* keeps a middle range.
  - *Band-stop* removes a middle range.
- **Cutoff** is where the filter starts cutting. **Sharpness** is how steep the cut is.
- **Show response** draws the filter's shape as a dashed line on the frequency graph.
  **Apply filter** uses it on the sound.

### Clean up tab — remove noise
- **Problem**: pick what's wrong with the sound, or leave it on **Auto** and the app works
  it out by itself.
  - *Background noise*: steady hiss or room noise.
  - *Hiss / static*: fine high-pitched hiss.
  - *Mains hum*: low electrical buzz.
  - *Clicks & pops*: short sharp noises.
  - *Smooth the waveform*: general smoothing.
- **How hard to work**: how strongly to clean.
- **Clean it up** does it. **Compare all** tries all 13 cleaning methods and ranks them.
- **Need something to clean?** **Add noise** puts noise on a clean sound on purpose. Because
  the app keeps the clean copy, it can tell you *exactly* how much the cleaning helped (SNR,
  in dB). **Back to clean** undoes the noise.
- On your **own recording** (with no clean copy), the panel on the right shows how much
  quieter the **background** got and whether the **voice level** stayed the same.

---

## 3. Audio Analysis — look inside a sound

Two views at the top.

- **Frequency**:
  - The **spectrum** shows which pitches are in the sound.
  - The **spectrogram** is a picture of the sound over time: time left to right, pitch bottom
    to top, loudness as colour.
  - Drag **Frame length** to see the trade-off: short frames show *when* things happen
    clearly, and long frames show *which pitch* clearly. You can't have both at once.
- **Speech**:
  - **Energy**: how loud the voice is over time.
  - **Pitch**: how high the voice is over time.
  - **MFCC**: the features speech-recognition systems use to understand a voice.
  - **Apply pre-emphasis** boosts the high frequencies, which makes speech clearer.

---

## 4. Image Restore — fix a photo

Follow the numbered panels.

1. **Pick an image**: **Load image…** or **Demo scene** (a ready-made city picture).
2. **Damage it (optional)**: add **Blur**, **Noise** and a **Stripe pattern** on purpose,
   then press **Damage it**. The app remembers exactly what it did, so it can measure the
   repair properly. **Undo damage** goes back.
3. **Make it clear**: press **✦ Auto enhance** and the app fixes only what's wrong. Or use the
   sliders yourself:
   - **Remove noise**, **Remove blur**, **Sharpen**, **Contrast**.
   - Tick **Noise is speckly** for dot-like noise.
   - Tick **Remove stripe pattern** to remove stripes.
   - Then press **Apply**.
4. **Find the edges**: draws the outlines of everything in the picture.
   - **Detector**: 7 methods (Canny, Sobel, Prewitt, Scharr, Roberts, Laplacian, LoG). The
     text under it explains each one.
   - **Sensitivity**: more or fewer edges. **Smooth first**: ignore the grain.
   - **Edge map** shows only the outlines. **Overlay** draws them in red on the picture.
   - **Save edge map…** saves it.

**What you see:**
- Before and after pictures, side by side.
- **How much better is it?** **PSNR** and **SSIM** scores; higher means closer to the
  original.
- **Brightness distribution**: a graph of how bright the pixels are.
- **2-D frequency spectrum**: stripes show up here as bright dots, and they disappear once
  removed. Circles mark them.
- **Edge accuracy**: how well the edges match the original picture's edges.

---

## 5. Video Restore — fix a video

1. **Pick a clip**: **Load video…** or **Make test clip** (a city with a bouncing ball and
   lots of grain).
   - **Preview frame** picks which moment to look at.
   - **Export at most** limits how many frames get saved.
2. **Clean it up**: **✦ Auto clean** sets the sliders for you, or set **Remove noise**,
   **Remove blur**, **Sharpen** and **Contrast** yourself. **Preview** shows the result on one
   frame. **Export video…** saves the whole cleaned clip (with a progress bar and **Cancel**).
   - The trick a video has over a photo: noise changes every frame but the scene doesn't, so
     the app **averages each still pixel over several frames**. Moving things (the ball) are
     left alone, so they don't blur.
3. **Find the edges**: switch **Cleaned / Edge map / Overlay** to see the outlines. This also
   applies to the exported video.

**What you see:**
- The original and cleaned frame side by side.
- Noise and sharpness numbers.
- A **brightness graph**.
- A **motion graph** (how much the picture changes over time).

---

## 6. Report & Export — save your results

- **Export metrics (CSV)**: all the numbers, as a spreadsheet.
- **Export report (PDF)**: a full report with every number, every step you took, and
  before/after pictures for audio, image and video.
- **Export processed audio (WAV)**: the cleaned sound.
- **Export original + processed images**: both pictures as PNG files.
- **Export figure (PNG)**: the report graph as an image.
- **Open the exports folder**: shows where the files went.

---

## 7. About Us

The team, what the project covers, and the list of DSP techniques used.

---

## Rubric checklist, in simple words

| # | What the teacher grades | Where it is in the app |
|---|---|---|
| 1 | **Everything works** (25) | Every page and button. `validate.py` passes 157/157 |
| 2 | **Correct DSP** (20) | Filters, noise removal, deblurring, edge detection; all tested against known answers |
| 3 | **Audio, speech, image, video** (15) | Audio Studio (audio), Audio Analysis → Speech (speech), Image Restore (image), Video Restore (video) |
| 4 | **Noise reduction, filtering, restoration, enhancement** (15) | Noise reduction: Clean up, Remove noise. Filtering: Filter tab. Restoration: Remove blur, Clicks & pops. Enhancement: Sharpen, Contrast, edges |
| 5 | **Before vs after, clearly shown** (10) | Every page shows before/after side by side with graphs, plus the PDF report |
| 6 | **Neat, easy-to-use app** (5) | One job per page, Auto buttons, sliders named after the problem |
| 7 | **Demo and explaining it** (10) | The demo script in `README.md`, and the questions and answers in `DEFENSE.md` |

### The five general instructions

1. **All functions working**: yes, every page and button tested.
2. **Python with DSP techniques**: yes, all of it is Python (numpy, scipy, OpenCV).
3. **Audio, speech, image and/or video**: all four.
4. **Noise reduction, restoration, filtering, analysis**: all four, for every signal type.
5. **Every member can explain it**: use this guide for *what* it does and `DEFENSE.md` for
   *how* it works.

---

## Sample files for testing

The `samples` folder has one real file of each kind, all free to use:

- `samples/audio/harvard_sentences_8k.wav`: a man reading standard test sentences, with
  pauses. Load it in **Audio Studio**, press **Add noise**, then **Clean it up**.
- `samples/image/kodim08_buildings.png`: a street of old buildings (Kodak test photo). Load
  it in **Image Restore**. Gentle damage (Blur 30%, Noise 15%, Stripe pattern 50%) shows the
  repair best on a real photo; heavy damage destroys fine detail that nothing can bring back.
- `samples/video/big_buck_bunny_360p_10s.mp4`: a 10-second animated clip (Blender
  Foundation, free licence). It is already clean, so it's best for **edge detection**; use
  **Make test clip** to show noise removal.

---

## A 5-minute demo, if you're short on time

1. **Dashboard → Noisy speech**, then **Audio Studio → Clean up → Clean it up**. Read out
   the dB improvement.
2. **Image Restore → Demo scene → Stripe pattern 50% → Damage it → ✦ Auto enhance**. Point
   at the PSNR/SSIM going up, the stripes' dots vanishing in the spectrum, and the edges
   getting cleaner.
3. **Video Restore → Make test clip → ✦ Auto clean**. The city gets clean, the ball stays
   sharp.
4. **Report & Export → Export report (PDF)**. Show the before/after pages.
