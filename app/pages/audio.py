"""
Audio Studio - everything you *do* to an audio signal, in one place.

Getting a signal, filtering it and cleaning it up used to be three separate tabs, which hid
the fact that they all act on the same signal: you had to take it on trust that the filter
you designed on one tab had changed the waveform on another.  Here the three toolsets share
one set of before/after plots, so every action is visible where it happens.

The three panels swap in the same slot rather than sitting side by side, which keeps the
number of controls on screen at any moment about the same as before.

One subtlety worth knowing: a digital filter's coefficients are tied to the sample rate they
were designed for.  Designing at 8 kHz and applying to a 16 kHz signal doubles every cutoff
frequency, silently.  `_apply_filter` re-designs against the signal's actual rate whenever
they differ rather than letting that happen.
"""
from __future__ import annotations

import os
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

from dsp import (analysis as A, audio_io, denoise as D, filters as F, metrics as M,
                 presets as P, signals as S)
from .. import theme as T
from ..widgets import PageStack, PlotCard, SliderRow, metric_lines
from .base import Page

TOOLS = ("Get audio", "Filter", "Clean up")

TEST_SIGNALS = {
    "Speech (synthetic voice)": "A voice with real pauses — the realistic case for noise "
                                "reduction, and the only one where a noise estimator can "
                                "learn from the silences.",
    "Pure tone": "A single sine. Everything a filter does to it is checkable against theory.",
    "Musical tone (harmonics)": "A fundamental plus harmonics — what a note actually is.",
    "Frequency sweep (chirp)": "Rises through the whole band, so a filter's shape becomes "
                               "audible and the spectrogram shows a diagonal line.",
    "Square wave": "All odd harmonics, so it shows filtering and aliasing dramatically.",
    "Heartbeat (ECG)": "A biomedical signal — sharp peaks that smoothing must not flatten.",
}

_WAVEFORM = {"Speech (synthetic voice)": "Speech-like formants", "Pure tone": "Sine",
             "Musical tone (harmonics)": "Multi-tone",
             "Frequency sweep (chirp)": "Chirp (linear)", "Square wave": "Square",
             "Heartbeat (ECG)": "ECG-like"}

SHAPES = {"Low-pass — keep the low notes": "Low-pass",
          "High-pass — keep the high notes": "High-pass",
          "Band-pass — keep a middle range": "Band-pass",
          "Band-stop — remove a middle range": "Band-stop"}

SHAPE_HELP = {
    "Low-pass": "Passes everything below the cutoff. Removes hiss, and it is the anti-alias "
                "filter used before decimating.",
    "High-pass": "Passes everything above the cutoff. Removes rumble, mains hum and slow "
                 "baseline drift.",
    "Band-pass": "Keeps only what lies between the two cutoffs — 300 to 3400 Hz is the "
                 "telephone speech band.",
    "Band-stop": "Removes everything between the two cutoffs. A very narrow one is a notch, "
                 "which is how mains hum is removed.",
}

NOISE_CHOICES = ("White Gaussian (AWGN)", "Mains hum (60 Hz)", "Impulse / click",
                 "Pink (1/f)")


class AudioStudioPage(Page):
    title = "Audio Studio"
    subtitle = "Get a signal, filter it, clean it up — all acting on the same audio"
    icon = ""
    nav_name = "Audio Studio"

    def build(self):
        self.player = audio_io.Player()
        self.recorder = None
        self.spec = None
        page = self.scroll()

        chooser = self.row(page, pady=(0, 8))
        self.tool = T.segmented(chooser, TOOLS, self._on_tool)
        self.tool.set(TOOLS[0])
        self.tool.pack(anchor="w")

        top = self.row(page)
        holder = ctk.CTkFrame(top, fg_color="transparent")
        holder.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.tools = PageStack(holder)
        self.tools.pack(fill="both", expand=True)
        self._build_source(self.tools.add("Get audio"))
        self._build_filter(self.tools.add("Filter"))
        self._build_clean(self.tools.add("Clean up"))

        self.result = T.card(top)
        self.result.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(self.result, "The signal right now",
                        "Updated after every action, whichever tool did it.")
        self.result_box = T.table_box(self.result, height=300)
        self.result_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        plots = self.row(page)
        self.wave = PlotCard(plots, "Waveform · before and after",
                             "The signal as it started above, the signal now below, on the "
                             "same amplitude scale.", height=270, nrows=2)
        self.wave.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.spectrum = PlotCard(plots, "Frequency content · before and after",
                                 "What every action added or removed, in frequency. The "
                                 "gap between the curves is what was taken out.",
                                 height=270)
        self.spectrum.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.sgram = PlotCard(page, "Spectrograms · before and after",
                              "Time runs left to right, frequency bottom to top. Noise is a "
                              "haze; a filter cuts a horizontal band out of it.",
                              height=235, ncols=2)
        self.sgram.pack(fill="both", expand=True, pady=(0, 10))

        self.tools.set("Get audio")
        self._on_signal_choice(self.kind.get())
        self._on_shape(self.shape.get())
        self._on_kind(self.filter_kind.get())
        self._on_preset(self.preset.get())
        self.refresh()

    def _on_tool(self, name):
        """Switch tools, redrawing only what depends on the tool.

        This used to call `refresh()`, which redrew every plot - over a second on a 6 s
        recording - before Tk could paint the panel it had just swapped in, so the panel
        appeared with its controls missing.  The waveform and spectrograms do not change with
        the tool; only the right-hand panel does, and the spectrum's filter overlay when
        moving onto or off the Filter tool.  That redraw is also deferred so the new panel is
        painted first.
        """
        previous = self.tools.get()
        self.tools.set(name)
        if not self.session.has_signal:
            return
        overlay_changed = (self.spec is not None
                           and (previous == "Filter") != (name == "Filter"))
        if getattr(self, "_tool_job", None) is not None:
            self.after_cancel(self._tool_job)
            overlay_changed = True        # a cancelled redraw may have owed the spectrum one

        def redraw():
            self._tool_job = None
            self._draw_result()
            if overlay_changed:
                self._draw_spectrum()

        self._tool_job = self.after(30, redraw)

    # ================================================================= 1. source
    def _build_source(self, parent):
        box, body = self.controls(parent, "Get audio",
                                  "Load real audio, record your own voice, or make a test "
                                  "signal whose answer you already know.")
        box.pack(fill="both", expand=True)

        bar = self.button_bar(body, pady=(0, 6))
        T.accent_button(bar, "Load audio…", self._load_file, width=120).pack(side="left")
        self.rec_btn = T.ghost_button(bar, "● Record", self._toggle_record, width=104)
        self.rec_btn.pack(side="left", padx=6)
        T.ghost_button(bar, "Save…", self._save_file, width=80).pack(side="left")
        self.rec_status = ctk.CTkLabel(body, text="", text_color=T.MUTED, font=T.font(10.5))
        self.rec_status.pack(anchor="w")

        T.divider(body, pady=(8, 8), padx=0)

        self.kind = T.option_menu(body, tuple(TEST_SIGNALS), 240,
                                  command=self._on_signal_choice)
        self.kind.set("Speech (synthetic voice)")
        self._row(body, "Test signal", self.kind)
        self.kind_help = T.hint(body, "")
        self.kind_help.pack(anchor="w", fill="x", pady=(6, 4))

        self.freq = SliderRow(body, "Pitch", 60, 2000, 220, "{:.0f} Hz", width=160)
        self.freq.pack(fill="x", pady=3)
        self.dur = SliderRow(body, "Length", 0.5, 8.0, 2.5, "{:.1f} s", width=160)
        self.dur.pack(fill="x", pady=3)

        bar2 = self.button_bar(body)
        T.accent_button(bar2, "Generate", self._generate, width=104).pack(side="left")
        T.ghost_button(bar2, "▶ Play", self._play, width=76).pack(side="left", padx=5)
        T.ghost_button(bar2, "■", self._stop, width=38).pack(side="left")

        T.divider(body, pady=(10, 8), padx=0)
        ctk.CTkLabel(body, text="SAMPLING AND QUANTISATION", text_color=T.FAINT,
                     font=T.font(9.5, "bold")).pack(anchor="w")
        self.bits = SliderRow(body, "Bits per sample", 1, 16, 8, "{:.0f} bit", steps=15,
                              width=160)
        self.bits.pack(fill="x", pady=3)
        self.decim = SliderRow(body, "Keep 1 sample in", 1, 12, 4, "{:.0f}", steps=11,
                               width=160)
        self.decim.pack(fill="x", pady=3)
        self.anti_alias = T.checkbox(body, "Filter first (the correct way)")
        self.anti_alias.select()
        self.anti_alias.pack(anchor="w", pady=(4, 2))
        T.hint(body, "Untick it to see aliasing: anything above the new Nyquist frequency "
                     "folds back into the band as a false tone.").pack(anchor="w", fill="x")
        bar3 = self.button_bar(body)
        T.accent_button(bar3, "Quantise", self._quantise, width=104).pack(side="left")
        T.ghost_button(bar3, "Decimate", self._decimate, width=104).pack(side="left", padx=6)

    # ================================================================= 2. filter
    def _build_filter(self, parent):
        box, body = self.controls(
            parent, "Filter the signal",
            "FIR is always stable and never distorts the waveform's shape but needs a high "
            "order; IIR is far cheaper for the same sharpness, at the cost of delaying some "
            "frequencies more than others.")
        box.pack(fill="both", expand=True)

        self.filter_kind = T.segmented(body, ("FIR", "IIR"), self._on_kind)
        self.filter_kind.set("FIR")
        self.filter_kind.pack(anchor="w", pady=(0, 8))

        self.shape = T.option_menu(body, tuple(SHAPES), 270, command=self._on_shape)
        self.shape.set("Low-pass — keep the low notes")
        self._row(body, "Shape", self.shape)
        self.shape_help = T.hint(body, "")
        self.shape_help.pack(anchor="w", fill="x", pady=(6, 4))

        self.cut1 = SliderRow(body, "Cutoff", 10, 3900, 1000, "{:.0f} Hz", width=165)
        self.cut1.pack(fill="x", pady=3)
        self.cut2 = SliderRow(body, "Upper cutoff", 10, 3900, 2500, "{:.0f} Hz", width=165)
        self.cut2.pack(fill="x", pady=3)
        self.sharpness = SliderRow(body, "Sharpness", 1, 10, 4, "{:.0f}", steps=9, width=165)
        self.sharpness.pack(fill="x", pady=3)

        self.family = T.option_menu(body, F.IIR_FAMILIES, 190)
        self.family_row = self._row(body, "IIR type", self.family)

        bar = self.button_bar(body)
        # Remembered so the IIR-type row can be re-packed *above* the buttons when it is
        # shown again; without an explicit anchor Tk would append it after them.
        self.filter_bar = bar
        T.accent_button(bar, "Apply filter", self._apply_filter, width=120).pack(side="left")
        T.ghost_button(bar, "Show response", self._show_response,
                       width=125).pack(side="left", padx=6)
        T.hint(body, "The filter is designed for this signal's sample rate every time it is "
                     "applied, so the cutoff you set is the cutoff you get.").pack(
            anchor="w", fill="x", pady=(8, 0))

    # ================================================================= 3. clean up
    def _build_clean(self, parent):
        box, body = self.controls(
            parent, "Clean up the signal",
            "Pick the problem, not the algorithm — Auto listens to the signal and decides "
            "for itself.")
        box.pack(fill="both", expand=True)

        self.preset = T.option_menu(body, P.CLEANUP_PRESETS, 250, command=self._on_preset)
        self.preset.set(P.AUTO)
        self._row(body, "Problem", self.preset)
        self.preset_help = T.hint(body, "")
        self.preset_help.pack(anchor="w", fill="x", pady=(6, 4))

        self.strength = SliderRow(body, "How hard to work", 0.0, 1.0, 0.5, "{:.0%}", width=170)
        self.strength.pack(fill="x", pady=3)

        bar = self.button_bar(body)
        T.accent_button(bar, "Clean it up", self._clean, width=116).pack(side="left")
        T.ghost_button(bar, "Compare all", self._benchmark,
                       width=110).pack(side="left", padx=6)
        T.ghost_button(bar, "▶", self._play, width=38).pack(side="left", padx=(0, 4))
        T.ghost_button(bar, "■", self._stop, width=38).pack(side="left")

        T.divider(body, pady=(12, 8), padx=0)
        ctk.CTkLabel(body, text="NEED SOMETHING TO CLEAN?", text_color=T.FAINT,
                     font=T.font(9.5, "bold")).pack(anchor="w")
        self.noise_kind = T.option_menu(body, NOISE_CHOICES, 230)
        self._row(body, "Add noise", self.noise_kind)
        self.snr = SliderRow(body, "Noise level", -5.0, 25.0, 5.0, "{:.0f} dB SNR", width=170)
        self.snr.pack(fill="x", pady=3)
        bar2 = self.button_bar(body)
        T.accent_button(bar2, "Add noise", self._add_noise, width=110).pack(side="left")
        T.ghost_button(bar2, "Back to clean", self._use_clean,
                       width=120).pack(side="left", padx=6)
        T.hint(body, "A lower SNR means more noise. The clean copy is kept, which is what "
                     "makes the improvement a measurement rather than an impression.").pack(
            anchor="w", fill="x", pady=(8, 0))

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _row(master, caption, widget, width=100):
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=caption, text_color=T.MUTED, anchor="w", width=width,
                     font=T.font(11.5)).pack(side="left", padx=(0, 8))
        widget.pack(in_=row, side="left")
        widget.lift(row)
        return row

    def _on_signal_choice(self, choice):
        self.kind_help.configure(text=TEST_SIGNALS.get(choice, ""))

    def _on_preset(self, preset):
        self.preset_help.configure(text=P.PRESET_HELP.get(preset, ""))

    def _on_kind(self, value):
        if value == "IIR":
            self.family_row.pack(fill="x", pady=3, before=self.filter_bar)
        else:
            self.family_row.pack_forget()

    def _on_shape(self, label):
        band = SHAPES[label]
        self.shape_help.configure(text=SHAPE_HELP[band])
        if band in ("Band-pass", "Band-stop"):
            self.cut2.pack(fill="x", pady=3, after=self.cut1)
        else:
            self.cut2.pack_forget()

    def on_show(self):
        self._rescale_cutoffs()
        super().on_show()

    def _rescale_cutoffs(self):
        """Keep the cutoff sliders inside the band the current signal can represent."""
        nyq = (self.session.fs / 2.0) if self.session.has_signal else 4000.0
        for row, limit, default in ((self.cut1, nyq * 0.45, nyq * 0.25),
                                    (self.cut2, nyq * 0.98, nyq * 0.6)):
            current = row.get()
            row.slider.configure(from_=max(nyq * 0.004, 1.0), to=nyq * 0.98)
            row.set(min(current, limit) if current > 1 else default)
        if self.cut1.get() >= self.cut2.get():
            self.cut1.set(max(self.cut2.get() * 0.5, nyq * 0.004))

    # ----------------------------------------------------------------- source actions
    def _generate(self):
        label = self.kind.get()
        kind = _WAVEFORM[label]
        fs = 8000.0
        freq = 72.0 if kind == "ECG-like" else self.freq.get()
        dur = self.dur.get()
        if freq >= fs / 2:
            self.status(f"Pitch must stay below {fs / 2:,.0f} Hz.", "WARN")
            return

        def done(x):
            self.session.set_signal(x, fs, label, "generated", reference=x)
            self.session.add_step("Generate", f"{label}, {dur:.1f}s at {fs:,.0f} Hz")
            self._rescale_cutoffs()
            self.status(f"Generated {label}.", "OK")

        self.run(lambda: S.generate(kind, dur, fs, freq, 0.8, freq2=min(freq * 8, 3200.0)),
                 done, busy="Generating…")

    def _load_file(self):
        path = filedialog.askopenfilename(title="Open an audio file",
                                          filetypes=audio_io.FILE_TYPES, parent=self)
        if not path:
            return

        def done(result):
            x, fs = result
            if x.size == 0:
                self.status("That file contains no audio.", "WARN")
                return
            # A file from disk has no clean version, so the reference stays empty and the
            # reference-based scores honestly read n/a instead of inventing a baseline.
            self.session.set_signal(x, fs, os.path.basename(path), "file")
            self.session.add_step("Load", os.path.basename(path))
            self._rescale_cutoffs()
            self.status(f"Loaded {os.path.basename(path)} at {fs:,.0f} Hz.", "OK")

        self.run(lambda: audio_io.load(path), done, busy="Reading the audio file…")

    def _save_file(self):
        if not self.require_signal():
            return
        path = filedialog.asksaveasfilename(title="Save the signal", defaultextension=".wav",
                                            filetypes=[("WAV", "*.wav")], parent=self)
        if not path:
            return
        cur = self.session.current
        self.run(lambda: audio_io.save(path, cur.data, cur.fs),
                 lambda _r: self.status(f"Saved {os.path.basename(path)}.", "OK"),
                 busy="Writing…")

    def _toggle_record(self):
        if self.recorder is not None and self.recorder.recording:
            x, fs = self.recorder.stop()
            self.recorder = None
            self.rec_btn.configure(text="● Record", fg_color=T.PANEL_ALT)
            self.rec_status.configure(text="", text_color=T.MUTED)
            if x.size == 0:
                self.status("Nothing was recorded.", "WARN")
                return
            self.session.set_signal(x, fs, "your recording", "recorded")
            self.session.add_step("Record", f"{x.size / fs:.2f}s at {fs:,.0f} Hz")
            self._rescale_cutoffs()
            self.status(f"Recorded {x.size / fs:.2f} seconds.", "OK")
            return
        try:
            self.recorder = audio_io.Recorder(16000.0)
            self.recorder.start()
        except audio_io.AudioError as exc:
            self.recorder = None
            self.toast_error("Recording failed", str(exc))
            return
        self.rec_btn.configure(text="■ Stop", fg_color=T.RED)
        self._tick_record()

    def _tick_record(self):
        if self.recorder is not None and self.recorder.recording:
            self.rec_status.configure(text=f"● recording {self.recorder.seconds:.1f}s",
                                      text_color=T.RED)
            self.after(200, self._tick_record)

    def _quantise(self):
        if not self.require_signal():
            return
        bits = int(self.bits.get())
        cur = self.session.current

        def done(y):
            snr = M.snr_db(cur.data, y)
            self.session.set_current(y, f"{cur.name} ({bits}-bit)")
            self.session.add_step("Quantise", f"{bits}-bit, SNR {snr:.1f} dB",
                                  {"bits": bits}, {"SNR (dB)": snr})
            self.status(f"Quantised to {bits} bits — SNR {snr:.1f} dB, against a theoretical "
                        f"best of {6.02 * bits + 1.76:.1f} dB for a full-scale sine.", "OK")

        self.run(lambda: S.quantize(cur.data, bits), done, busy="Quantising…")

    def _decimate(self):
        if not self.require_signal():
            return
        factor = int(self.decim.get())
        anti = bool(self.anti_alias.get())
        cur = self.session.current

        def done(result):
            y, new_fs = result
            self.session.set_signal(y, new_fs, f"{cur.name} ÷{factor}", "derived", reset=False)
            self.session.add_step("Decimate", f"kept 1 sample in {factor} → {new_fs:,.0f} Hz, "
                                              + ("filtered first" if anti else "no filter"))
            self._rescale_cutoffs()
            if anti:
                self.status(f"Decimated to {new_fs:,.0f} Hz with the anti-alias filter.", "OK")
            else:
                self.status(f"Decimated to {new_fs:,.0f} Hz with no filter — everything that "
                            f"was above {new_fs / 2:,.0f} Hz has folded back into the band "
                            f"as false tones.", "WARN")

        self.run(lambda: S.downsample_demo(cur.data, cur.fs, factor, anti), done,
                 busy="Decimating…")

    # ----------------------------------------------------------------- filter actions
    def _design_for(self, fs: float):
        """Build the filter specification against a given sample rate."""
        band = SHAPES[self.shape.get()]
        sharp = int(self.sharpness.get())
        if band in ("Band-pass", "Band-stop"):
            lo, hi = sorted((self.cut1.get(), self.cut2.get()))
            if hi - lo < fs * 1e-4:
                raise F.FilterError("The two cutoffs are too close together to form a band.")
            cutoff = (lo, hi)
        else:
            cutoff = self.cut1.get()
        nyq = fs / 2.0
        for c in (cutoff if isinstance(cutoff, tuple) else (cutoff,)):
            if c >= nyq:
                raise F.FilterError(
                    f"A {c:,.0f} Hz cutoff is above this signal's Nyquist limit of "
                    f"{nyq:,.0f} Hz — the signal cannot represent that frequency.")
        if self.filter_kind.get() == "FIR":
            taps = int(np.clip(sharp * 24 + 9, 9, 401)) | 1
            return F.design_fir(band, cutoff, fs, taps, "hamming", "Window method")
        return F.design_iir(self.family.get(), band, cutoff, fs,
                            int(np.clip(sharp, 1, 12)), ripple_db=1.0, atten_db=40.0)

    def _apply_filter(self):
        if not self.require_signal():
            return
        cur = self.session.current

        def work():
            # Always design against the signal's own rate.  A filter's coefficients are tied
            # to the rate they were made for, so reusing an 8 kHz design on a 16 kHz signal
            # would double every cutoff without any warning.
            spec = self._design_for(cur.fs)
            return spec, F.apply_filter(spec, cur.data, zero_phase=True)

        def done(result):
            spec, y = result
            self.spec = spec
            self.session.set_current(y, cur.name)
            self.session.add_step("Filter", spec.describe(), {"filter": spec.describe()},
                                  F.estimate_attenuation(spec))
            self.status(f"Applied {spec.describe()} to the signal.", "OK")

        self.run(work, done, busy="Filtering…",
                 on_error=lambda exc: self.status(str(exc), "WARN"))

    def _show_response(self):
        """Design without applying, so the response can be inspected first."""
        if not self.require_signal():
            return
        fs = self.session.fs

        def done(spec):
            self.spec = spec
            self.status(f"Designed {spec.describe()} — the dashed line on the spectrum plot "
                        f"is its shape.", "OK")
            self.refresh()

        self.run(lambda: self._design_for(fs), done, busy="Designing…",
                 on_error=lambda exc: self.status(str(exc), "WARN"))

    # ----------------------------------------------------------------- clean-up actions
    def _add_noise(self):
        if not self.require_signal():
            return
        kind, snr = self.noise_kind.get(), self.snr.get()
        base = self.session.reference or self.session.original
        fs = base.fs

        def work():
            y = S.add_noise(base.data, fs, kind, snr, density=0.01, seed=7)
            return y, (y - base.data)

        def done(result):
            y, noise = result
            self.session.set_signal(y, fs, f"{base.name} + {kind}", "generated",
                                    reference=base.data)
            self.session.noise_clip = noise[:int(fs * 0.5)]
            self.session.add_step("Add noise", f"{kind} at {snr:.0f} dB SNR")
            self.status(f"Added {kind} — measured SNR {M.snr_db(base.data, y):.2f} dB.", "OK")

        self.run(work, done, busy="Adding noise…")

    def _use_clean(self):
        ref = self.session.reference
        if ref is None:
            self.status("There is no clean copy of this signal.", "WARN")
            return
        self.session.set_signal(ref.data, ref.fs, ref.name, "generated", reference=ref.data)
        self.status("Restored the clean signal.", "OK")

    def _clean(self):
        if not self.require_signal():
            return
        preset, strength = self.preset.get(), self.strength.get()
        cur = self.session.current
        ref = self.session.reference_array()
        before = M.snr_db(ref, cur.data) if ref is not None else None
        clip = self.session.noise_clip

        def done(result):
            y, desc = result
            self.session.set_current(y, cur.name)
            metrics = {}
            if ref is not None:
                n = min(ref.size, y.size)
                after = M.snr_db(ref[:n], y[:n])
                metrics = {"SNR before (dB)": before, "SNR after (dB)": after,
                           "Improvement (dB)": after - before}
                self.status(f"{desc} — SNR {before:.2f} → {after:.2f} dB "
                            f"({after - before:+.2f} dB).", "OK")
            else:
                self.status(f"Cleaned up: {desc}.", "OK")
            self.session.add_step("Clean up", desc,
                                  {"preset": preset, "strength": strength}, metrics)

        self.run(lambda: P.clean_audio(cur.data, cur.fs, preset, strength, noise_clip=clip),
                 done, busy="Cleaning up…")

    def _benchmark(self):
        if not self.require_signal():
            return
        ref = self.session.reference_array()
        if ref is None:
            self.status("Comparing needs a clean copy — add noise here first.", "WARN")
            return
        # Always compare from the noisy input, never from whatever is on screen: run after a
        # cleanup it would be scoring every method on an already-clean signal.
        source = self.session.original or self.session.current
        x, fs = source.data.copy(), source.fs
        base = M.snr_db(ref, x)
        base_seg = M.segmental_snr_db(ref, x, fs)
        clip = self.session.noise_clip

        def work():
            rows = []
            for method in D.METHODS:
                try:
                    y, _info = D.enhance(method, x, fs, {"noise_clip": clip})
                    n = min(ref.size, y.size)
                    rows.append((method, M.snr_db(ref[:n], y[:n]) - base,
                                 M.segmental_snr_db(ref[:n], y[:n], fs) - base_seg))
                except Exception as exc:
                    rows.append((method, float("nan"), str(exc)[:22]))
            rows.sort(key=lambda r: (-r[1]) if np.isfinite(r[1]) else 1e9)
            return rows

        def done(rows):
            lines = [(f"Every denoiser, run on the noisy input at {base:.2f} dB SNR", "HEAD"),
                     (f"{'METHOD':<32}{'SNR gain':>10}{'frame-wise':>12}", "MUTED"),
                     ("-" * 54, "RULE")]
            for method, gain, seg in rows:
                if not np.isfinite(gain):
                    lines.append((f"{method:<32}{'failed':>10}  {seg}", "BAD"))
                    continue
                tag = "GOOD" if gain > 0.5 else ("WARN" if gain > -0.5 else "BAD")
                lines.append((f"{method:<32}{gain:>+9.2f} {seg:>+11.2f}", tag))
            lines.append(("", None))
            lines.append(("A negative score is a real answer: that method is the wrong tool "
                          "for this kind of noise.", "MUTED"))
            T.write_lines(self.result_box, lines)
            if rows and np.isfinite(rows[0][1]):
                self.status(f"Best here: {rows[0][0]} at {rows[0][1]:+.2f} dB.", "OK")

        self.run(work, done, busy="Running every denoiser…")

    # ----------------------------------------------------------------- playback
    def _play(self):
        if not self.require_signal():
            return
        cur = self.session.current
        try:
            self.player.play(cur.data, cur.fs)
            self.status(f"Playing {cur.duration:.2f} seconds…", "INFO")
        except audio_io.AudioError as exc:
            self.toast_error("Playback failed", str(exc))

    def _stop(self):
        self.player.stop()

    # ----------------------------------------------------------------- drawing
    def refresh(self):
        s = self.session
        if not s.has_signal:
            for p in (self.wave, self.spectrum, self.sgram):
                p.message("Load, record or generate a signal to begin")
            T.write_lines(self.result_box, [("No signal loaded.", "MUTED")])
            return
        self._draw_wave()
        self._draw_spectrum()
        self._draw_spectrograms()
        self._draw_result()

    def _draw_wave(self):
        """Before above, now below, on the same amplitude scale.

        Overlaid on one axis the two traces hid each other - after a clean-up the new signal
        sits almost exactly on top of the old one, so the comparison showed one red shape.
        Stacked with a shared scale, every change in amplitude is visible at a glance.
        """
        s = self.session
        cur = s.current
        self.wave.reset_axes()
        t = np.arange(cur.samples) / cur.fs
        before = s.original.data if s.original is not None else cur.data
        tb = np.arange(before.size) / (s.original.fs if s.original is not None else cur.fs)
        peak = max(float(np.max(np.abs(before))) if before.size else 0.0,
                   float(np.max(np.abs(cur.data))) if cur.samples else 0.0, 1e-6) * 1.08

        top, bottom = self.wave.axes(0), self.wave.axes(1)
        top.plot(tb, before, color=T.MUTED, lw=0.6)
        top.set_title("As it started", fontsize=8, loc="left")
        if s.reference is not None:
            r = s.reference.data[:cur.samples]
            bottom.plot(t[:r.size], r, color=T.TEAL, lw=0.6, alpha=0.7, label="clean original")
        bottom.plot(t, cur.data, color=T.ACCENT, lw=0.6, label="now")
        bottom.set_title("Now", fontsize=8, loc="left")
        if s.reference is not None:
            bottom.legend(loc="upper right", ncol=2, fontsize=7)
        span = max(tb[-1] if tb.size else 0.0, t[-1] if t.size else 0.0, 1e-3)
        for ax in (top, bottom):
            ax.set_ylim(-peak, peak)
            ax.set_xlim(0, span)
            ax.set_ylabel("Amplitude", fontsize=7)
        top.tick_params(labelbottom=False)
        bottom.set_xlabel("Time (s)")
        self.wave.draw()

    def _draw_spectrum(self):
        s = self.session
        cur = s.current
        self.spectrum.reset_axes()
        ax = self.spectrum.axes()
        f1, m1 = A.psd_welch(cur.data, cur.fs)
        ax.plot(f1, m1, color=T.ACCENT, lw=1.0, label="now")
        levels = [m1]
        if s.original is not None and s.original.data.size:
            # Drawn on top and dashed, so it stays visible where the two curves coincide and
            # the gap shows exactly where something was removed.
            f0, m0 = A.psd_welch(s.original.data, s.original.fs)
            ax.plot(f0, m0, color=T.TEXT, lw=0.8, ls=(0, (3, 2)), alpha=0.75,
                    label="as it started", zorder=3)
            levels.append(m0)
        finite = np.concatenate([lv[np.isfinite(lv)] for lv in levels])
        lo = float(np.percentile(finite, 0.5)) - 6.0 if finite.size else -120.0
        hi = float(np.max(finite)) + 6.0 if finite.size else 0.0
        # The filter's shape belongs on the Filter tool only - left on the Clean-up view it
        # describes something that is not being done there.  It is floored at the bottom of
        # the plot, because a notch's -200 dB null would otherwise squash the real curves.
        if (self.tool.get() == "Filter" and self.spec is not None
                and abs(self.spec.fs - cur.fs) < 1e-6):
            w, mag, _p = F.frequency_response(self.spec, 1024)
            ax.plot(w, np.maximum(mag + float(np.max(m1)), lo), color=T.TEAL, lw=0.9,
                    ls="--", alpha=0.85, label="filter shape")
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB/Hz)")
        ax.set_xlim(0, cur.fs / 2)
        ax.legend(loc="upper right", fontsize=7)
        self.spectrum.draw()

    def _draw_spectrograms(self):
        s = self.session
        cur = s.current
        pairs = (("Before", s.original.data if s.original is not None else cur.data),
                 ("Now", cur.data))
        self.sgram.reset_axes()
        vmax = None
        for i, (label, data) in enumerate(pairs):
            ax = self.sgram.axes(i)
            t, f, db = A.spectrogram(data, cur.fs, 25.0, 0.5)
            if vmax is None:
                vmax = float(np.max(db))
            ax.pcolormesh(t, f, db, cmap=T.CMAP_SPECTRO, vmin=vmax - 80, vmax=vmax,
                          shading="auto", rasterized=True)
            ax.set_title(label)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)" if i == 0 else "")
            ax.grid(False)
        self.sgram.draw()

    def _draw_result(self):
        """The right-hand panel, written for whichever tool is in front."""
        s = self.session
        cur = s.current
        tool = self.tool.get()
        # Keep every line inside the panel: it is a no-wrap monospace box in a narrow card,
        # so anything longer than about 38 characters becomes a sideways scrollbar.
        rows = [("THE SIGNAL", "HEAD"),
                (f"{'Name':<15}{cur.name[:22]}", None),
                (f"{'Length':<15}{cur.duration:.2f} s", None),
                (f"{'Samples':<15}{cur.samples:,}", None),
                (f"{'Rate':<15}{cur.fs:,.0f} Hz", None),
                (f"{'Nyquist':<15}{cur.fs / 2:,.0f} Hz", None),
                ("", None)]

        if tool == "Filter" and self.spec is not None:
            attn = F.estimate_attenuation(self.spec)
            stable = F.is_stable(self.spec)
            rows.append(("THE FILTER", "HEAD"))
            rows.append((f"{'Type':<15}{self.spec.kind} {self.spec.band}", None))
            rows.append((f"{'Cut-off':<15}"
                         + ", ".join(f"{c:,.0f}" for c in self.spec.cutoff) + " Hz", None))
            rows.append((f"{'Built for':<15}{self.spec.fs:,.0f} Hz", None))
            rows.append((f"{'Order':<15}"
                         f"{self.spec.taps if self.spec.kind == 'FIR' else self.spec.order}"
                         + (" taps" if self.spec.kind == "FIR" else ""), None))
            rows.append((f"{'Stable':<15}" + ("yes" if stable else "NO"),
                         "GOOD" if stable else "BAD"))
            rows.append((f"{'Phase':<15}" + ("linear" if self.spec.kind == "FIR"
                                             else "non-linear"),
                         "GOOD" if self.spec.kind == "FIR" else "WARN"))
            stop = attn.get("Stop-band attenuation (dB)", float("nan"))
            if np.isfinite(stop):
                rows.append((f"{'Rejection':<15}{stop:.1f} dB",
                             "GOOD" if stop > 25 else "WARN"))
            rows.append(("", None))

        improvement = s.improvement()
        estimated = {} if improvement else s.estimated_improvement()
        if improvement:
            rows.append(("HOW MUCH BETTER IS IT?", "HEAD"))
            rows.extend(metric_lines(improvement, 21))
        elif estimated:
            noise, speech = (estimated["Background change (dB)"],
                             estimated["Speech level change (dB)"])
            rows.append(("HOW MUCH BETTER IS IT?", "HEAD"))
            rows.append((f"{'Background':<15}{estimated['Background before (dB)']:.1f} → "
                         f"{estimated['Background after (dB)']:.1f} dB", None))
            rows.append((f"{'Noise removed':<15}{-noise:.1f} dB",
                         "GOOD" if noise < -1.0 else "MUTED"))
            rows.append((f"{'Voice level':<15}{speech:+.1f} dB",
                         "GOOD" if abs(speech) < 1.5 else "WARN"))
            rows.append(("", None))
            rows.extend((line, "MUTED") for line in _wrap(
                "No clean copy exists, so this is measured from the recording itself: "
                "background from the pauses, voice from the loud parts. A voice level near "
                "0 dB means the speech survived the clean-up."))
        else:
            rows.append(("MEASUREMENTS", "HEAD"))
            rows.extend(metric_lines(s.signal_metrics(), 21))
            if s.reference is None:
                rows.append(("", None))
                rows.extend((line, "MUTED") for line in _wrap(
                    "No clean copy exists, so SNR against a reference cannot be measured. "
                    "Clean it up to see the background and voice levels before and after."))
        T.write_lines(self.result_box, rows)


def _wrap(text: str, width: int = 38):
    """Break a note into lines that fit the no-wrap monospace result panel."""
    import textwrap
    return textwrap.wrap(text, width)
