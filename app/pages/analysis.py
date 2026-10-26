"""
Audio Analysis - everything you *look at*, as opposed to everything you do.

Two views of the same signal, swapped by the chooser at the top.  Frequency applies to any
signal; Speech only makes sense on a voice, and keeping them apart stops a page full of
pitch contours appearing when the signal is a square wave.

Neither view changes the signal.  This page only measures.
"""
from __future__ import annotations

import customtkinter as ctk
import numpy as np

from dsp import analysis as A, denoise as D, metrics as M
from .. import theme as T
from ..widgets import PageStack, PlotCard, SliderRow, metric_lines
from .base import Page

VIEWS = ("Frequency", "Speech")

#: Typical fundamental-frequency ranges.  Narrowing the search is what stops the tracker
#: locking onto a harmonic instead of the fundamental.
VOICE_RANGES = {"Any voice (60–400 Hz)": (60.0, 400.0),
                "Low / male (70–200 Hz)": (70.0, 200.0),
                "High / female (150–350 Hz)": (150.0, 350.0),
                "Child (200–500 Hz)": (200.0, 500.0)}


class AnalysisPage(Page):
    title = "Audio Analysis"
    subtitle = "What the signal is made of, and what the voice in it is doing"
    icon = ""
    nav_name = "Audio Analysis"

    def build(self):
        self._pending = None
        page = self.scroll()

        chooser = self.row(page, pady=(0, 8))
        self.view = T.segmented(chooser, VIEWS, self._on_view)
        self.view.set(VIEWS[0])
        self.view.pack(anchor="w")

        self.stack = PageStack(page)
        self.stack.pack(fill="both", expand=True)
        self._build_frequency(self.stack.add("Frequency"))
        self._build_speech(self.stack.add("Speech"))
        self.stack.set("Frequency")

        self.refresh()

    def _on_view(self, name):
        self.stack.set(name)
        self.refresh()

    # ================================================================= frequency
    def _build_frequency(self, parent):
        top = self.row(parent)
        box, body = self.controls(
            top, "Analysis settings",
            "These change only how the signal is measured — the signal itself is untouched.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.frame_ms = SliderRow(body, "Frame length", 4, 120, 25, "{:.0f} ms",
                                  command=lambda _v: self._debounce(), width=180)
        self.frame_ms.pack(fill="x", pady=5)
        T.hint(body, "Short frames pinpoint when something happened. Long frames pinpoint "
                     "what pitch it was. You cannot have both — that is the uncertainty "
                     "principle, and this slider is it. 25 ms is the usual compromise for "
                     "speech.").pack(anchor="w", fill="x", pady=(2, 8))

        self.log_freq = T.checkbox(body, "Spread out the low frequencies (log scale)",
                                   command=self.refresh)
        self.log_freq.pack(anchor="w", pady=(2, 6))
        self.compare = T.checkbox(body, "Overlay the original for comparison",
                                  command=self.refresh)
        self.compare.select()
        self.compare.pack(anchor="w")

        summary = T.card(top)
        summary.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(summary, "Summary of the spectrum",
                        "Single numbers describing its shape, for comparing two signals "
                        "without reading the plots.")
        self.summary_box = T.table_box(summary, height=185)
        self.summary_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self.fft_plot = PlotCard(
            parent, "Which frequencies are present",
            "The whole signal at once. A tone is a spike, noise is a flat carpet, and a "
            "harmonic sound is a row of evenly spaced spikes.", height=245)
        self.fft_plot.pack(fill="both", expand=True, pady=(0, 10))

        self.sgram_plot = PlotCard(
            parent, "When each frequency happens",
            "Time runs left to right, frequency bottom to top, brightness is loudness.",
            height=285)
        self.sgram_plot.pack(fill="both", expand=True, pady=(0, 10))

    # ================================================================= speech
    def _build_speech(self, parent):
        top = self.row(parent)
        box, body = self.controls(
            top, "Voice settings",
            "Telling the pitch tracker roughly what range to expect is what keeps it from "
            "locking onto a harmonic and reporting the wrong octave.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.voice = T.option_menu(body, tuple(VOICE_RANGES), 240,
                                   command=lambda _v: self.refresh())
        self.voice.set("Any voice (60–400 Hz)")
        self._row(body, "Voice range", self.voice)

        T.hint(body, "Pre-emphasis boosts the high frequencies before the features are "
                     "extracted, offsetting the natural roll-off of the human voice. It is "
                     "the standard first step of every speech front end.").pack(
            anchor="w", fill="x", pady=(10, 8))

        bar = self.button_bar(body)
        T.accent_button(bar, "Analyse", self.refresh, width=100).pack(side="left")
        T.ghost_button(bar, "Apply pre-emphasis", self._apply_pre_emph,
                       width=160).pack(side="left", padx=6)

        stats = T.card(top)
        stats.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(stats, "What the voice is doing",
                        "Measured only over the frames that were actually voiced.")
        self.stats_box = T.table_box(stats, height=190)
        self.stats_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        r1 = self.row(parent)
        self.energy_plot = PlotCard(
            r1, "Loudness and noisiness over time",
            "Loud with few zero-crossings is a vowel; quiet with many is an 's' or 'f'; both "
            "low is silence. That is a working voice-activity detector.",
            height=275, nrows=3)
        self.energy_plot.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.pitch_plot = PlotCard(
            r1, "Pitch over time",
            "The fundamental frequency, from the autocorrelation peak in each frame. Gaps "
            "are frames with no pitch to measure.", height=275)
        self.pitch_plot.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.mfcc_plot = PlotCard(
            parent, "MFCC features",
            "Each column is one 25 ms frame compressed into 13 numbers describing the shape "
            "of its spectrum, on the mel scale that matches how the ear hears. This is the "
            "input a speech recogniser is given.", height=255)
        self.mfcc_plot.pack(fill="both", expand=True, pady=(0, 10))

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

    def _debounce(self):
        """Redraw once the slider settles, so a drag is one redraw rather than fifty."""
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(220, self._fire)

    def _fire(self):
        self._pending = None
        self.refresh()

    def _apply_pre_emph(self):
        if not self.require_signal():
            return
        cur = self.session.current
        self.session.set_current(D.pre_emphasis(cur.data, 0.97), cur.name)
        self.session.add_step("Pre-emphasis", "y[n] = x[n] − 0.97·x[n−1]")
        self.status("Applied pre-emphasis to the signal.", "OK")

    # ----------------------------------------------------------------- drawing
    def refresh(self):
        if self.view.get() == "Frequency":
            self._refresh_frequency()
        else:
            self._refresh_speech()

    # ---- frequency view
    def _refresh_frequency(self):
        s = self.session
        if not s.has_signal:
            self.fft_plot.message("Load a signal on the Audio Studio page")
            self.sgram_plot.message("Load a signal on the Audio Studio page")
            T.write_lines(self.summary_box, [("No signal loaded.", "MUTED")])
            return

        cur = s.current
        x, fs = cur.data, cur.fs
        nyq = fs / 2.0
        logf = bool(self.log_freq.get())
        overlay = bool(self.compare.get()) and s.original is not None

        self.fft_plot.reset_axes()
        ax = self.fft_plot.axes()
        if overlay and s.original.data.size:
            f0, m0 = A.spectrum(s.original.data, s.original.fs)
            ax.plot(f0, m0, color=T.FAINT, lw=0.7, label="original")
        f, m = A.spectrum(x, fs)
        ax.plot(f, m, color=T.ACCENT, lw=0.9, label="now")
        peak = A.dominant_frequency(x, fs)
        if np.isfinite(peak):
            ax.axvline(peak, color=T.TEAL, ls="--", lw=0.9)
            ax.annotate(f"loudest: {peak:,.0f} Hz", (peak, float(np.max(m))), color=T.TEAL,
                        fontsize=7.5, xytext=(5, -9), textcoords="offset points")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Level (dB)")
        if logf:
            ax.set_xscale("log")
            ax.set_xlim(max(nyq / 2000.0, 1.0), nyq)
        else:
            ax.set_xlim(0, nyq)
        ax.legend(loc="upper right", fontsize=7)
        self.fft_plot.draw()

        self.sgram_plot.reset_axes()
        ax2 = self.sgram_plot.axes()
        t, fsg, db = A.spectrogram(x, fs, self.frame_ms.get(), 0.5)
        vmax = float(np.max(db))
        ax2.pcolormesh(t, fsg, db, cmap=T.CMAP_SPECTRO, vmin=vmax - 80, vmax=vmax,
                       shading="auto", rasterized=True)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        ax2.grid(False)
        frame_samples = max(int(self.frame_ms.get() * fs / 1000.0), 1)
        self.sgram_plot.set_status(
            f"{self.frame_ms.get():.0f} ms frames · {fs / frame_samples:,.0f} Hz resolution",
            T.MUTED)
        self.sgram_plot.draw()

        rows = {"Loudest frequency (Hz)": A.dominant_frequency(x, fs),
                "Centre of gravity (Hz)": A.spectral_centroid(x, fs),
                "Spread (Hz)": A.spectral_bandwidth(x, fs),
                "85% of energy is below (Hz)": A.spectral_rolloff(x, fs),
                "Tonal or noisy (0 = tone, 1 = noise)": A.spectral_flatness(x, fs),
                "Frequency resolution (Hz)": fs / frame_samples,
                "Time resolution (ms)": self.frame_ms.get()}
        T.write_lines(self.summary_box, metric_lines(rows, 38))

    # ---- speech view
    def _refresh_speech(self):
        s = self.session
        plots = (self.energy_plot, self.pitch_plot, self.mfcc_plot)
        if not s.has_signal:
            for p in plots:
                p.message("Record or generate speech on the Audio Studio page")
            T.write_lines(self.stats_box, [("No signal loaded.", "MUTED")])
            return

        cur = s.current
        x, fs = cur.data, cur.fs
        f0_lo, f0_hi = VOICE_RANGES[self.voice.get()]

        def work():
            te, energy = A.short_time_energy(x, fs, 25.0, 10.0)
            tz, zcr = A.short_time_zcr(x, fs, 25.0, 10.0)
            tp, f0 = A.pitch_track(x, fs, f0_lo, f0_hi, 40.0, 10.0, 0.3)
            tm, coeffs = A.mfcc(x, fs, 13, 26, 25.0, 10.0, 0.97)
            return te, energy, tz, zcr, tp, f0, tm, coeffs

        self.run(work, self._render_speech, busy="Measuring the voice…")

    def _render_speech(self, result):
        te, energy, tz, zcr, tp, f0, tm, coeffs = result
        cur = self.session.current
        x, fs = cur.data, cur.fs
        f0_lo, f0_hi = VOICE_RANGES[self.voice.get()]

        self.energy_plot.reset_axes()
        ax = self.energy_plot.axes(0)
        ax.plot(np.arange(x.size) / fs, x, color=T.FAINT, lw=0.6)
        ax.set_ylabel("Waveform")
        ax.margins(x=0)

        ax1 = self.energy_plot.axes(1)
        ax1.plot(te, energy, color=T.ACCENT, lw=1.0)
        ax1.fill_between(te, energy, float(np.min(energy)),
                         color=T.blend(T.ACCENT, T.PANEL_DEEP, 0.2))
        ax1.set_ylabel("Loudness (dB)")
        ax1.margins(x=0)

        ax2 = self.energy_plot.axes(2)
        ax2.plot(tz, zcr, color=T.TEAL, lw=1.0)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Zero crossings")
        ax2.margins(x=0)
        self.energy_plot.draw()

        self.pitch_plot.reset_axes()
        ax3 = self.pitch_plot.axes()
        voiced = np.isfinite(f0)
        ax3.plot(tp, f0, color=T.ACCENT, lw=1.5)
        ax3.scatter(tp[voiced], f0[voiced], s=9, color=T.ACCENT, zorder=3)
        if np.any(voiced):
            median = float(np.nanmedian(f0))
            ax3.axhline(median, color=T.TEAL, ls="--", lw=0.9)
            ax3.annotate(f"average pitch {median:,.0f} Hz",
                         (tp[0] if tp.size else 0, median), color=T.TEAL, fontsize=7.5,
                         xytext=(5, 5), textcoords="offset points")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pitch (Hz)")
        ax3.set_ylim(f0_lo * 0.8, f0_hi * 1.1)
        ax3.margins(x=0)
        self.pitch_plot.set_status(f"{int(np.sum(voiced))} of {f0.size} frames voiced",
                                   T.MUTED)
        self.pitch_plot.draw()

        self.mfcc_plot.reset_axes()
        ax4 = self.mfcc_plot.axes()
        extent = [tm[0] if tm.size else 0, tm[-1] if tm.size else 1, 0, coeffs.shape[0]]
        ax4.imshow(coeffs, aspect="auto", origin="lower", cmap=T.CMAP_DIVERGING,
                   extent=extent, interpolation="nearest")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Feature number")
        ax4.grid(False)
        self.mfcc_plot.set_status(f"{coeffs.shape[0]} features × {coeffs.shape[1]} frames",
                                  T.MUTED)
        self.mfcc_plot.draw()

        voiced_f0 = f0[voiced]
        rows = {"Frames measured": int(f0.size),
                "Frames with a pitch": int(np.sum(voiced)),
                "Proportion voiced": float(np.mean(voiced)) if f0.size else 0.0}
        if voiced_f0.size:
            rows.update({"Average pitch (Hz)": float(np.median(voiced_f0)),
                         "Pitch range (Hz)": float(np.ptp(voiced_f0)),
                         "Pitch steadiness (Hz std)": float(np.std(voiced_f0))})
        else:
            rows["Pitch"] = "no voiced frames found — try a wider voice range"
        rows.update({"Average loudness (dB)": float(np.mean(energy)),
                     "Loudness range (dB)": float(np.ptp(energy)),
                     "Average zero-crossing rate": float(np.mean(zcr)),
                     "Brightness (Hz)": A.spectral_centroid(x, fs),
                     "MFCC matrix": f"{coeffs.shape[0]} × {coeffs.shape[1]}"})
        ref = self.session.reference_array()
        if ref is not None:
            rows["Speech-weighted SNR (dB)"] = M.segmental_snr_db(ref, x, fs)
        T.write_lines(self.stats_box, metric_lines(rows, 30))
