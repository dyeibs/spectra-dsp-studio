"""
Dashboard - the state of the session at a glance.

Deliberately read-only apart from the quick-start buttons: it answers "what is loaded, what
has been done to it, and did it help", and sends the user to the right page for anything
else.
"""
from __future__ import annotations

import customtkinter as ctk
import numpy as np

from dsp import analysis as A
from .. import theme as T
from ..widgets import MetricCard, PlotCard, metric_lines
from .base import Page


class DashboardPage(Page):
    title = "Dashboard"
    subtitle = "Session overview, signal summary and the processing pipeline"
    icon = ""
    nav_name = "Dashboard"
    topics = ("signal", "log")

    def build(self):
        page = self.scroll()

        # ---- KPI tiles
        cards = self.row(page, pady=(0, 10))
        self.card_source = MetricCard(cards, "Source", "")
        self.card_rate = MetricCard(cards, "Sample rate", "")
        self.card_duration = MetricCard(cards, "Duration", "")
        self.card_dominant = MetricCard(cards, "Dominant freq.", "")
        self.card_quality = MetricCard(cards, "SNR improvement", "")
        self.cards = [self.card_source, self.card_rate, self.card_duration,
                      self.card_dominant, self.card_quality]
        for c in self.cards:
            c.pack(side="left", fill="x", expand=True, padx=4)

        # ---- waveform + spectrum preview
        plots = self.row(page, pady=(0, 10))
        self.wave = PlotCard(plots, "Waveform · original vs processed",
                             "The time-domain view: the original above, the processed "
                             "signal below, on the same amplitude scale so differences in "
                             "level and envelope are directly comparable.", height=250,
                             nrows=2)
        self.wave.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.spec = PlotCard(plots, "Magnitude spectrum",
                             "The frequency-domain view of the same pair - what the DFT says "
                             "was added or removed.", height=250)
        self.spec.pack(side="left", fill="both", expand=True, padx=(5, 0))

        # ---- pipeline + metrics
        lower = self.row(page, pady=(0, 10))
        left = T.card(lower)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        T.section_title(left, "Processing pipeline",
                        "Every stage applied to the current signal, in order.")
        self.pipeline_box = T.table_box(left, height=150)
        self.pipeline_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        right = T.card(lower)
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(right, "Signal metrics",
                        "Reference-based rows appear only when a clean ground truth exists.")
        self.metrics_box = T.table_box(right, height=150)
        self.metrics_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        # ---- quick start + activity
        bottom = self.row(page, pady=(0, 10))
        quick = T.card(bottom)
        quick.pack(side="left", fill="both", expand=True, padx=(0, 5))
        T.section_title(quick, "Quick start",
                        "Load a problem signal, then open Audio Studio → Clean up.")
        bar = T.scroll_frame(quick, orientation="horizontal", height=52)
        bar.pack(fill="x", padx=14, pady=(0, 12))
        presets = (("Noisy speech", self._demo_speech),
                   ("Buzzing hum", self._demo_hum),
                   ("Crackly clicks", self._demo_clicks),
                   ("Frequency sweep", self._demo_chirp))
        for text, fn in presets:
            T.ghost_button(bar, text, fn, width=152, height=30).pack(side="left", padx=4, pady=6)

        activity = T.card(bottom)
        activity.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(activity, "Activity log", "The most recent events in this session.")
        self.log_box = T.table_box(activity, height=120)
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self.refresh()

    # ----------------------------------------------------------------- demos
    def _load_demo(self, builder, label):
        """Build a demonstration signal on a worker thread, then install it."""
        def work():
            return builder()

        def done(result):
            clean, noisy, fs, name = result
            self.session.set_signal(noisy, fs, name, source="generated", reference=clean)
            self.session.add_step("Demo signal", name)
            self.status(f"Loaded the {label} demonstration.", "OK")

        self.run(work, done, busy=f"Building the {label} demo…")

    def _demo_speech(self):
        from dsp import signals as S

        def build():
            fs = 8000.0
            clean = S.generate("Speech-like formants", 2.5, fs, 120.0)
            return clean, S.add_noise(clean, fs, "White Gaussian (AWGN)", 5.0, seed=7), fs, \
                "Speech + white noise (5 dB SNR)"
        self._load_demo(build, "noisy speech")

    def _demo_hum(self):
        from dsp import signals as S

        def build():
            fs = 8000.0
            clean = S.generate("Multi-tone", 2.0, fs, 440.0)
            return clean, S.add_noise(clean, fs, "Mains hum (60 Hz)", 6.0, seed=3), fs, \
                "Multi-tone + 60 Hz mains hum"
        self._load_demo(build, "mains hum")

    def _demo_clicks(self):
        from dsp import signals as S

        def build():
            fs = 8000.0
            clean = S.generate("Multi-tone", 2.0, fs, 220.0)
            return clean, S.add_noise(clean, fs, "Impulse / click", density=0.01, seed=5), fs, \
                "Tone complex with impulse clicks"
        self._load_demo(build, "click removal")

    def _demo_chirp(self):
        from dsp import signals as S

        def build():
            fs = 8000.0
            clean = S.generate("Chirp (linear)", 3.0, fs, 100.0, freq2=3200.0)
            return clean, S.add_noise(clean, fs, "Pink (1/f)", 8.0, seed=11), fs, \
                "Linear chirp 100 Hz to 3.2 kHz + pink noise"
        self._load_demo(build, "chirp sweep")

    # ----------------------------------------------------------------- refresh
    def refresh(self):
        self._refresh_cards()
        self._refresh_plots()
        T.write_lines(self.pipeline_box, self.session.pipeline_lines())
        metrics = self.session.signal_metrics()
        T.write_lines(self.metrics_box,
                      metric_lines(metrics) if metrics else [("No signal loaded.", "MUTED")])
        self._refresh_log()

    def _refresh_cards(self):
        s = self.session
        if not s.has_signal:
            for c in self.cards:
                c.clear()
            return
        cur = s.current
        self.card_source.set(cur.source.title(), cur.name[:40], T.TEAL)
        self.card_rate.set(f"{cur.fs:,.0f} Hz", f"Nyquist {cur.fs / 2:,.0f} Hz", T.BLUE)
        self.card_duration.set(f"{cur.duration:.2f} s", f"{cur.samples:,} samples", T.ACCENT)

        dom = A.dominant_frequency(cur.data, cur.fs)
        self.card_dominant.set("—" if not np.isfinite(dom) else f"{dom:,.0f} Hz",
                               f"centroid {A.spectral_centroid(cur.data, cur.fs):,.0f} Hz",
                               T.PURPLE)

        imp = s.improvement()
        if imp:
            delta = imp["SNR improvement (dB)"]
            colour = T.GREEN if delta > 0.5 else (T.ORANGE if delta > -0.5 else T.RED)
            self.card_quality.set(f"{delta:+.2f} dB",
                                  f"{imp['SNR before (dB)']:.1f} → "
                                  f"{imp['SNR after (dB)']:.1f} dB", colour)
        else:
            self.card_quality.set("n/a", "No clean copy to compare against", T.FAINT)

    def _refresh_plots(self):
        s = self.session
        if not s.has_signal:
            self.wave.message("Load or generate a signal to begin")
            self.spec.message("Load or generate a signal to begin")
            return

        cur = s.current
        self.wave.reset_axes()
        t = np.arange(cur.samples) / cur.fs
        before = s.original.data if s.original is not None else cur.data
        tb = np.arange(before.size) / (s.original.fs if s.original is not None else cur.fs)
        peak = max(float(np.max(np.abs(before))) if before.size else 0.0,
                   float(np.max(np.abs(cur.data))) if cur.samples else 0.0, 1e-6) * 1.08
        top, bottom = self.wave.axes(0), self.wave.axes(1)
        top.plot(tb, before, color=T.MUTED, lw=0.6)
        top.set_title("Original", fontsize=8, loc="left")
        if s.reference is not None:
            ref = s.reference.data[:cur.samples]
            bottom.plot(t[:ref.size], ref, color=T.TEAL, lw=0.6, alpha=0.7,
                        label="clean reference")
        bottom.plot(t, cur.data, color=T.ACCENT, lw=0.6, label="processed")
        bottom.set_title("Processed", fontsize=8, loc="left")
        if s.reference is not None:
            bottom.legend(loc="upper right", ncol=2, fontsize=7)
        span = max(tb[-1] if tb.size else 0.0, t[-1] if t.size else 0.0, 1e-3)
        for ax in (top, bottom):
            ax.set_ylim(-peak, peak)
            ax.set_xlim(0, span)
        top.tick_params(labelbottom=False)
        bottom.set_xlabel("Time (s)")
        self.wave.draw()

        self.spec.reset_axes()
        ax2 = self.spec.axes()
        f1, m1 = A.spectrum(cur.data, cur.fs)
        ax2.plot(f1, m1, color=T.ACCENT, lw=0.9, label="processed")
        if s.original is not None and s.original.data.size:
            # On top and dashed so it stays visible where the curves coincide.
            f0, m0 = A.spectrum(s.original.data, s.original.fs)
            ax2.plot(f0, m0, color=T.TEXT, lw=0.7, ls=(0, (3, 2)), alpha=0.6,
                     label="original", zorder=3)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlim(0, cur.fs / 2)
        ax2.legend(loc="upper right", ncol=2)
        self.spec.draw()

    def _refresh_log(self):
        colours = {"OK": "GOOD", "WARN": "WARN", "ERROR": "BAD", "INFO": None}
        lines = [(f"{e.at:%H:%M:%S}  {e.level:<5} {e.message}", colours.get(e.level))
                 for e in self.session.events[-40:]]
        T.write_lines(self.log_box, lines or [("Nothing has happened yet.", "MUTED")])
        try:
            self.log_box.see("end")
        except Exception:
            pass
