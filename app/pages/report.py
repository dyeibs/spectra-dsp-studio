"""
Report & Export - turning the session into evidence.

Collects everything measured during the session into one place and writes it out: a CSV of
the metrics, a multi-page PDF with the plots embedded, the processed audio, and the
before/after images.  This is the page to have open during the defence, because it states
what was done, in order, with the numbers that resulted.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

from dsp import analysis as A, audio_io, imaging as I, presets as P
from .. import theme as T
from ..widgets import PlotCard, metric_lines
from .base import Page

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "exports")


class ReportPage(Page):
    title = "Report & Export"
    subtitle = "Session summary, objective metrics and export to CSV, PDF, WAV or PNG"
    icon = ""
    nav_name = "Report & Export"
    topics = ("signal", "image", "video")

    def build(self):
        page = self.scroll()

        top = self.row(page)
        self._build_actions(top)

        summary = T.card(top)
        summary.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(summary, "Session summary", "What this session has done so far.")
        self.summary_box = T.table_box(summary, height=230)
        self.summary_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        mid = self.row(page)
        pipeline = T.card(mid)
        pipeline.pack(side="left", fill="both", expand=True, padx=(0, 5))
        T.section_title(pipeline, "Processing pipeline",
                        "Every stage applied, in order, with its parameters.")
        self.pipeline_box = T.table_box(pipeline, height=230)
        self.pipeline_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        metrics = T.card(mid)
        metrics.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(metrics, "Objective metrics",
                        "The numbers that go into the exported report.")
        self.metrics_box = T.table_box(metrics, height=230)
        self.metrics_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        self.figure_card = PlotCard(
            page, "Report figure · original, processed and spectra",
            "Exactly the figure embedded in the PDF export.", height=330, nrows=2, ncols=2)
        self.figure_card.pack(fill="both", expand=True, pady=(0, 10))

        self.refresh()

    def _build_actions(self, parent):
        box, body = self.controls(parent, "Export",
                                  f"Files are written to the project's exports folder "
                                  f"unless you choose another location.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        for text, fn, hint in (
                ("Export metrics (CSV)", self._export_csv,
                 "Every metric and pipeline step as a spreadsheet."),
                ("Export report (PDF)", self._export_pdf,
                 "A formatted report with the plots embedded."),
                ("Export processed audio (WAV)", self._export_wav,
                 "The current signal, peak-limited."),
                ("Export original + processed images", self._export_images,
                 "Both images as PNG, for the before/after comparison."),
                ("Export figure (PNG)", self._export_figure,
                 "The report figure at presentation resolution.")):
            T.ghost_button(body, text, fn, width=330, height=32).pack(anchor="w", pady=(4, 0))
            T.hint(body, hint).pack(anchor="w", fill="x", pady=(0, 2))

        T.divider(body, pady=(10, 8), padx=0)
        T.accent_button(body, "Open the exports folder", self._open_folder,
                        width=330, height=32).pack(anchor="w")

    # ----------------------------------------------------------------- data
    def _gather(self) -> dict:
        """Assemble every section of the report as an ordered dict of dicts."""
        s = self.session
        out = {}
        out["Session"] = {
            "Application": "SPECTRA — DSP Studio",
            "Generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Session started": s.started.strftime("%Y-%m-%d %H:%M:%S"),
            "Processing steps": len(s.steps)}

        if s.has_signal:
            cur = s.current
            out["Signal"] = {"Name": cur.name, "Source": cur.source,
                             "Sample rate (Hz)": cur.fs, "Samples": cur.samples,
                             "Duration (s)": cur.duration,
                             "Clean reference available": "yes" if s.reference else "no"}
            out["Spectral descriptors"] = A.summary(cur.data, cur.fs)
            out["Signal metrics"] = s.signal_metrics()
            improvement = s.improvement()
            if improvement:
                out["Enhancement result"] = improvement

        if s.has_image:
            out["Image"] = I.summary(s.image_current)
            quality = s.image_improvement() or I.compare(s.image_original, s.image_current)
            out["Image quality"] = quality
            out["Edge detection"] = _edge_section(s)

        if s.has_video:
            out["Video"] = dict(s.video_info)
            if s.video_preview is not None:
                out["Video frame"] = _video_section(s.video_preview)

        return out

    # ----------------------------------------------------------------- exports
    def _default_path(self, name: str, ext: str) -> str:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(EXPORT_DIR, f"spectra-{name}-{stamp}{ext}")

    def _ask(self, name, ext, types):
        os.makedirs(EXPORT_DIR, exist_ok=True)
        return filedialog.asksaveasfilename(
            title=f"Export {name}", parent=self, defaultextension=ext, filetypes=types,
            initialdir=EXPORT_DIR,
            initialfile=os.path.basename(self._default_path(name, ext)))

    def _export_csv(self):
        path = self._ask("metrics", ".csv", [("CSV", "*.csv")])
        if not path:
            return
        sections = self._gather()
        steps = list(self.session.steps)

        def work():
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh)
                for section, rows in sections.items():
                    writer.writerow([section])
                    writer.writerow(["Metric", "Value"])
                    for key, value in rows.items():
                        writer.writerow([key, _fmt(value)])
                    writer.writerow([])
                writer.writerow(["Processing pipeline"])
                writer.writerow(["#", "Time", "Stage", "Detail", "Parameters"])
                for i, step in enumerate(steps, 1):
                    writer.writerow([i, step.at.strftime("%H:%M:%S"), step.stage, step.detail,
                                     "; ".join(f"{k}={_fmt(v)}" for k, v in step.params.items()
                                               if not isinstance(v, np.ndarray))])
            return path

        self.run(work, lambda p: self.status(f"Wrote {os.path.basename(p)}.", "OK"),
                 busy="Writing the CSV…")

    def _export_pdf(self):
        path = self._ask("report", ".pdf", [("PDF", "*.pdf")])
        if not path:
            return
        sections = self._gather()
        steps = [(s.at.strftime("%H:%M:%S"), s.stage, s.detail) for s in self.session.steps]
        figure_png = self._render_figure_png() if self.session.has_signal else None
        session = self.session
        image = ((session.image_original, session.image_current, session.image_edges)
                 if session.has_image else None)
        video = session.video_preview if session.has_video else None

        def work():
            figures = []
            if figure_png is not None:
                figures.append(("Audio: original vs processed", figure_png, 110))
            if image is not None:
                figures.append(("Image: before vs after — pixels, edges and 2-D spectrum",
                                _image_figure(*image), 225))
            if video is not None:
                figures.append((f"Video: frame {video['index']}, original vs processed",
                                _video_figure(video), 150))
            _write_pdf(path, sections, steps, figures)
            return path

        self.run(work, lambda p: self.status(f"Wrote {os.path.basename(p)}.", "OK"),
                 busy="Building the PDF report…")

    def _export_wav(self):
        if not self.require_signal():
            return
        path = self._ask("audio", ".wav", [("WAV", "*.wav")])
        if not path:
            return
        cur = self.session.current
        self.run(lambda: audio_io.save(path, cur.data, cur.fs),
                 lambda _r: self.status(f"Wrote {os.path.basename(path)}.", "OK"),
                 busy="Writing the audio…")

    def _export_images(self):
        if not self.require_image():
            return
        folder = filedialog.askdirectory(title="Choose a folder for the images", parent=self,
                                         initialdir=EXPORT_DIR)
        if not folder:
            return
        s = self.session
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        original, current = s.image_original, s.image_current

        def work():
            a = os.path.join(folder, f"spectra-original-{stamp}.png")
            b = os.path.join(folder, f"spectra-processed-{stamp}.png")
            I.save_image(a, original)
            I.save_image(b, current)
            return b

        self.run(work, lambda p: self.status(f"Wrote both images to "
                                             f"{os.path.dirname(p)}.", "OK"),
                 busy="Writing the images…")

    def _export_figure(self):
        path = self._ask("figure", ".png", [("PNG", "*.png")])
        if not path:
            return
        try:
            self.figure_card.figure.savefig(path, dpi=200, facecolor=T.PANEL_DEEP)
            self.status(f"Wrote {os.path.basename(path)}.", "OK")
        except Exception as exc:
            self.toast_error("Export failed", str(exc))

    def _render_figure_png(self):
        """Render the report figure to an in-memory PNG for the PDF."""
        import io
        buffer = io.BytesIO()
        try:
            self.figure_card.figure.savefig(buffer, format="png", dpi=160,
                                            facecolor=T.PANEL_DEEP)
            buffer.seek(0)
            return buffer
        except Exception:
            return None

    def _open_folder(self):
        os.makedirs(EXPORT_DIR, exist_ok=True)
        try:
            os.startfile(EXPORT_DIR)                      # noqa: S606 - Windows shell open
            self.status(f"Opened {EXPORT_DIR}.", "INFO")
        except Exception as exc:
            self.status(f"Could not open the folder: {exc}", "WARN")

    # ----------------------------------------------------------------- drawing
    def refresh(self):
        sections = self._gather()
        lines = []
        for name, rows in sections.items():
            lines.append((name.upper(), "HEAD"))
            lines.extend(metric_lines(rows, 28))
            lines.append(("", None))
        T.write_lines(self.summary_box, lines or [("Nothing to report yet.", "MUTED")])

        T.write_lines(self.pipeline_box,
                      [(f"{i:>2}. {s.line()}", None)
                       for i, s in enumerate(self.session.steps, 1)]
                      or [("No processing applied yet.", "MUTED")])

        metrics = dict(self.session.signal_metrics())
        metrics.update(self.session.improvement())
        T.write_lines(self.metrics_box,
                      metric_lines(metrics, 30) if metrics
                      else [("No signal loaded.", "MUTED")])
        self._draw_figure()

    def _draw_figure(self):
        s = self.session
        self.figure_card.reset_axes()
        if not s.has_signal:
            self.figure_card.message("Load a signal to build the report figure")
            return

        cur = s.current
        t = np.arange(cur.samples) / cur.fs

        ax = self.figure_card.axes(0)
        ax.plot(t, cur.data, color=T.ACCENT, lw=0.6, alpha=0.8, label="processed")
        if s.original is not None:
            # The original drawn thin on top, so it shows wherever processing removed level.
            n = min(s.original.data.size, cur.samples)
            ax.plot(t[:n], s.original.data[:n], color=T.TEXT, lw=0.35, alpha=0.45,
                    label="original", zorder=3)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=6)
        ax.margins(x=0)

        ax2 = self.figure_card.axes(1)
        f1, m1 = A.spectrum(cur.data, cur.fs)
        ax2.plot(f1, m1, color=T.ACCENT, lw=0.8, label="processed")
        if s.original is not None:
            f0, m0 = A.spectrum(s.original.data, s.original.fs)
            ax2.plot(f0, m0, color=T.TEXT, lw=0.6, ls=(0, (3, 2)), alpha=0.6,
                     label="original", zorder=3)
        ax2.set_title("Magnitude spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_xlim(0, cur.fs / 2)
        ax2.legend(loc="upper right", fontsize=6)

        for index, (label, data) in enumerate((("Original", s.original.data
                                                if s.original is not None else cur.data),
                                               ("Processed", cur.data))):
            ax3 = self.figure_card.axes(2 + index)
            tt, ff, db = A.spectrogram(data, cur.fs, 25.0, 0.5)
            vmax = float(np.max(db))
            ax3.pcolormesh(tt, ff, db, cmap=T.CMAP_SPECTRO, vmin=vmax - 80, vmax=vmax,
                           shading="auto", rasterized=True)
            ax3.set_title(f"{label} spectrogram")
            ax3.set_xlabel("Time (s)")
            ax3.grid(False)
        self.figure_card.draw()


def _edge_section(s) -> dict:
    """Edge-detection figures for the report, with the settings last used on the Image page."""
    settings = s.image_edges or P.EdgeSettings()
    e_before = settings.detect(s.image_original)
    e_after = settings.detect(s.image_current)
    out = {"Detector": settings.describe(),
           "Edge pixels before (%)": P.edge_density(e_before),
           "Edge pixels after (%)": P.edge_density(e_after)}
    ref = s.image_reference
    if ref is not None and not np.array_equal(ref, s.image_original):
        e_ref = settings.detect(ref)
        before, after = P.edge_match(e_ref, e_before), P.edge_match(e_ref, e_after)
        for key in ("Precision", "Recall", "F1"):
            out[f"Edge {key} before"] = before[key]
            out[f"Edge {key} after"] = after[key]
    return out


def _video_section(preview: dict) -> dict:
    before = P.analyse_image(preview["frame"])
    after = P.analyse_image(preview["cleaned"])
    return {"Frame": preview["index"], "Cleaning": preview["cleaning"],
            "Noise sigma before": before["Noise (sigma)"],
            "Noise sigma after": after["Noise (sigma)"],
            "Sharpness before": before["Sharpness"], "Sharpness after": after["Sharpness"],
            "Edge detector": preview["edges"].describe(),
            "Edge pixels before (%)": P.edge_density(preview["edge before"]),
            "Edge pixels after (%)": P.edge_density(preview["edge after"])}


def _png(figure):
    """Render an off-screen figure to an in-memory PNG.  Uses the Agg canvas directly, never
    pyplot, so it is safe on the worker thread the PDF is built on."""
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, facecolor=T.PANEL_DEEP)
    buffer.seek(0)
    return buffer


def _show(ax, img, title):
    arr = np.clip(np.nan_to_num(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)
    ax.imshow(arr, cmap=None if arr.ndim == 3 else "gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def _image_figure(before, after, settings):
    """Before/after image, edge maps and 2-D spectra, side by side."""
    from matplotlib.figure import Figure
    settings = settings or P.EdgeSettings()
    fig = Figure(figsize=(7.0, 9.3), facecolor=T.PANEL_DEEP)
    axes = fig.subplots(3, 2)
    _show(axes[0][0], before, "Before")
    _show(axes[0][1], after, "After")
    _show(axes[1][0], settings.detect(before), f"{settings.operator} edges · before")
    _show(axes[1][1], settings.detect(after), f"{settings.operator} edges · after")
    spectra = (I.fft2_spectrum(before), I.fft2_spectrum(after))
    lo = min(float(np.percentile(sp, 1)) for sp in spectra)
    hi = max(float(sp.max()) for sp in spectra)
    for ax, spec, name in zip(axes[2], spectra, ("before", "after")):
        ax.imshow(spec, cmap=T.CMAP_SPECTRO, vmin=lo, vmax=hi)
        ax.set_title(f"2-D spectrum log|F(u,v)| · {name}", fontsize=9)
        ax.set_axis_off()
    fig.tight_layout(pad=1.0)
    return _png(fig)


def _video_figure(preview):
    """Original and processed frame, the edges of the processed one, and both histograms."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7.0, 6.2), facecolor=T.PANEL_DEEP)
    axes = fig.subplots(2, 2)
    _show(axes[0][0], preview["frame"], "Original frame")
    _show(axes[0][1], preview["cleaned"], "Processed frame")
    _show(axes[1][0], preview["edge after"],
          f"{preview['edges'].operator} edges of processed frame")
    ax = axes[1][1]
    for img, colour, label in ((preview["frame"], T.FAINT, "original"),
                               (preview["cleaned"], T.ACCENT, "processed")):
        centres, counts = I.histogram(img)
        ax.plot(centres, counts, color=colour, lw=1.0, label=label)
    ax.set_title("Brightness histogram", fontsize=9)
    ax.set_xlim(0, 255)
    ax.legend(fontsize=7)
    fig.tight_layout(pad=1.0)
    return _png(fig)


def _fmt(value) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "n/a"
        if np.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def _write_pdf(path, sections, steps, figures) -> None:
    """Write the PDF report with reportlab.

    Laid out as flowables rather than drawn on a canvas, so a long pipeline simply flows onto
    further pages instead of running off the bottom of the first one.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    accent = colors.HexColor("#c4232f")
    ink = colors.HexColor("#1a1a1e")
    grey = colors.HexColor("#6b6b75")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=accent, fontSize=20,
                        spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=grey, fontSize=9,
                         spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=accent, fontSize=11,
                        spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], textColor=ink, fontSize=8.5)

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="SPECTRA DSP Report")
    story = [Paragraph("SPECTRA — DSP Processing Report", h1),
             Paragraph("Signal Processing, Enhancement, Characterisation, Transform &amp; "
                       "Restoration Application", sub)]

    def table(rows, widths=(70 * mm, 95 * mm)):
        data = [[Paragraph(f"<b>{k}</b>", body), Paragraph(_fmt(v), body)] for k, v in rows]
        tbl = Table(data, colWidths=list(widths), hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d8d8de")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#f5f5f7")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return tbl

    for name, rows in sections.items():
        if not rows:
            continue
        story.append(Paragraph(name, h2))
        story.append(table(list(rows.items())))

    story.append(Paragraph("Processing pipeline", h2))
    if steps:
        data = [[Paragraph("<b>#</b>", body), Paragraph("<b>Time</b>", body),
                 Paragraph("<b>Stage</b>", body), Paragraph("<b>Detail</b>", body)]]
        data += [[Paragraph(str(i), body), Paragraph(when, body), Paragraph(stage, body),
                  Paragraph(detail, body)]
                 for i, (when, stage, detail) in enumerate(steps, 1)]
        tbl = Table(data, colWidths=[10 * mm, 20 * mm, 38 * mm, 97 * mm], hAlign="LEFT",
                    repeatRows=1)
        tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d8d8de")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ececf0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(tbl)
    else:
        story.append(Paragraph("No processing was applied in this session.", body))

    for title, png, height in figures:
        story.append(PageBreak())
        story.append(Paragraph(title, h2))
        story.append(Spacer(1, 4))
        story.append(Image(png, width=170 * mm, height=height * mm))

    doc.build(story)
