"""
Image Restore - take a blurry, noisy or flat image and make it clear again.

The page is built around one job rather than a toolbox.  Three things can be wrong with a
photograph (it is soft, it is grainy, it is flat), so there are three dials and an Auto
button that measures the image and sets them itself.  Everything else the engine can do
stays in `dsp/imaging.py` where it belongs.

"Damage it" exists so the restoration can be *measured*: blurring an image inside the
application records the exact point-spread function, which turns the deblur step into a true
Wiener deconvolution against a known kernel and lets the page report a real PSNR and SSIM
against the pristine original.

Below the repair sit two views that make the processing visible: the 2-D spectrum before and
after (where periodic interference shows up as bright points that vanish once removed),
and edge detection on both images, scored against the edges of the untouched original.
"""
from __future__ import annotations

import os
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

from dsp import imaging as I, presets as P
from .. import theme as T
from ..widgets import ImageCard, PlotCard, SliderRow, metric_lines
from .base import Page


class ImagePage(Page):
    title = "Image Restore"
    subtitle = "Make a blurry, noisy or flat image clear again — and measure the result"
    icon = ""
    nav_name = "Image Restore"
    topics = ("image",)

    def build(self):
        page = self.scroll()

        top = self.row(page)
        self._build_source(top)
        self._build_damage(top)

        views = self.row(page)
        self.card_before = ImageCard(views, "Before", height=320)
        self.card_before.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.card_after = ImageCard(views, "After", height=320)
        self.card_after.pack(side="left", fill="both", expand=True, padx=(5, 0))

        lower = self.row(page)
        self._build_restore(lower)
        self._build_quality(lower)

        self.hist = PlotCard(page, "Brightness distribution",
                             "How the pixel values are spread, before and after. Denoising "
                             "narrows this curve; contrast widens it.", height=210)
        self.hist.pack(fill="both", expand=True, pady=(0, 10))

        self.spectrum = PlotCard(
            page, "2-D frequency spectrum",
            "log |F(u, v)| of the luminance, centred on DC. Low frequencies (smooth shading) "
            "sit in the middle, fine detail and grain towards the edge. A repeating pattern "
            "is a pair of bright points; circles mark the peaks that were removed.",
            height=290, ncols=2)
        self.spectrum.pack(fill="both", expand=True, pady=(0, 10))

        edges = self.row(page)
        self._build_edges(edges)
        self._build_edge_scores(edges)

        edge_views = self.row(page)
        self.edge_before = ImageCard(edge_views, "Edges · before", height=300)
        self.edge_before.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.edge_after = ImageCard(edge_views, "Edges · after", height=300)
        self.edge_after.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self._edge_job = 0
        self._edge_pending = None
        self._edge_result = None
        self._notched = []
        self.refresh()

    # ----------------------------------------------------------------- panels
    def _build_source(self, parent):
        box, body = self.controls(parent, "1 · Pick an image",
                                  "Load a photo of your own, or use the built-in demo scene.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        bar = self.button_bar(body, pady=(0, 6))
        T.accent_button(bar, "Load image…", self._load, width=130).pack(side="left")
        T.ghost_button(bar, "Demo scene", self._demo, width=110).pack(side="left", padx=6)
        T.ghost_button(bar, "Save result…", self._save, width=110).pack(side="left")

        self.condition = T.table_box(body, height=118)
        self.condition.pack(fill="x", pady=(8, 0))

    def _build_damage(self, parent):
        box, body = self.controls(
            parent, "2 · Damage it (optional)",
            "Blur and add noise on purpose. The exact blur kernel is remembered, so the "
            "repair below becomes a real deconvolution and the quality scores are measured "
            "against the untouched original.")
        box.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.blur = SliderRow(body, "Blur", 0.0, 1.0, 0.5, "{:.0%}", width=175)
        self.blur.pack(fill="x", pady=4)
        self.noise = SliderRow(body, "Noise", 0.0, 1.0, 0.3, "{:.0%}", width=175)
        self.noise.pack(fill="x", pady=4)
        self.pattern = SliderRow(body, "Stripe pattern", 0.0, 1.0, 0.0, "{:.0%}", width=175)
        self.pattern.pack(fill="x", pady=4)

        bar = self.button_bar(body)
        T.accent_button(bar, "Damage it", self._degrade, width=120).pack(side="left")
        T.ghost_button(bar, "Undo damage", self._undo_damage,
                       width=130).pack(side="left", padx=6)

    def _build_restore(self, parent):
        box, body = self.controls(
            parent, "3 · Make it clear",
            "Auto measures the image and fixes what is actually wrong with it. The dials are "
            "there when you want to take over.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        T.accent_button(body, "✦  Auto enhance", self._auto, width=330,
                        height=38).pack(anchor="w", pady=(0, 4))
        self.auto_note = T.hint(body, "Press Auto, or set the dials yourself and press Apply.")
        self.auto_note.pack(anchor="w", fill="x", pady=(0, 8))

        self.d_denoise = SliderRow(body, "Remove noise", 0.0, 1.0, 0.0, "{:.0%}", width=175)
        self.d_denoise.pack(fill="x", pady=4)
        self.d_deblur = SliderRow(body, "Remove blur", 0.0, 1.0, 0.0, "{:.0%}", width=175)
        self.d_deblur.pack(fill="x", pady=4)
        self.d_sharpen = SliderRow(body, "Sharpen", 0.0, 1.0, 0.0, "{:.0%}", width=175)
        self.d_sharpen.pack(fill="x", pady=4)
        self.d_contrast = SliderRow(body, "Contrast", 0.0, 1.0, 0.0, "{:.0%}", width=175)
        self.d_contrast.pack(fill="x", pady=4)

        self.speckle = T.checkbox(body, "Noise is speckly (dots), not grainy")
        self.speckle.pack(anchor="w", pady=(6, 0))
        self.notch = T.checkbox(body, "Remove stripe pattern (notch the spectral peaks)")
        self.notch.pack(anchor="w", pady=(6, 0))

        bar = self.button_bar(body)
        T.accent_button(bar, "Apply", self._apply_dials, width=100).pack(side="left")
        T.ghost_button(bar, "Reset dials", self._reset_dials,
                       width=110).pack(side="left", padx=6)
        T.ghost_button(bar, "Start over", self._revert, width=100).pack(side="left")

    def _build_quality(self, parent):
        box = T.card(parent)
        box.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(box, "How much better is it?",
                        "PSNR and SSIM against the untouched original. Both rise when the "
                        "repair genuinely recovered the picture.")
        self.quality_box = T.table_box(box, height=250)
        self.quality_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

    def _build_edges(self, parent):
        box, body = self.controls(
            parent, "4 · Find the edges",
            "Edges are where brightness changes quickly, so every detector is a derivative — "
            "a high-pass filter. Shown for both images: noise adds false edges, blur erases "
            "real ones, and a good repair brings back the original's outline.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Detector", text_color=T.MUTED, anchor="w", width=130,
                     font=T.font(11.5)).pack(side="left", padx=(0, 8))
        self.edge_op = ctk.StringVar(value=P.EDGE_OPERATORS[0])
        T.option_menu(row, P.EDGE_OPERATORS, width=200, variable=self.edge_op,
                      command=lambda _v: self._edges_changed()).pack(side="left")

        self.edge_help = T.hint(body, P.EDGE_HELP[P.EDGE_OPERATORS[0]])
        self.edge_help.pack(anchor="w", fill="x", pady=(2, 6))

        self.edge_sens = SliderRow(body, "Sensitivity", 0.0, 1.0, 0.5, "{:.0%}", width=175,
                                   command=lambda _v: self._edges_changed())
        self.edge_sens.pack(fill="x", pady=4)
        self.edge_smooth = T.checkbox(body, "Smooth first (Gaussian low-pass before the "
                                            "derivative)", command=self._edges_changed)
        self.edge_smooth.select()
        self.edge_smooth.pack(anchor="w", pady=(6, 6))

        self.edge_view = ctk.StringVar(value=P.EDGE_VIEWS[0])
        T.segmented(body, P.EDGE_VIEWS, lambda _v: self._edges_changed(),
                    variable=self.edge_view).pack(anchor="w", pady=(4, 0))

        bar = self.button_bar(body)
        T.ghost_button(bar, "Save edge map…", self._save_edges,
                       width=140).pack(side="left")

    def _build_edge_scores(self, parent):
        box = T.card(parent)
        box.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(box, "Edge accuracy",
                        "Edges found in each image, scored against the edges of the untouched "
                        "original with one pixel of tolerance.")
        self.edge_box = T.table_box(box, height=250)
        self.edge_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

    # ----------------------------------------------------------------- source
    def _load(self):
        path = filedialog.askopenfilename(
            title="Open an image", parent=self,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        self.run(lambda: I.resize_max(I.load_image(path), 900),
                 lambda img: self._install(img, os.path.basename(path)),
                 busy="Reading the image…")

    def _demo(self):
        self.run(_demo_scene, lambda img: self._install(img, "demo scene"),
                 busy="Building the demo scene…")

    def _install(self, img, name):
        # The image as loaded is the reference: everything from here is a degradation whose
        # recovery can be measured against it.
        self.session.set_image(img, name, reference=img)
        self._reset_dials()
        self.status(f"Loaded {name}. Damage it, then make it clear again.", "OK")

    def _save(self):
        if not self.require_image():
            return
        path = filedialog.asksaveasfilename(title="Save the restored image", parent=self,
                                            defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        img = self.session.image_current
        self.run(lambda: I.save_image(path, img),
                 lambda _r: self.status(f"Saved {os.path.basename(path)}.", "OK"),
                 busy="Writing the image…")

    # ----------------------------------------------------------------- damage
    def _degrade(self):
        if not self.require_image():
            return
        base = self.session.image_reference
        if base is None:
            base = self.session.image_original
        blur, noise, pattern = self.blur.get(), self.noise.get(), self.pattern.get()

        def work():
            return P.degrade_image(base, blur, noise, seed=7, pattern=pattern)

        def done(result):
            damaged, psf = result
            self._notched = []
            self.session.set_image_input(damaged, psf)
            self.session.add_step("Image damage",
                                  f"blur {blur:.0%}, noise {noise:.0%}, stripes {pattern:.0%}"
                                  + (" (blur kernel recorded)" if psf is not None else ""))
            self._reset_dials()
            self.status("Damaged the image — now press Auto enhance to repair it.", "OK")

        self.run(work, done, busy="Damaging the image…")

    def _undo_damage(self):
        base = self.session.image_reference
        if base is None:
            self.status("There is no untouched original to go back to.", "WARN")
            return
        self.session.set_image_input(base.copy(), None)
        self._reset_dials()
        self.status("Restored the untouched original.", "OK")

    def _revert(self):
        if self.require_image():
            self.session.revert_image()
            self._reset_dials()

    # ----------------------------------------------------------------- repair
    def _auto(self):
        if not self.require_image():
            return
        src = self.session.image_original
        psf = getattr(self.session, "image_psf", None)

        def work():
            return P.auto_enhance(src, psf), P.find_periodic_peaks(src)

        def done(result):
            (out, steps), peaks = result
            self._notched = peaks
            self.session.set_image_current(out)
            self.session.add_step("Image auto-enhance", "; ".join(steps))
            self.auto_note.configure(text="Auto did: " + "; ".join(steps) + ".")
            self.status("Auto enhance finished — see the scores on the right.", "OK")

        self.run(work, done, busy="Measuring the image and repairing it…")

    def _apply_dials(self):
        if not self.require_image():
            return
        src = self.session.image_original
        psf = getattr(self.session, "image_psf", None)
        dials = (self.d_denoise.get(), self.d_deblur.get(), self.d_sharpen.get(),
                 self.d_contrast.get())
        speckle = bool(self.speckle.get())
        notch = bool(self.notch.get())

        def work():
            peaks = P.find_periodic_peaks(src) if notch else []
            return P.restore_image(src, *dials, psf=psf, salt_pepper=speckle,
                                   notch=notch), peaks

        def done(result):
            (out, desc), peaks = result
            self._notched = peaks
            self.session.set_image_current(out)
            self.session.add_step("Image restore", desc)
            self.auto_note.configure(text="Applied: " + desc + ".")
            self.status(f"Applied {desc}.", "OK")

        self.run(work, done, busy="Repairing the image…")

    def _reset_dials(self):
        for row in (self.d_denoise, self.d_deblur, self.d_sharpen, self.d_contrast):
            row.set(0.0)

    # ----------------------------------------------------------------- drawing
    def refresh(self):
        s = self.session
        if not s.has_image:
            self.card_before.clear()
            self.card_after.clear()
            self.hist.message("Load an image, or press Demo scene")
            T.write_lines(self.condition, [("No image loaded.", "MUTED")])
            T.write_lines(self.quality_box, [("No image loaded.", "MUTED")])
            self.spectrum.message("Load an image to see its 2-D spectrum")
            self.edge_before.clear()
            self.edge_after.clear()
            T.write_lines(self.edge_box, [("No image loaded.", "MUTED")])
            return

        before, after = s.image_original, s.image_current
        if np.array_equal(before, after):
            self._notched = []
        self.card_before.show(before, _shape(before))
        self.card_after.show(after, _shape(after))
        self._draw_condition(after)
        self._draw_quality(before, after)
        self._draw_histogram(before, after)
        self._draw_spectrum(before, after)
        self._detect_edges()

    # ----------------------------------------------------------------- edges
    def _edge_settings(self) -> P.EdgeSettings:
        return P.EdgeSettings(self.edge_op.get(), self.edge_sens.get(),
                              bool(self.edge_smooth.get()), self.edge_view.get())

    def _edges_changed(self):
        self.edge_help.configure(text=P.EDGE_HELP.get(self.edge_op.get(), ""))
        if self._edge_pending is not None:
            self.after_cancel(self._edge_pending)
        self._edge_pending = self.after(220, self._detect_edges)

    def _detect_edges(self):
        """Run the detector on both images (and the reference) off the Tk thread.

        A slider drag can queue several runs; each carries a job number and only the newest
        is drawn, so a slow early result can never overwrite a later one.
        """
        self._edge_pending = None
        s = self.session
        if not s.has_image:
            return
        settings = self._edge_settings()
        before, after, ref = s.image_original, s.image_current, s.image_reference
        has_ref = ref is not None and not np.array_equal(ref, before)
        self._edge_job += 1
        job = self._edge_job

        def work():
            e_before = settings.detect(before)
            e_after = settings.detect(after)
            e_ref = settings.detect(ref) if has_ref else None
            return (settings, e_before, e_after, e_ref,
                    settings.render(before, e_before), settings.render(after, e_after))

        def done(result):
            if job != self._edge_job:
                return
            self._edge_result = result
            settings, e_before, e_after, e_ref, show_before, show_after = result
            self.session.image_edges = settings
            self.edge_before.show(show_before,
                                  f"{settings.operator} · {P.edge_density(e_before):.1f}% edge")
            self.edge_after.show(show_after,
                                 f"{settings.operator} · {P.edge_density(e_after):.1f}% edge")
            self._draw_edge_scores(settings, e_before, e_after, e_ref)

        self.run(work, done)

    def _draw_edge_scores(self, settings, e_before, e_after, e_ref):
        rows = [(settings.describe(), "HEAD"), ("", None),
                (f"{'Edge pixels':<12}{P.edge_density(e_before):>9.2f}% → "
                 f"{P.edge_density(e_after):>6.2f}%", None)]
        if e_ref is None:
            rows += [("", None),
                     ("Damage the image first to score the edges against a known original.",
                      "MUTED")]
            T.write_lines(self.edge_box, rows)
            return
        rows.append((f"{'Original':<12}{P.edge_density(e_ref):>9.2f}%", "MUTED"))
        rows += [("", None), ("AGREEMENT WITH THE ORIGINAL'S EDGES", "HEAD")]
        m_before, m_after = P.edge_match(e_ref, e_before), P.edge_match(e_ref, e_after)
        for key, meaning in (("Precision", "found edges that are real"),
                             ("Recall", "real edges that were found"),
                             ("F1", "overall")):
            b, a = m_before[key], m_after[key]
            rows.append((f"{key:<12}{b:>9.1%} → {a:>6.1%}   {meaning}",
                         "GOOD" if a > b + 0.005 else ("WARN" if a < b - 0.005 else None)))
        rows.append(("", None))
        if m_before["Precision"] < 0.8:
            rows.append(("Low precision before: noise is being detected as edges.", "MUTED"))
        if m_before["Recall"] < 0.8:
            rows.append(("Low recall before: blur has washed real edges out.", "MUTED"))
        T.write_lines(self.edge_box, rows)

    def _save_edges(self):
        if not self.require_image() or self._edge_result is None:
            return
        path = filedialog.asksaveasfilename(title="Save the edge map", parent=self,
                                            defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        image = self._edge_result[5]
        self.run(lambda: I.save_image(path, image),
                 lambda _r: self.status(f"Saved {os.path.basename(path)}.", "OK"),
                 busy="Writing the edge map…")

    def _draw_condition(self, img):
        report = P.analyse_image(img)
        rows = [("CONDITION OF THE CURRENT IMAGE", "HEAD")]
        for key, value, verdict, good in (
                ("Sharpness", report["Sharpness"], report["Sharpness verdict"],
                 report["Sharpness verdict"] in ("acceptable", "sharp")),
                ("Noise", report["Noise (sigma)"], report["Noise verdict"],
                 report["Noise verdict"] in ("clean", "slightly noisy")),
                ("Contrast", report["Contrast (std)"], report["Contrast verdict"],
                 report["Contrast verdict"] == "good"),
                ("Brightness", report["Brightness"], report["Brightness verdict"],
                 report["Brightness verdict"] == "balanced")):
            rows.append((f"{key:<12}{value:>8.1f}   {verdict}", "GOOD" if good else "WARN"))
        T.write_lines(self.condition, rows)

    def _draw_quality(self, before, after):
        s = self.session
        reference = s.image_reference
        rows = []
        if reference is not None and not np.array_equal(reference, before):
            damaged = I.compare(reference, before)
            repaired = I.compare(reference, after)
            rows.append(("MEASURED AGAINST THE UNTOUCHED ORIGINAL", "HEAD"))
            for key in ("PSNR (dB)", "SSIM", "MSE"):
                b, a = damaged[key], repaired[key]
                better = (a < b) if key == "MSE" else (a > b)
                verdict = "better" if better else ("same" if abs(a - b) < 1e-9 else "worse")
                rows.append((f"{key:<12}{b:>10.3f} → {a:>9.3f}   {verdict}",
                             "GOOD" if better else "WARN"))
            gain = repaired["PSNR (dB)"] - damaged["PSNR (dB)"]
            rows.append(("", None))
            rows.append((f"Recovered {gain:+.2f} dB of PSNR", "GOOD" if gain > 0 else "BAD"))
        else:
            rows.append(("COMPARED WITH THE INPUT IMAGE", "HEAD"))
            rows.extend(metric_lines(I.compare(before, after), 14))
            rows.append(("", None))
            rows.append(("Damage the image first to score a repair against a known original.",
                         "MUTED"))
        rows.append(("", None))
        rows.append((f"{'Sharpness':<12}{P.estimate_sharpness(before):>10.2f} → "
                     f"{P.estimate_sharpness(after):>9.2f}", None))
        rows.append((f"{'Noise sigma':<12}{P.estimate_noise(before):>10.2f} → "
                     f"{P.estimate_noise(after):>9.2f}", None))
        T.write_lines(self.quality_box, rows)

    def _draw_histogram(self, before, after):
        self.hist.reset_axes()
        ax = self.hist.axes()
        b_centres, b_counts = I.histogram(before)
        a_centres, a_counts = I.histogram(after)
        ax.fill_between(b_centres, b_counts, color=T.blend(T.FAINT, T.PANEL_DEEP, 0.55),
                        label="before")
        ax.plot(a_centres, a_counts, color=T.ACCENT, lw=1.1, label="after")
        ax.set_xlabel("Brightness (0 = black, 255 = white)")
        ax.set_ylabel("Pixels")
        ax.set_xlim(0, 255)
        ax.legend(loc="upper right", fontsize=7)
        self.hist.draw()

    def _draw_spectrum(self, before, after):
        self.spectrum.reset_axes()
        spectra = (I.fft2_spectrum(before), I.fft2_spectrum(after))
        # One shared colour scale, so a darker region after processing really is less energy.
        lo = min(float(np.percentile(sp, 1)) for sp in spectra)
        hi = max(float(sp.max()) for sp in spectra)
        h, w = spectra[0].shape
        for i, (spec, name) in enumerate(zip(spectra, ("Before", "After"))):
            ax = self.spectrum.axes(i)
            ax.imshow(spec, cmap=T.CMAP_SPECTRO, vmin=lo, vmax=hi, aspect="equal",
                      extent=(-w / 2, w / 2, h / 2, -h / 2))
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("u (cycles across the width)")
            if i == 0:
                ax.set_ylabel("v", rotation=0, labelpad=8)
            for py, px, *_ in self._notched:
                ax.scatter([px - w / 2], [py - h / 2], s=140, facecolors="none",
                           edgecolors=T.TEAL, linewidths=1.2)
        self.spectrum.set_status(f"{len(self._notched)} interference peaks removed"
                                 if self._notched else "")
        self.spectrum.draw()


def _shape(img) -> str:
    h, w = img.shape[:2]
    return f"{w} × {h}"


def _demo_scene(size: int = 420) -> np.ndarray:
    """A synthetic photograph: smooth sky, hard-edged buildings, fine texture and thin bars.

    Chosen so every kind of damage shows somewhere - blur destroys the thin bars and the
    windows first, noise is most visible in the flat sky, and low contrast flattens the
    gradient - which makes a before/after comparison readable at a glance.
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w, 3))

    sky = np.clip(215 - yy / h * 90, 0, 255)
    img[..., 0] = sky * 0.65
    img[..., 1] = sky * 0.80
    img[..., 2] = sky

    horizon = int(h * 0.62)
    sun_y, sun_x, sun_r = int(h * 0.22), int(w * 0.72), int(size * 0.075)
    img[((yy - sun_y) ** 2 + (xx - sun_x) ** 2) < sun_r ** 2] = (252, 236, 180)

    ground = yy >= horizon
    texture = 12.0 * np.sin(xx / 2.2) * np.sin(yy / 2.6)
    for c, base in enumerate((78, 96, 62)):
        img[..., c] = np.where(ground, np.clip(base + texture, 0, 255), img[..., c])

    for left, width, height, tone in ((0.06, 0.16, 0.30, 118), (0.24, 0.12, 0.22, 92),
                                      (0.38, 0.19, 0.38, 136), (0.60, 0.14, 0.26, 104)):
        x0, x1 = int(left * w), int((left + width) * w)
        y0 = horizon - int(height * h)
        img[y0:horizon, x0:x1] = tone
        for wy in range(y0 + 8, horizon - 8, 16):
            for wx in range(x0 + 7, x1 - 7, 14):
                img[wy:wy + 7, wx:wx + 7] = (236, 224, 170)

    bar_y = int(h * 0.90)
    for i, gap in enumerate(range(2, 9)):
        x0 = int(w * 0.08) + i * 44
        img[bar_y:bar_y + 16, x0:x0 + gap] = 245

    return np.clip(img, 0, 255)
