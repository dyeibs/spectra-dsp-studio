"""
Video Restore - clean up a grainy, soft clip and write it back out.

The same dials as the image page, plus the one tool a video has that a still image does not:
temporal filtering.  Noise is independent from frame to frame while the scene is not, so
averaging over time removes noise at no cost in spatial detail - something no 2-D filter can
do.  It is folded into the denoise dial and is motion-adaptive: each pixel averages only while
it stays still, so the static scene comes out clean and moving objects do not smear.

Preview a single frame before committing, then export the whole clip on a worker thread with
progress and a cancel that works.

Edge detection runs *after* cleaning, on purpose: a derivative is a high-pass filter, so on a
grainy frame it finds the grain.  Cleaning first is the difference between an outline and
static, and the frame condition panel counts the edge pixels to show it.
"""
from __future__ import annotations

import os
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

from dsp import imaging as I, presets as P, video as V
from .. import theme as T
from ..widgets import ImageCard, PlotCard, SliderRow, metric_lines
from .base import Page

#: What the exported clip and the right-hand preview show.
OUTPUTS = ("Cleaned", "Edge map", "Overlay")


class VideoPage(Page):
    title = "Video Restore"
    subtitle = "Clean up a noisy or soft video clip, frame by frame, then export it"
    icon = ""
    nav_name = "Video Restore"
    topics = ("video",)

    def build(self):
        self._stop_flag = False
        self._frame_cache = None
        self._progress_state = (0, 0)
        self._polling = False
        self._pending = None
        self._motion = None

        page = self.scroll()

        top = self.row(page)
        self._build_source(top)
        self._build_clean(top)

        views = self.row(page)
        self.card_before = ImageCard(views, "Original frame", height=320)
        self.card_before.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.card_after = ImageCard(views, "Cleaned frame", height=320)
        self.card_after.pack(side="left", fill="both", expand=True, padx=(5, 0))

        lower = self.row(page)
        info = T.card(lower)
        info.pack(side="left", fill="both", expand=True, padx=(0, 5))
        T.section_title(info, "Clip details", "Read from the video file.")
        self.info_box = T.table_box(info, height=165)
        self.info_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        quality = T.card(lower)
        quality.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(quality, "Frame condition",
                        "Measured on the previewed frame, before and after cleaning.")
        self.quality_box = T.table_box(quality, height=165)
        self.quality_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        edges = self.row(page)
        self._build_edges(edges)
        self.graphs = PlotCard(
            edges, "Frame and clip signals",
            "Left: brightness distribution of the previewed frame, before and after. Right: "
            "motion energy — mean |frame − previous frame| over time, the whole clip reduced "
            "to a 1-D signal. The marker is the previewed frame.",
            height=250, ncols=2)
        self.graphs.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.refresh()

    # ----------------------------------------------------------------- panels
    def _build_source(self, parent):
        box, body = self.controls(parent, "1 · Pick a clip",
                                  "Load a video, or build a deliberately noisy test clip.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        bar = self.button_bar(body, pady=(0, 8))
        T.accent_button(bar, "Load video…", self._load, width=130).pack(side="left")
        T.ghost_button(bar, "Make test clip", self._make_clip,
                       width=140).pack(side="left", padx=6)

        self.frame_index = SliderRow(body, "Preview frame", 0, 100, 0, "{:.0f}",
                                     command=lambda _v: self._debounce(), width=185)
        self.frame_index.pack(fill="x", pady=4)
        self.max_frames = SliderRow(body, "Export at most", 10, 2000, 300, "{:.0f} frames",
                                    width=185)
        self.max_frames.pack(fill="x", pady=4)

        self.progress = T.progress(body, 340)
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(12, 2))
        self.progress_label = ctk.CTkLabel(body, text="Idle", text_color=T.MUTED,
                                           font=T.font(10.5), anchor="w")
        self.progress_label.pack(anchor="w", fill="x")

    def _build_clean(self, parent):
        box, body = self.controls(
            parent, "2 · Clean it up",
            "Auto reads one frame and sets the dials. Remove noise also averages each pixel "
            "over time while it stays still, so moving objects stay sharp.")
        box.pack(side="left", fill="both", expand=True, padx=(5, 0))

        T.accent_button(body, "✦  Auto clean", self._auto, width=340,
                        height=38).pack(anchor="w", pady=(0, 4))
        self.auto_note = T.hint(body, "Press Auto, or set the dials and press Preview.")
        self.auto_note.pack(anchor="w", fill="x", pady=(0, 8))

        self.d_denoise = SliderRow(body, "Remove noise", 0.0, 1.0, 0.0, "{:.0%}",
                                   command=lambda _v: self._debounce(), width=175)
        self.d_denoise.pack(fill="x", pady=4)
        self.d_deblur = SliderRow(body, "Remove blur", 0.0, 1.0, 0.0, "{:.0%}",
                                  command=lambda _v: self._debounce(), width=175)
        self.d_deblur.pack(fill="x", pady=4)
        self.d_sharpen = SliderRow(body, "Sharpen", 0.0, 1.0, 0.0, "{:.0%}",
                                   command=lambda _v: self._debounce(), width=175)
        self.d_sharpen.pack(fill="x", pady=4)
        self.d_contrast = SliderRow(body, "Contrast", 0.0, 1.0, 0.0, "{:.0%}",
                                    command=lambda _v: self._debounce(), width=175)
        self.d_contrast.pack(fill="x", pady=4)

        bar = self.button_bar(body)
        T.accent_button(bar, "Preview", self._preview, width=100).pack(side="left")
        T.ghost_button(bar, "Export video…", self._export,
                       width=130).pack(side="left", padx=6)
        T.ghost_button(bar, "Cancel", self._cancel, width=90).pack(side="left")

    def _build_edges(self, parent):
        box, body = self.controls(
            parent, "3 · Find the edges",
            "Runs on the cleaned frame, in the preview and in the exported clip. Turn Remove "
            "noise down to zero to see what a derivative makes of raw grain.")
        box.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.output = ctk.StringVar(value=OUTPUTS[0])
        T.segmented(body, OUTPUTS, lambda _v: self._debounce(),
                    variable=self.output).pack(anchor="w", pady=(0, 8))

        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text="Detector", text_color=T.MUTED, anchor="w", width=130,
                     font=T.font(11.5)).pack(side="left", padx=(0, 8))
        self.edge_op = ctk.StringVar(value=P.EDGE_OPERATORS[0])
        T.option_menu(row, P.EDGE_OPERATORS, width=200, variable=self.edge_op,
                      command=lambda _v: self._edge_changed()).pack(side="left")
        self.edge_help = T.hint(body, P.EDGE_HELP[P.EDGE_OPERATORS[0]])
        self.edge_help.pack(anchor="w", fill="x", pady=(2, 6))

        self.edge_sens = SliderRow(body, "Sensitivity", 0.0, 1.0, 0.5, "{:.0%}", width=175,
                                   command=lambda _v: self._debounce())
        self.edge_sens.pack(fill="x", pady=4)
        self.edge_smooth = T.checkbox(body, "Smooth first (Gaussian low-pass)",
                                      command=self._debounce)
        self.edge_smooth.select()
        self.edge_smooth.pack(anchor="w", pady=(6, 0))

    # ----------------------------------------------------------------- helpers
    def _cleaner(self) -> P.VideoCleaner:
        return P.VideoCleaner(denoise=self.d_denoise.get(), deblur=self.d_deblur.get(),
                              sharpen=self.d_sharpen.get(), contrast=self.d_contrast.get())

    def _edges(self) -> P.EdgeSettings:
        """Current edge settings; the view is the chosen output when it is an edge view."""
        view = self.output.get()
        return P.EdgeSettings(self.edge_op.get(), self.edge_sens.get(),
                              bool(self.edge_smooth.get()),
                              view if view in P.EDGE_VIEWS else P.EDGE_VIEWS[0])

    def _edge_changed(self):
        self.edge_help.configure(text=P.EDGE_HELP.get(self.edge_op.get(), ""))
        self._debounce()

    def _debounce(self):
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(280, self._fire)

    def _fire(self):
        self._pending = None
        if self.session.has_video:
            self._preview()

    # ----------------------------------------------------------------- source
    def _load(self):
        path = filedialog.askopenfilename(title="Open a video", parent=self,
                                          filetypes=V.FILE_TYPES)
        if not path:
            return
        self._open(path, "Loaded")

    def _make_clip(self):
        path = filedialog.asksaveasfilename(title="Save the test clip", parent=self,
                                            defaultextension=".mp4",
                                            filetypes=[("MP4", "*.mp4")])
        if not path:
            return

        def work():
            _write_test_clip(path)
            return _probe(path)

        self.run(work, lambda r: self._installed(path, r, "Built a noisy test clip —"),
                 busy="Rendering the test clip…")

    def _open(self, path, verb):
        self.run(lambda: _probe(path),
                 lambda r: self._installed(path, r, f"{verb} {os.path.basename(path)} —"),
                 busy="Opening the video…")

    def _installed(self, path, result, prefix):
        info, frame = result
        self.session.set_video(path, info)
        self.frame_index.slider.configure(to=max(info["Frames"] - 1, 1))
        self.frame_index.set(0)
        self._frame_cache = frame
        self._motion = None
        self.status(f"{prefix} {info['Frames']} frames at "
                    f"{info['Frame rate (fps)']} fps. Press Auto clean.", "OK")
        self._preview()

        def got_motion(result):
            if self.session.video_path == path:
                self._motion = result
                self._draw_motion()
                self.graphs.draw()

        self.run(lambda: V.motion_energy(path, max_frames=600), got_motion)

    # ----------------------------------------------------------------- clean
    def _auto(self):
        if not self.require_video():
            return
        path = self.session.video_path
        index = int(self.frame_index.get())

        def work():
            source = V.VideoSource(path)
            try:
                frame = source.frame_at(index)
            finally:
                source.close()
            if frame is None:
                return None
            return P.auto_video_settings(frame), P.analyse_image(frame)

        def done(result):
            if result is None:
                self.status("Could not read that frame.", "WARN")
                return
            cleaner, report = result
            self.d_denoise.set(cleaner.denoise)
            self.d_deblur.set(cleaner.deblur)
            self.d_sharpen.set(cleaner.sharpen)
            self.d_contrast.set(cleaner.contrast)
            self.auto_note.configure(
                text=f"Auto read frame {index}: noise sigma {report['Noise (sigma)']:.1f} "
                     f"({report['Noise verdict']}) → {cleaner.describe()}.")
            self.status(f"Auto set the dials: {cleaner.describe()}.", "OK")
            self._preview()

        self.run(work, done, busy="Measuring a frame…")

    def _preview(self):
        if not self.session.has_video:
            return
        path = self.session.video_path
        index = int(self.frame_index.get())
        cleaner = self._cleaner()
        window = cleaner.window
        edges = self._edges()
        output = self.output.get()

        def work():
            source = V.VideoSource(path)
            try:
                # Temporal denoising needs the preceding frames to have been seen, or the
                # preview would show the filter's start-up transient instead of its output.
                start = max(index - window + 1, 0)
                out = None
                for _i, frame in source.frames(start=start, limit=index - start + 1):
                    out = cleaner.process(frame)
                original = source.frame_at(index)
            finally:
                source.close()
            if original is None or out is None:
                return None
            e_before, e_after = edges.detect(original), edges.detect(out)
            shown = out if output == "Cleaned" else edges.render(out, e_after)
            desc = cleaner.describe()
            if output != "Cleaned":
                desc = f"{edges.operator} on {desc}"
            return original, out, shown, desc, e_before, e_after

        def done(result):
            if result is None:
                self.status("That frame is past the end of the clip.", "WARN")
                return
            source_frame, cleaned, shown, desc, e_before, e_after = result
            self._frame_cache = source_frame
            self.session.video_preview = {
                "index": index, "frame": source_frame, "cleaned": cleaned,
                "cleaning": cleaner.describe(), "edges": edges,
                "edge before": e_before, "edge after": e_after}
            self.card_before.show(source_frame, f"frame {index}")
            self.card_after.set_title("Edges of cleaned frame" if output != "Cleaned"
                                      else "Cleaned frame")
            self.card_after.show(shown, desc)
            self._draw_quality(source_frame, cleaned, e_before, e_after, edges.operator)
            self._draw_graphs(source_frame, cleaned, index)

        self.run(work, done)

    # ----------------------------------------------------------------- export
    def _export(self):
        if not self.require_video():
            return
        dst = filedialog.asksaveasfilename(title="Export the cleaned video", parent=self,
                                           defaultextension=".mp4",
                                           filetypes=[("MP4", "*.mp4")])
        if not dst:
            return
        src = self.session.video_path
        cleaner = self._cleaner()
        desc = cleaner.describe()
        edges = self._edges() if self.output.get() != "Cleaned" else None
        if edges is not None:
            desc = f"{desc}, then {edges.describe()} ({edges.view.lower()})"
        limit = int(self.max_frames.get())
        self._stop_flag = False

        def report(done_count, total):
            # Runs on the worker thread.  Tk's `after` is not thread-safe, so progress is
            # published to a plain attribute and the main thread picks it up in _poll.
            self._progress_state = (done_count, total)

        def work():
            return _export_clip(src, dst, cleaner, limit, report,
                                lambda: self._stop_flag, edges)

        def done(stats):
            self._polling = False
            self.session.add_step("Video clean-up",
                                  f"{desc} → {os.path.basename(dst)}", {},
                                  {"Frames written": stats["Frames written"]})
            self.progress.set(1.0)
            self.progress_label.configure(
                text=f"Wrote {stats['Frames written']} frames to {os.path.basename(dst)}",
                text_color=T.GREEN)
            self.status(f"Exported {stats['Frames written']} cleaned frames.", "OK")

        def failed(exc):
            self._polling = False
            self.progress_label.configure(text=f"Failed: {exc}", text_color=T.RED)
            self.toast_error("Video export failed", str(exc))

        self._progress_state = (0, limit)
        self._polling = True
        self._poll()
        self.run(work, done, busy=f"Cleaning the clip ({desc})…", on_error=failed)

    def _poll(self):
        if not self._polling:
            return
        done_count, total = self._progress_state
        fraction = done_count / max(total, 1)
        self.progress.set(fraction)
        self.progress_label.configure(text=f"{done_count}/{total} frames ({fraction:.0%})",
                                      text_color=T.MUTED)
        self.after(150, self._poll)

    def _cancel(self):
        self._stop_flag = True
        self.progress_label.configure(text="Cancelling…", text_color=T.ORANGE)

    # ----------------------------------------------------------------- drawing
    def refresh(self):
        if not self.session.has_video:
            self.card_before.clear()
            self.card_after.clear()
            T.write_lines(self.info_box, [("No video loaded.", "MUTED")])
            T.write_lines(self.quality_box, [("No video loaded.", "MUTED")])
            self.graphs.message("Load a video, or make the test clip")
            return
        T.write_lines(self.info_box, metric_lines(self.session.video_info, 20))
        if self._frame_cache is not None:
            self.card_before.show(self._frame_cache, f"frame {int(self.frame_index.get())}")

    def _draw_quality(self, before, after, e_before, e_after, operator):
        a = P.analyse_image(before)
        b = P.analyse_image(after)
        rows = [("                     before      after", "MUTED")]
        for key, fmt in (("Noise (sigma)", "{:8.2f}"), ("Sharpness", "{:8.2f}"),
                         ("Contrast (std)", "{:8.1f}"), ("Brightness", "{:8.1f}")):
            rows.append((f"{key:<16}" + fmt.format(a[key]) + "  →" + fmt.format(b[key]), None))
        rows.append((f"{'Edge pixels %':<16}{P.edge_density(e_before):8.2f}  →"
                     f"{P.edge_density(e_after):8.2f}   ({operator})", None))
        drop = a["Noise (sigma)"] - b["Noise (sigma)"]
        rows.append(("", None))
        rows.append((f"Noise reduced by {drop:.2f} grey levels"
                     if drop > 0.05 else "Noise unchanged",
                     "GOOD" if drop > 0.05 else "MUTED"))
        rows.append((f"Verdict: was {a['Noise verdict']}, now {b['Noise verdict']}", None))
        T.write_lines(self.quality_box, rows)

    def _draw_graphs(self, before, after, index):
        self.graphs.reset_axes()
        ax = self.graphs.axes(0)
        b_centres, b_counts = I.histogram(before)
        a_centres, a_counts = I.histogram(after)
        ax.fill_between(b_centres, b_counts, color=T.blend(T.FAINT, T.PANEL_DEEP, 0.55),
                        label="original")
        ax.plot(a_centres, a_counts, color=T.ACCENT, lw=1.1, label="cleaned")
        ax.set_xlim(0, 255)
        ax.set_xlabel("Brightness")
        ax.set_ylabel("Pixels")
        ax.set_title("Frame histogram", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        self._draw_motion(index)
        self.graphs.draw()

    def _draw_motion(self, index=None):
        ax = self.graphs.axes(1)
        ax.set_axis_on()
        ax.set_title("Motion energy", fontsize=9)
        if self._motion is None or len(self._motion[0]) == 0:
            ax.text(0.5, 0.5, "Measuring the clip…", ha="center", va="center",
                    color=T.FAINT, fontsize=9, transform=ax.transAxes)
            return
        times, energy = self._motion
        ax.plot(times, energy, color=T.TEAL, lw=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean |Δ| (grey levels)")
        if index is None:
            index = int(self.frame_index.get())
        fps = float(self.session.video_info.get("Frame rate (fps)") or 25.0)
        ax.axvline(index / fps, color=T.ACCENT, lw=1.0, ls="--")


# ----------------------------------------------------------------- helpers
def _probe(path):
    """Open a clip, read its properties and its first frame, then close it again."""
    source = V.VideoSource(path)
    try:
        return source.summary(), source.frame_at(0)
    finally:
        source.close()


def _export_clip(src_path, dst_path, cleaner, limit, progress, should_stop, edges=None):
    """Stream the clip through the cleaner (and the edge detector, if one is given) and write
    a new file."""
    import cv2

    source = V.VideoSource(src_path)
    cleaner.reset()
    total = min(source.count, limit)
    writer = None
    written = 0
    try:
        writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), source.fps,
                                 (source.width, source.height))
        if not writer.isOpened():
            raise V.VideoError(f"Could not open {dst_path} for writing")
        for _idx, frame in source.frames(limit=total):
            if should_stop():
                break
            out = cleaner.process(frame)
            if edges is not None:
                out = edges.render(out)
            out = I.to_uint8(out)
            if out.ndim == 2:
                out = np.stack([out] * 3, axis=2)
            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            written += 1
            if written % 5 == 0:
                progress(written, total)
    finally:
        if writer is not None:
            writer.release()
        source.close()
    progress(written, total)
    return {"Frames written": written, "Frames requested": total}


def _test_frame(i: int, frames: int = 90, width: int = 320, height: int = 240,
                _backdrop={}) -> np.ndarray:
    """Clean frame `i` of the test clip: the demo city scene with a ball rolling across it.

    Kept separate from the writer so the temporal filter can be measured against the true
    clean frames rather than judged by eye.
    """
    import cv2
    from .image import _demo_scene

    key = (width, height)
    if key not in _backdrop:
        scene = I.to_uint8(_demo_scene(max(width, height)))
        top = (scene.shape[0] - height) // 2
        _backdrop[key] = scene[top:top + height, :width].astype(np.float64)
    frame = _backdrop[key].copy()
    phase = 2 * np.pi * i / frames
    cx = int(40 + (width - 80) * (0.5 - 0.5 * np.cos(phase)))
    cy = int(height * 0.72 - 30 * abs(np.sin(2 * phase)))          # bouncing along the ground
    cv2.circle(frame, (cx, cy), 16, (225, 60, 52), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx - 5, cy - 5), 5, (250, 170, 160), -1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(frame, (5, 5), 0.9)                     # slightly soft optics


def _write_test_clip(path: str, width: int = 320, height: int = 240, frames: int = 90,
                     fps: int = 24) -> None:
    """Render a noisy, slightly soft clip with a moving subject on a static background.

    Static background plus known motion is exactly the case the temporal filter is built for:
    the city should come out clean while the ball stays sharp instead of smearing.
    """
    import cv2

    rng = np.random.default_rng(4)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise V.VideoError(f"Could not open {path} for writing")
    try:
        for i in range(frames):
            frame = _test_frame(i, frames, width, height)
            frame = frame + rng.normal(0, 14, frame.shape)          # per-frame sensor noise
            writer.write(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8),
                                      cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
