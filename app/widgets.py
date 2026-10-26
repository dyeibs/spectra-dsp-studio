"""
Reusable SPECTRA widgets.

The pieces every page is assembled from: KPI tiles, the sidebar navigation entry, the page
stack, labelled parameter controls and - the important one - `PlotCard`, which embeds a
matplotlib figure inside a CustomTkinter panel and owns the redraw lifecycle.
"""
from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from . import theme as T


class PageStack(ctk.CTkFrame):
    """A tab view with no tab strip - the sidebar is the tab strip.

    CTkTabview reserves a header row even when its buttons are hidden, which leaves a gap
    under the page title; this keeps the same add/set/get API without that gap.
    """

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._pages, self._current = {}, None

    def add(self, name):
        page = ctk.CTkFrame(self, fg_color="transparent")
        self._pages[name] = page
        if self._current is None:
            self.set(name)
        return page

    def set(self, name):
        if name not in self._pages or name == self._current:
            if name == self._current:
                return
            raise KeyError(name)
        if self._current:
            self._pages[self._current].pack_forget()
        self._pages[name].pack(fill="both", expand=True)
        self._current = name

    def get(self):
        return self._current


class NavItem(ctk.CTkFrame):
    """Sidebar entry: red indicator bar + icon + label, with hover and active states."""

    def __init__(self, master, icon, text, command):
        super().__init__(master, fg_color="transparent", corner_radius=9, height=38)
        self.pack_propagate(False)
        self.command, self.active = command, False
        self.bar = ctk.CTkFrame(self, width=3, height=18, corner_radius=2, fg_color=T.SIDEBAR)
        self.bar.pack(side="left", padx=(4, 0))
        self.icon = ctk.CTkLabel(self, text=icon, width=28, text_color=T.MUTED,
                                 font=T.font(14, family=T.ICON_FONT))
        self.icon.pack(side="left", padx=(8, 4))
        self.label = ctk.CTkLabel(self, text=text, anchor="w", text_color=T.MUTED,
                                  font=T.font(12.5))
        self.label.pack(side="left", fill="x", expand=True)
        for w in (self, self.icon, self.label, self.bar):
            w.bind("<Button-1>", lambda _e: self.command())
            w.bind("<Enter>", lambda _e: self._hover(True))
            w.bind("<Leave>", lambda _e: self.after(15, self._check_leave))
            w.configure(cursor="hand2")

    def _hover(self, on):
        if not self.active:
            self.configure(fg_color=T.PANEL_HOVER if on else "transparent")
            self.label.configure(text_color=T.TEXT if on else T.MUTED)

    def _check_leave(self):
        """Tk fires <Leave> when the pointer crosses onto a *child*, which would flicker the
        hover state; walk up from the widget under the cursor to see if we really left."""
        try:
            widget = self.winfo_containing(*self.winfo_pointerxy())
        except (KeyError, tk.TclError):
            widget = None
        while widget is not None:
            if widget is self:
                return
            widget = widget.master
        self._hover(False)

    def set_active(self, active):
        self.active = active
        self.configure(fg_color=T.blend(T.ACCENT, T.SIDEBAR, .14) if active else "transparent")
        self.bar.configure(fg_color=T.ACCENT if active else T.SIDEBAR)
        self.icon.configure(text_color=T.ACCENT if active else T.MUTED)
        self.label.configure(text_color=T.TEXT if active else T.MUTED,
                             font=T.font(12.5, "bold" if active else "normal"))


class MetricCard(ctk.CTkFrame):
    """KPI tile: coloured accent stripe, small caps title, large value, detail line."""

    def __init__(self, master, title, icon="", **kwargs):
        super().__init__(master, fg_color=T.PANEL, corner_radius=14, border_width=1,
                         border_color=T.BORDER, **kwargs)
        self.stripe = ctk.CTkFrame(self, width=4, height=40, corner_radius=2, fg_color=T.FAINT)
        self.stripe.pack(side="left", fill="y", padx=(10, 0), pady=14)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side="left", fill="both", expand=True, padx=(12, 12), pady=(10, 10))
        head = ctk.CTkFrame(body, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text=title.upper(), text_color=T.MUTED, height=16,
                     font=T.font(10.5, "bold")).pack(side="left")
        if icon:
            ctk.CTkLabel(head, text=icon, text_color=T.FAINT, height=16,
                         font=T.font(12, family=T.ICON_FONT)).pack(side="right")
        self.value = ctk.CTkLabel(body, text="—", text_color=T.TEXT, height=32, anchor="w",
                                  font=T.font(24, "bold", T.NUM_FONT))
        self.value.pack(fill="x")
        self.detail = ctk.CTkLabel(body, text="No signal loaded", text_color=T.MUTED, height=15,
                                   anchor="w", font=T.font(10.5))
        self.detail.pack(fill="x")

    def set(self, value, detail="", color=T.ACCENT):
        self.value.configure(text=value, text_color=color)
        self.detail.configure(text=detail)
        self.stripe.configure(fg_color=color)

    def clear(self):
        self.set("—", "No signal loaded", T.FAINT)


class PlotCard(ctk.CTkFrame):
    """A matplotlib figure embedded in a SPECTRA panel.

    The figure is created once and its axes are cleared and redrawn on each update, rather
    than rebuilding the canvas - recreating a FigureCanvasTkAgg per update leaks Tk widgets
    and visibly flickers.  Use `axes()` to get cleared axes, then call `draw()`.
    """

    def __init__(self, master, title="", subtitle="", height=240, nrows=1, ncols=1,
                 toolbar=False, **kw):
        super().__init__(master, fg_color=T.PANEL, corner_radius=14, border_width=1,
                         border_color=T.BORDER, **kw)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if title:
            head = ctk.CTkFrame(self, fg_color="transparent")
            head.pack(fill="x", padx=16, pady=(12, 0))
            ctk.CTkLabel(head, text=title.upper(), text_color=T.ACCENT,
                         font=T.font(11, "bold")).pack(side="left")
            self.status = ctk.CTkLabel(head, text="", text_color=T.MUTED, font=T.font(10))
            self.status.pack(side="right")
            if subtitle:
                lbl = ctk.CTkLabel(self, text=subtitle, text_color=T.MUTED, font=T.font(10.5),
                                   justify="left", anchor="w")
                lbl.pack(anchor="w", fill="x", padx=16, pady=(2, 0))
                T.autowrap(lbl, self, pad=44)
        else:
            self.status = None

        self.figure = Figure(figsize=(6, height / 100.0), dpi=100, facecolor=T.PANEL_DEEP)
        self.nrows, self.ncols = nrows, ncols
        self._axes = self.figure.subplots(nrows, ncols, squeeze=False)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=T.PANEL_DEEP, highlightthickness=0, bd=0)
        widget.pack(fill="both", expand=True, padx=12, pady=(8, 12 if not toolbar else 0))

        self.toolbar = None
        if toolbar:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            bar = ctk.CTkFrame(self, fg_color=T.PANEL_ALT, corner_radius=8, height=30)
            bar.pack(fill="x", padx=12, pady=(4, 10))
            self.toolbar = NavigationToolbar2Tk(self.canvas, bar, pack_toolbar=False)
            self.toolbar.configure(background=T.PANEL_ALT)
            for child in self.toolbar.winfo_children():
                try:
                    child.configure(background=T.PANEL_ALT, foreground=T.TEXT)
                except tk.TclError:
                    pass
            self.toolbar.update()
            self.toolbar.pack(side="left")

    def axes(self, index: int = 0):
        """Return cleared axes `index` in row-major order."""
        ax = self._axes[index // self.ncols][index % self.ncols]
        ax.clear()
        return ax

    def all_axes(self):
        return [self._axes[r][c] for r in range(self.nrows) for c in range(self.ncols)]

    def clear(self):
        for ax in self.all_axes():
            ax.clear()

    def draw(self, tight: bool = True):
        if tight:
            try:
                self.figure.tight_layout(pad=1.1)
            except Exception:
                pass
        self.canvas.draw_idle()

    def set_status(self, text, color=T.MUTED):
        if self.status is not None:
            self.status.configure(text=text, text_color=color)

    def message(self, text="No data yet"):
        """Replace the plot with a centred hint - the empty state before anything is loaded."""
        self.clear()
        ax = self._axes[0][0]
        for a in self.all_axes():
            a.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", color=T.FAINT, fontsize=10,
                transform=ax.transAxes)
        self.draw(tight=False)

    def reset_axes(self):
        for a in self.all_axes():
            a.set_axis_on()


class ParamRow(ctk.CTkFrame):
    """A labelled control on one line: caption on the left, widget on the right.

    Keeps every parameter panel in the application aligned without each page re-inventing
    its own grid.
    """

    def __init__(self, master, caption, widget_factory, hint=None, caption_width=130):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=caption, text_color=T.MUTED, anchor="w", width=caption_width,
                     font=T.font(11.5)).pack(side="left", padx=(0, 8))
        self.widget = widget_factory(self)
        self.widget.pack(side="left")
        self.value_label = None
        if hint:
            self.value_label = ctk.CTkLabel(self, text=hint, text_color=T.FAINT,
                                            font=T.font(10.5))
            self.value_label.pack(side="left", padx=(8, 0))

    def set_hint(self, text, color=T.FAINT):
        if self.value_label is not None:
            self.value_label.configure(text=text, text_color=color)


class SliderRow(ctk.CTkFrame):
    """A slider with a live numeric readout - the standard control for continuous parameters."""

    def __init__(self, master, caption, from_, to, value, fmt="{:.2f}", steps=None,
                 command=None, caption_width=130, width=160):
        super().__init__(master, fg_color="transparent")
        self.fmt, self._command = fmt, command
        ctk.CTkLabel(self, text=caption, text_color=T.MUTED, anchor="w", width=caption_width,
                     font=T.font(11.5)).pack(side="left", padx=(0, 8))
        self.slider = T.slider(self, from_, to, command=self._on_change,
                               number_of_steps=steps, width=width)
        self.slider.set(value)
        self.slider.pack(side="left")
        self.readout = ctk.CTkLabel(self, text=fmt.format(value), text_color=T.TEXT, width=62,
                                    anchor="e", font=T.font(11, family=T.NUM_FONT))
        self.readout.pack(side="left", padx=(8, 0))

    def _on_change(self, value):
        self.readout.configure(text=self.fmt.format(value))
        if self._command:
            self._command(value)

    def get(self) -> float:
        return float(self.slider.get())

    def set(self, value):
        self.slider.set(value)
        self.readout.configure(text=self.fmt.format(value))


class Chip(ctk.CTkLabel):
    """A small rounded status pill."""

    def __init__(self, master, text="", color=T.MUTED, width=120, **kw):
        super().__init__(master, text=text, text_color=color, fg_color=T.PANEL_ALT,
                         corner_radius=13, height=26, width=width,
                         font=T.font(10.5, "bold"), **kw)

    def set(self, text, color=T.MUTED, fill=T.PANEL_ALT):
        self.configure(text=text, text_color=color, fg_color=fill)


class ImageCard(ctk.CTkFrame):
    """Displays a numpy image, scaled to fit, with a caption strip.

    Rescaling happens on every `show`, against the widget's current size, so the pair of
    before/after panels stay matched when the window is resized.
    """

    def __init__(self, master, title="", height=260, **kw):
        super().__init__(master, fg_color=T.PANEL, corner_radius=14, border_width=1,
                         border_color=T.BORDER, **kw)
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(10, 0))
        self.title_label = ctk.CTkLabel(head, text=title.upper(), text_color=T.ACCENT,
                                        font=T.font(11, "bold"))
        self.title_label.pack(side="left")
        self.info = ctk.CTkLabel(head, text="", text_color=T.MUTED, font=T.font(10))
        self.info.pack(side="right")
        self.holder = ctk.CTkLabel(self, text="No image", text_color=T.FAINT,
                                   font=T.font(11), fg_color=T.PANEL_DEEP, corner_radius=10)
        self.holder.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        self._height = height
        self._ctk_image = None

    def show(self, array, info=""):
        """Render a float/uint8 array (greyscale or RGB)."""
        import numpy as np
        from PIL import Image

        if array is None:
            self.clear()
            return
        arr = np.clip(np.nan_to_num(np.asarray(array, dtype=np.float64)), 0, 255).astype("uint8")
        pil = Image.fromarray(arr if arr.ndim == 3 else arr, mode="RGB" if arr.ndim == 3 else "L")

        self.update_idletasks()
        box_w = max(self.holder.winfo_width() - 8, 160)
        box_h = max(self.holder.winfo_height() - 8, self._height - 60)
        scale = min(box_w / pil.width, box_h / pil.height)
        size = (max(int(pil.width * scale), 1), max(int(pil.height * scale), 1))

        self._ctk_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
        self.holder.configure(image=self._ctk_image, text="")
        self.info.configure(text=info)

    def clear(self):
        self._ctk_image = None
        self.holder.configure(image=None, text="No image")
        self.info.configure(text="")

    def set_title(self, text):
        self.title_label.configure(text=text.upper())


def metric_lines(data: dict, width: int = 30):
    """Format a metric dict as aligned (text, tag) rows for `theme.write_lines`.

    Values are coloured by plausibility, not by threshold: a non-finite result means the
    metric did not apply to this signal, and saying so is more useful than printing 'nan'.
    """
    import math
    out = []
    for key, value in data.items():
        if isinstance(value, float):
            if math.isnan(value):
                text, tag = "n/a", "MUTED"
            elif math.isinf(value):
                text, tag = ("∞" if value > 0 else "-∞"), "GOOD"
            else:
                text, tag = (f"{value:,.4g}" if abs(value) < 1e5 else f"{value:,.3e}"), None
        else:
            text, tag = str(value), None
        out.append((f"{key:<{width}} {text}", tag))
    return out
