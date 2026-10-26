"""
SPECTRA visual identity - the graphite + red palette, fonts and widget factories.

Shared with the VIGILANT monitoring system so the two applications read as one family: near
black panels, a single red accent reserved for the active element and for emphasis, and
muted grey for everything secondary.  Every colour the application uses is named here, so a
restyle is a one-file change.

Matplotlib is styled to the same palette in `apply_mpl_style`, which is what keeps the
embedded plots from looking like they were pasted in from a different program.
"""
from __future__ import annotations

import customtkinter as ctk

# ----------------------------------------------------------------- palette
BG = "#09090b"
PANEL = "#131316"
PANEL_ALT = "#1c1c21"
PANEL_DEEP = "#0e0e11"
PANEL_HOVER = "#232329"
SIDEBAR = "#0d0d10"
BORDER = "#26262c"
BORDER_STRONG = "#34343c"
TEXT = "#f4f4f5"
MUTED = "#9a9aa5"
FAINT = "#5c5c66"
ACCENT = "#e5323e"
ACCENT_DARK = "#8f1720"
ACCENT_MID = "#c4232f"
ACCENT_HOVER = "#e5323e"
GREY = "#6b6b75"
TRACK = "#22222a"

GREEN = "#2ecc71"
YELLOW = "#f4c430"
ORANGE = "#ff8c1a"
RED = "#df2531"
TEAL = "#1fd1a5"
BLUE = "#7f8cff"
PURPLE = "#c07bff"

# Series colours for multi-trace plots: the accent first (it carries the eye), then hues that
# stay distinguishable against near-black and from each other.
SERIES = (ACCENT, TEAL, BLUE, YELLOW, PURPLE, ORANGE, GREEN, "#c9c9d1")

# Fonts - Windows ships all of these; Tk falls back silently elsewhere.
UI_FONT = "Segoe UI"
NUM_FONT = "Bahnschrift"
MONO_FONT = "Cascadia Mono"
ICON_FONT = "Segoe Fluent Icons"


def blend(color: str, bg: str, alpha: float) -> str:
    """Mix two #rrggbb colours.  Tk has no alpha channel, so every tint, glow and fill in the
    application is pre-computed against the background it will sit on."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * alpha + y * (1 - alpha)):02x}" for x, y in zip(a, b))


def font(size=12, weight="normal", family=UI_FONT):
    """A themed font.  Tk only accepts an integer point size, so fractional sizes used at the
    call sites for fine typographic tuning are rounded here rather than at each one."""
    return ctk.CTkFont(family=family, size=int(round(size)), weight=weight)


# ----------------------------------------------------------------- widget factories
def card(master, **kw):
    """The standard content container: panel fill, hairline border, generous radius."""
    kw.setdefault("fg_color", PANEL)
    kw.setdefault("corner_radius", 14)
    kw.setdefault("border_width", 1)
    kw.setdefault("border_color", BORDER)
    return ctk.CTkFrame(master, **kw)


def subcard(master, **kw):
    kw.setdefault("fg_color", PANEL_DEEP)
    kw.setdefault("corner_radius", 12)
    kw.setdefault("border_width", 1)
    kw.setdefault("border_color", BORDER)
    return ctk.CTkFrame(master, **kw)


def accent_button(master, text, command, width=130, height=32, **kw):
    kw.setdefault("font", font(12, "bold"))
    return ctk.CTkButton(master, text=text, command=command, width=width, height=height,
                         corner_radius=8, fg_color=ACCENT_MID, hover_color=ACCENT_HOVER, **kw)


def ghost_button(master, text, command, width=110, height=32, **kw):
    """Secondary action: dark, outlined, lifts on hover."""
    kw.setdefault("font", font(12))
    return ctk.CTkButton(master, text=text, command=command, width=width, height=height,
                         corner_radius=8, fg_color=PANEL_ALT, hover_color=PANEL_HOVER,
                         border_width=1, border_color=BORDER_STRONG, text_color=TEXT, **kw)


def option_menu(master, values, width=150, command=None, variable=None):
    return ctk.CTkOptionMenu(master, values=list(values), width=width, command=command,
                             variable=variable, fg_color=PANEL_ALT, button_color=PANEL_HOVER,
                             button_hover_color=ACCENT_MID, corner_radius=8,
                             dropdown_fg_color=PANEL, dropdown_hover_color=ACCENT_DARK,
                             text_color=TEXT, font=font(12))


def segmented(master, values, command, variable=None):
    return ctk.CTkSegmentedButton(master, values=list(values), command=command,
                                  variable=variable, corner_radius=8, height=30,
                                  fg_color=PANEL_ALT, selected_color=ACCENT_MID,
                                  selected_hover_color=ACCENT_HOVER, unselected_color=PANEL_ALT,
                                  unselected_hover_color=PANEL_HOVER, text_color=TEXT,
                                  font=font(12, "bold"))


def entry(master, width=110, placeholder="", **kw):
    return ctk.CTkEntry(master, width=width, height=30, corner_radius=8, fg_color=PANEL_ALT,
                        border_color=BORDER_STRONG, border_width=1, text_color=TEXT,
                        placeholder_text=placeholder, font=font(12), **kw)


def switch(master, text, command=None, **kw):
    return ctk.CTkSwitch(master, text=text, command=command, progress_color=ACCENT_MID,
                         button_color=TEXT, button_hover_color=TEXT, fg_color=BORDER_STRONG,
                         text_color=TEXT, font=font(12), **kw)


def slider(master, from_, to, command=None, number_of_steps=None, width=170):
    return ctk.CTkSlider(master, from_=from_, to=to, command=command, width=width,
                         number_of_steps=number_of_steps, progress_color=ACCENT_MID,
                         button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                         fg_color=TRACK, height=16)


def checkbox(master, text, command=None, **kw):
    return ctk.CTkCheckBox(master, text=text, command=command, fg_color=ACCENT_MID,
                           hover_color=ACCENT_HOVER, border_color=BORDER_STRONG,
                           text_color=TEXT, font=font(12), corner_radius=5,
                           checkbox_width=18, checkbox_height=18, **kw)


def progress(master, width=200):
    return ctk.CTkProgressBar(master, width=width, height=6, corner_radius=3,
                              progress_color=ACCENT, fg_color=TRACK)


def label(master, text, color=TEXT, size=12, weight="normal", family=UI_FONT, **kw):
    return ctk.CTkLabel(master, text=text, text_color=color, font=font(size, weight, family), **kw)


def autowrap(label, container=None, pad=44):
    """Keep a label's wraplength matched to the width of its container.

    A fixed wraplength only looks right at one card width.  Cards here share a row and so
    size themselves to the window, which left long explanations clipped in the narrow ones;
    re-wrapping on <Configure> makes the text fit whatever width the card actually gets.
    """
    target = container if container is not None else label.master

    def _on_configure(event):
        label.configure(wraplength=max(event.width - pad, 140))

    target.bind("<Configure>", _on_configure, add="+")
    return label


def hint(master, text, color=FAINT, size=10, **kw):
    """A small muted explanatory line that re-wraps with its container."""
    lbl = ctk.CTkLabel(master, text=text, text_color=color, font=font(size), justify="left",
                       anchor="w", **kw)
    return autowrap(lbl, master)


def section_title(master, text, sub=None, pady=(16, 8)):
    """Card heading: small red caps title with an optional muted explanation underneath."""
    ctk.CTkLabel(master, text=text.upper(), text_color=ACCENT, font=font(12, "bold")).pack(
        anchor="w", padx=18, pady=(pady[0], 0 if sub else pady[1]))
    if sub:
        lbl = ctk.CTkLabel(master, text=sub, text_color=MUTED, font=font(11), justify="left",
                           anchor="w")
        lbl.pack(anchor="w", fill="x", padx=18, pady=(2, pady[1]))
        autowrap(lbl, master, pad=48)


def divider(master, pady=(8, 8), padx=16):
    ctk.CTkFrame(master, height=1, fg_color=BORDER).pack(fill="x", padx=padx, pady=pady)


def table_box(master, size=11, **kw):
    """Monospace read-only table in a bordered panel - used for metrics and logs."""
    box = ctk.CTkTextbox(master, fg_color=PANEL_DEEP, text_color=TEXT, font=(MONO_FONT, size),
                         wrap="none", border_width=1, border_color=BORDER, corner_radius=12,
                         scrollbar_button_color=BORDER_STRONG,
                         scrollbar_button_hover_color=FAINT, **kw)
    try:
        box._textbox.configure(spacing1=2, spacing3=2, padx=8, pady=6)
    except Exception:
        pass
    box.tag_config("HEAD", foreground=ACCENT)
    box.tag_config("GOOD", foreground=GREEN)
    box.tag_config("WARN", foreground=ORANGE)
    box.tag_config("BAD", foreground=RED)
    box.tag_config("MUTED", foreground=MUTED)
    box.tag_config("RULE", foreground=BORDER_STRONG)
    return box


def write_lines(box, lines) -> None:
    """Replace a textbox's contents with (text, tag) pairs or plain strings."""
    box.configure(state="normal")
    box.delete("1.0", "end")
    for item in lines:
        text, tag = item if isinstance(item, tuple) else (item, None)
        box.insert("end", text + "\n", tag or ())
    box.configure(state="disabled")


def scroll_frame(master, **kw):
    """The standard scrollable page body.

    Safe only because `app.compat.patch_customtkinter` has removed the `update_idletasks()`
    call from CTkScrollbar's redraw; without that the scrollbar and an embedded matplotlib
    canvas re-enter each other forever.
    """
    kw.setdefault("fg_color", "transparent")
    return ctk.CTkScrollableFrame(master, scrollbar_button_color=BORDER_STRONG,
                                  scrollbar_button_hover_color=FAINT, **kw)


# ----------------------------------------------------------------- matplotlib
def apply_mpl_style() -> None:
    """Push the palette into matplotlib's rcParams.

    Called once at start-up, before any figure exists, so every embedded plot inherits the
    dark panel background, hairline spines and muted tick labels without per-figure styling.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import rcParams, cycler

    rcParams.update({
        "figure.facecolor": PANEL_DEEP,
        "figure.edgecolor": PANEL_DEEP,
        "savefig.facecolor": PANEL_DEEP,
        "savefig.edgecolor": PANEL_DEEP,
        "axes.facecolor": PANEL_DEEP,
        "axes.edgecolor": BORDER_STRONG,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler("color", list(SERIES)),
        "grid.color": TRACK,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": TEXT,
        "font.family": "sans-serif",
        "font.sans-serif": [UI_FONT, "DejaVu Sans"],
        "font.size": 8.5,
        "legend.facecolor": PANEL_ALT,
        "legend.edgecolor": BORDER,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.95,
        "legend.labelcolor": TEXT,
        "lines.linewidth": 1.2,
        "lines.antialiased": True,
        "figure.autolayout": False,
        "image.cmap": "magma",
    })


# Colormaps chosen to sit well on a near-black panel: perceptually uniform, dark at the low
# end so the background and the data floor agree.
CMAP_SPECTRO = "magma"
CMAP_IMAGE = "gray"
CMAP_DIVERGING = "RdBu_r"
