"""
Workarounds for third-party library bugs.

Kept in one place, and applied explicitly at start-up rather than on import, so it is obvious
what is being patched and why.
"""
from __future__ import annotations

import logging

log = logging.getLogger("spectra.compat")


def _noop():
    """Stand-in for `update_idletasks` while a CustomTkinter widget is redrawing."""


def patch_customtkinter() -> None:
    """Stop CustomTkinter's scrollbar and option menu from pumping the Tk event loop.

    `CTkScrollbar._draw` and `CTkOptionMenu._draw` both end with `self._canvas
    .update_idletasks()`, which runs every pending idle callback from inside a redraw.  In a
    window that also holds a matplotlib canvas, those pending callbacks include the canvas's
    `idle_draw`; drawing the figure re-lays out the scrolled area, which makes it re-issue
    `yscrollcommand`, which calls `CTkScrollbar.set`, which calls `_draw` again.  The two
    re-enter each other without bound and the process locks up before the window is ever
    mapped - `python main.py` simply hangs with no error.

    Dropping the call is safe: it only forces an immediate repaint of a canvas that Tk would
    paint on the next idle cycle anyway.  A CTkTextbox builds its own CTkScrollbar, so the
    fix has to be here rather than in the choice of scrolling container.

    Must be called before any widget is created.  Safe to call more than once.
    """
    try:
        from customtkinter.windows.widgets.ctk_optionmenu import CTkOptionMenu
        from customtkinter.windows.widgets.ctk_scrollbar import CTkScrollbar
    except ImportError:                  # an unfamiliar layout - leave the library alone
        log.info("CustomTkinter internals not found; skipping the redraw patch")
        return

    for cls in (CTkScrollbar, CTkOptionMenu):
        if getattr(cls, "_spectra_patched", False):
            continue
        original = cls._draw

        def _draw(self, no_color_updates=False, _original=original):
            canvas = getattr(self, "_canvas", None)
            if canvas is None:
                return _original(self, no_color_updates)
            canvas.update_idletasks = _noop      # shadow the bound method for this call only
            try:
                return _original(self, no_color_updates)
            finally:
                try:
                    del canvas.update_idletasks
                except AttributeError:
                    pass                          # a nested redraw already restored it

        cls._draw = _draw
        cls._spectra_patched = True
    log.debug("CustomTkinter redraw patch applied")
