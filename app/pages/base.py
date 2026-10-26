"""
Base class for SPECTRA pages.

Gives every page the same contract - build once, react to session changes, refresh when
shown - plus the helpers for running DSP off the Tk thread and reporting errors.

The threading rule is absolute: a worker thread must never touch a Tk widget.  `run` takes
the heavy function and a completion callback, runs the first on a worker and the second back
on the Tk main loop via the application's queue.  Anything that forgets this will appear to
work and then crash intermittently under load, so every long operation goes through it.
"""
from __future__ import annotations

import customtkinter as ctk

from .. import theme as T


class Page(ctk.CTkFrame):
    title = "Page"
    subtitle = ""
    icon = ""
    nav_name = "Page"

    #: Session topics this page cares about; a change in any other topic is ignored.
    topics = ("signal",)

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.session = app.session
        self._dirty = False
        self._built = False
        self.build()
        self._built = True

    # ------------------------------------------------------------- hooks
    def build(self) -> None:
        """Construct the widgets.  Called once, from __init__."""

    def refresh(self) -> None:
        """Redraw this page from the session.  Subclasses implement this."""

    @property
    def is_visible(self) -> bool:
        return self.app.stack.get() == self.nav_name

    def on_session(self, topic: str) -> None:
        """A session topic changed.

        Only the visible page redraws.  Rendering every page on every change meant a single
        filter application re-ran nine pages' worth of spectrograms, MFCCs and image
        transforms on the Tk thread - measured at nearly twenty seconds of frozen UI. Hidden
        pages are marked dirty instead and catch up in `on_show`.
        """
        if topic not in self.topics:
            return
        if self.is_visible:
            self._dirty = False
            self.refresh()
        else:
            self._dirty = True

    def on_show(self) -> None:
        """This page just became visible - redraw if anything changed while it was hidden."""
        if self._dirty or not self._built:
            self._dirty = False
        self.refresh()

    # ------------------------------------------------------------- helpers
    def run(self, fn, on_done=None, busy: str | None = None, on_error=None):
        """Run `fn()` on a worker thread, then `on_done(result)` on the Tk thread."""
        self.app.run_task(fn, on_done=on_done, busy=busy, on_error=on_error)

    def status(self, text: str, level: str = "INFO") -> None:
        self.app.set_status(text, level)

    def toast_error(self, title: str, message: str) -> None:
        self.app.show_error(title, message)

    def require_signal(self) -> bool:
        """Guard for actions that need a loaded signal; reports why if there is none."""
        if not self.session.has_signal:
            self.status("Load or generate a signal first — see the Audio Studio page.", "WARN")
            return False
        return True

    def require_image(self) -> bool:
        if not self.session.has_image:
            self.status("Load an image first — see the Image Restore page.", "WARN")
            return False
        return True

    def require_video(self) -> bool:
        if not self.session.has_video:
            self.status("Load a video first — see the Video Restore page.", "WARN")
            return False
        return True

    # ------------------------------------------------------------- layout sugar
    def scroll(self):
        """A scrollable page body.  Returns the frame to pack content into."""
        scroller = T.scroll_frame(self)
        scroller.pack(fill="both", expand=True)
        self.scroller = scroller
        return scroller

    @staticmethod
    def row(master, pady=(0, 10)):
        frame = ctk.CTkFrame(master, fg_color="transparent")
        frame.pack(fill="x", pady=pady)
        return frame

    @staticmethod
    def controls(master, title, sub=None):
        """A parameter panel: titled card with an inner body frame to pack controls into."""
        box = T.card(master)
        T.section_title(box, title, sub)
        body = ctk.CTkFrame(box, fg_color="transparent")
        body.pack(fill="x", padx=18, pady=(0, 14))
        return box, body

    @staticmethod
    def button_bar(master, pady=(10, 0)):
        bar = ctk.CTkFrame(master, fg_color="transparent")
        bar.pack(fill="x", pady=pady)
        return bar
