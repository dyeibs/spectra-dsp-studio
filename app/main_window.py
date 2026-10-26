"""
The SPECTRA main window: sidebar, header, page stack and footer.

Threading model
---------------
  * The Tk main loop owns every widget.
  * Heavy DSP runs on short-lived worker threads started by `run_task`.
  * Workers never touch widgets.  They put ("done", token, result) or ("error", token, exc)
    on `ui_queue`, which the main loop drains every 120 ms in `_drain_queue`.

That indirection is the whole reason the window stays responsive while a 30-second clip is
being deconvolved, and why a failure in a worker surfaces as an error dialog instead of a
silent dead thread.
"""
from __future__ import annotations

import datetime as dt
import logging
import queue
import threading
import traceback
from tkinter import messagebox

import customtkinter as ctk

from . import theme as T
from .compat import patch_customtkinter
from .state import Session
from .widgets import Chip, NavItem, PageStack

log = logging.getLogger("spectra.app")

APP_NAME = "SPECTRA — DSP Studio"
APP_FULL = ("Signal Processing, Enhancement, Characterisation,\n"
            "Transform & Restoration Application")
APP_TAG = "DIGITAL SIGNAL PROCESSING  ·  FINAL PROJECT"


class SpectraApp(ctk.CTk):
    def __init__(self):
        # Must happen before any widget exists - see app/compat.py for what it works around.
        patch_customtkinter()
        super().__init__()
        ctk.set_appearance_mode("dark")
        T.apply_mpl_style()

        self.title(APP_NAME)
        self.geometry("1480x920")
        self.minsize(1120, 720)
        self.configure(fg_color=T.BG)

        self.session = Session()
        self.ui_queue: queue.Queue = queue.Queue()
        self._tasks: dict = {}
        self._token = 0
        self._busy = 0
        self.pages: dict = {}
        self._page_classes: dict = {}
        self.nav_buttons: dict = {}
        self._show_job = None

        self._build_ui()
        self.session.subscribe(self._on_session_change)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.after(120, self._drain_queue)
        self.after(1000, self._clock_tick)
        self.session.log_event("SPECTRA started", "OK")

    # =============================================================== layout
    def _build_ui(self):
        shell = ctk.CTkFrame(self, fg_color=T.BG, corner_radius=0)
        shell.pack(fill="both", expand=True)
        self._build_sidebar(shell)

        main = ctk.CTkFrame(shell, fg_color=T.BG, corner_radius=0)
        main.pack(side="left", fill="both", expand=True, padx=(18, 18))
        self._build_header(main)

        self.stack = PageStack(main)
        self.stack.pack(fill="both", expand=True, pady=(0, 12))
        self._build_pages()
        self._build_footer()
        self.navigate(next(iter(self._page_classes)))

    def _build_header(self, main):
        bar = ctk.CTkFrame(main, fg_color="transparent", height=66)
        bar.pack(fill="x", pady=(14, 8))
        titles = ctk.CTkFrame(bar, fg_color="transparent")
        titles.pack(side="left")
        self.page_title = ctk.CTkLabel(titles, text="Dashboard", text_color=T.TEXT, height=30,
                                       anchor="w", font=T.font(23, "bold"))
        self.page_title.pack(anchor="w")
        self.page_sub = ctk.CTkLabel(titles, text="", text_color=T.MUTED, height=18, anchor="w",
                                     font=T.font(11.5))
        self.page_sub.pack(anchor="w")

        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right")
        self.busy_chip = Chip(right, "● READY", T.GREEN, width=116)
        self.busy_chip.pack(side="right")
        self.clock_label = ctk.CTkLabel(right, text="", text_color=T.MUTED,
                                        font=T.font(13, family=T.NUM_FONT))
        self.clock_label.pack(side="right", padx=(0, 16))

    def _build_sidebar(self, shell):
        sb = ctk.CTkFrame(shell, fg_color=T.SIDEBAR, width=244, corner_radius=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        ctk.CTkFrame(shell, width=1, fg_color=T.BORDER, corner_radius=0).pack(side="left",
                                                                             fill="y")
        ctk.CTkLabel(sb, text="◉", text_color=T.ACCENT,
                     font=T.font(40, "bold")).pack(pady=(20, 0))
        ctk.CTkLabel(sb, text="S P E C T R A", text_color=T.TEXT, height=26,
                     font=T.font(19, "bold", T.NUM_FONT)).pack()
        ctk.CTkLabel(sb, text=APP_FULL, text_color=T.FAINT, justify="center", wraplength=214,
                     font=T.font(9)).pack(pady=(2, 0))
        ctk.CTkLabel(sb, text=APP_TAG, text_color=T.ACCENT, height=18,
                     font=T.font(9, "bold")).pack(pady=(5, 14))

        self.nav_holder = ctk.CTkFrame(sb, fg_color="transparent")
        self.nav_holder.pack(fill="x")

        box = T.card(sb)
        box.pack(fill="x", padx=12, pady=(14, 0))
        ctk.CTkLabel(box, text="SESSION", text_color=T.FAINT, height=16,
                     font=T.font(9.5, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
        self.session_label = ctk.CTkLabel(box, text="No signal loaded", text_color=T.MUTED,
                                          height=16, anchor="w", font=T.font(10),
                                          wraplength=196, justify="left")
        self.session_label.pack(anchor="w", padx=12, pady=(0, 6))
        T.ghost_button(box, "Revert to original", self._revert, width=194,
                       height=28).pack(padx=10, pady=(0, 4))
        T.ghost_button(box, "Promote processed", self._promote, width=194,
                       height=28).pack(padx=10, pady=(0, 10))

        foot = ctk.CTkFrame(sb, fg_color="transparent")
        foot.pack(side="bottom", fill="x", padx=12, pady=(0, 14))
        self.audio_chip = Chip(foot, "", T.MUTED, width=194)
        self.audio_chip.pack(fill="x")
        self._refresh_audio_chip()

    def _build_pages(self):
        """Instantiate the pages in sidebar order.

        A page that fails to construct is replaced by a visible placeholder rather than being
        allowed to abort start-up - one broken page should not cost the whole application.
        """
        from .pages.about import AboutPage
        from .pages.analysis import AnalysisPage
        from .pages.audio import AudioStudioPage
        from .pages.dashboard import DashboardPage
        from .pages.image import ImagePage
        from .pages.report import ReportPage
        from .pages.video import VideoPage

        # Pages are constructed on first visit, not here.  Building all nine up front creates
        # roughly twenty-five matplotlib figures and draws every one of them before the window
        # is first painted, which costs several seconds of apparent hang at start-up.
        for cls in (DashboardPage, AudioStudioPage, AnalysisPage, ImagePage, VideoPage,
                    ReportPage, AboutPage):
            self._page_classes[cls.nav_name] = (cls, self.stack.add(cls.nav_name))
            item = NavItem(self.nav_holder, cls.icon, cls.title,
                           lambda n=cls.nav_name: self.navigate(n))
            item.pack(fill="x", padx=12, pady=1)
            self.nav_buttons[cls.nav_name] = item

    def page(self, name: str):
        """Return a page by name, building it if this is its first use.

        Pages are constructed lazily, so `self.pages` only holds the ones already opened;
        anything reaching for a page should come through here.
        """
        return self._ensure_page(name)

    def _ensure_page(self, name: str):
        """Build a page the first time it is opened; replace it with a visible placeholder if
        it cannot be built, so one broken page never costs the whole application."""
        if name in self.pages:
            return self.pages[name]
        cls, holder = self._page_classes[name]
        # Building a page is synchronous (it creates Tk widgets, which only the main thread
        # may do), so say what is happening and force a repaint before the slow part starts.
        self.busy_chip.set("● BUILDING", T.ORANGE)
        self.footer_status.configure(text=f"Preparing the {name} page…", text_color=T.MUTED)
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            page = cls(holder, self)
            page.pack(fill="both", expand=True)
        except Exception:
            log.exception("Could not build the %s page", name)
            page = self._placeholder(holder, name, traceback.format_exc())
        finally:
            self.busy_chip.set("● READY", T.GREEN)
            self.configure(cursor="")
            self.footer_status.configure(text="Ready.", text_color=T.MUTED)
        self.pages[name] = page
        return page

    def _placeholder(self, holder, name, detail):
        frame = ctk.CTkFrame(holder, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        card = T.card(frame)
        card.pack(fill="both", expand=True)
        T.section_title(card, f"{name} unavailable",
                        "This page failed to build. The rest of the application is unaffected.")
        box = T.table_box(card)
        box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        T.write_lines(box, [(line, "BAD") for line in detail.splitlines()[-25:]])
        frame.title, frame.subtitle = name, "Unavailable"
        frame.on_session = lambda _topic: None
        frame.on_show = lambda: None
        return frame

    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=T.SIDEBAR, corner_radius=0, height=34)
        footer.pack(fill="x", side="bottom", before=self.winfo_children()[0])
        footer.pack_propagate(False)
        ctk.CTkFrame(self, height=1, fg_color=T.BORDER,
                     corner_radius=0).pack(fill="x", side="bottom", before=footer)
        self.footer_status = ctk.CTkLabel(footer, text="Ready.", text_color=T.MUTED,
                                          font=T.font(10.5, "bold"), anchor="w")
        self.footer_status.pack(side="left", padx=16, fill="x", expand=True)
        self.footer_right = ctk.CTkLabel(footer, text="", text_color=T.FAINT, font=T.font(10))
        self.footer_right.pack(side="right", padx=16)

    # =============================================================== navigation
    def navigate(self, name: str):
        if name not in self._page_classes:
            return
        page = self._ensure_page(name)
        self.stack.set(name)
        self.page_title.configure(text=getattr(page, "title", name))
        self.page_sub.configure(text=getattr(page, "subtitle", ""))
        for key, item in self.nav_buttons.items():
            item.set_active(key == name)
        # Redraw a moment later, not now.  Tk only paints the page just swapped in once this
        # handler returns, so a synchronous redraw of its plots (a second or more on a long
        # recording) left the page half-painted - controls missing, titles as black boxes -
        # until it finished.  Deferring lets the page appear complete first.
        if self._show_job is not None:
            self.after_cancel(self._show_job)
        self._show_job = self.after(30, lambda: self._show_page(name))

    def _show_page(self, name: str):
        self._show_job = None
        page = self.pages.get(name)
        if page is None or self.stack.get() != name:
            return                       # the user has already moved on to another page
        try:
            page.on_show()
        except Exception:
            log.exception("on_show failed for %s", name)

    # =============================================================== tasks
    def run_task(self, fn, on_done=None, busy: str | None = None, on_error=None):
        """Run `fn()` on a worker thread; deliver the result on the Tk thread."""
        self._token += 1
        token = self._token
        self._tasks[token] = (on_done, on_error)
        self._set_busy(True, busy)

        def worker():
            try:
                self.ui_queue.put(("done", token, fn()))
            except Exception as exc:
                log.exception("Background task failed")
                self.ui_queue.put(("error", token, exc))

        threading.Thread(target=worker, daemon=True, name=f"spectra-task-{token}").start()

    def _drain_queue(self):
        """Deliver completed task results.  Bounded per tick so a burst of finished tasks
        cannot stall the redraw."""
        try:
            for _ in range(24):
                kind, token, payload = self.ui_queue.get_nowait()
                on_done, on_error = self._tasks.pop(token, (None, None))
                self._set_busy(False)
                if kind == "done":
                    if on_done:
                        try:
                            on_done(payload)
                        except Exception as exc:
                            log.exception("A completion callback failed")
                            self.show_error("Display error", str(exc))
                else:
                    if on_error:
                        on_error(payload)
                    else:
                        self.show_error("Processing failed", str(payload))
        except queue.Empty:
            pass
        finally:
            self.after(120, self._drain_queue)

    def _set_busy(self, busy: bool, message: str | None = None):
        self._busy = max(self._busy + (1 if busy else -1), 0)
        if self._busy:
            self.busy_chip.set("● WORKING", T.ORANGE)
            if message:
                self.set_status(message, "INFO")
            self.configure(cursor="watch")
        else:
            self.busy_chip.set("● READY", T.GREEN)
            self.configure(cursor="")

    # =============================================================== status
    def set_status(self, text: str, level: str = "INFO"):
        colour = {"OK": T.GREEN, "WARN": T.ORANGE, "ERROR": T.RED}.get(level, T.MUTED)
        self.footer_status.configure(text=text, text_color=colour)
        if level in ("WARN", "ERROR", "OK"):
            self.session.log_event(text, level)

    def show_error(self, title: str, message: str):
        self.set_status(f"{title}: {message}", "ERROR")
        messagebox.showerror(title, message, parent=self)

    # =============================================================== session
    def _on_session_change(self, topic):
        if topic in ("signal", "image", "video"):
            self._refresh_session_label()
        for page in list(self.pages.values()):
            try:
                page.on_session(topic)
            except Exception:
                log.exception("on_session failed for %s", getattr(page, "title", "?"))

    def _refresh_session_label(self):
        s = self.session
        bits = []
        if s.has_signal:
            bits.append(f"Signal · {s.current.duration:.2f}s @ {s.current.fs:,.0f} Hz")
        if s.has_image:
            h, w = s.image_current.shape[:2]
            bits.append(f"Image · {w}×{h}")
        if s.has_video:
            bits.append(f"Video · {s.video_info.get('Frames', '?')} frames")
        self.session_label.configure(text="\n".join(bits) if bits else "No signal loaded")
        self.footer_right.configure(text=f"{len(s.steps)} processing step(s)")

    def _revert(self):
        if self.session.has_signal:
            self.session.revert()
            self.set_status("Reverted to the original signal.", "OK")
        else:
            self.set_status("Nothing to revert - no signal is loaded.", "WARN")

    def _promote(self):
        if self.session.has_signal:
            self.session.commit_current_as_original()
            self.set_status("The processed signal is now the session source.", "OK")
        else:
            self.set_status("Nothing to promote - no signal is loaded.", "WARN")

    def _refresh_audio_chip(self):
        from dsp import audio_io
        if audio_io.has_output():
            self.audio_chip.set("● AUDIO DEVICE READY", T.GREEN)
        else:
            self.audio_chip.set("● NO AUDIO OUTPUT", T.ORANGE)

    # =============================================================== misc
    def _clock_tick(self):
        self.clock_label.configure(text=dt.datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._clock_tick)

    def close_app(self):
        try:
            from dsp import audio_io
            audio_io.Player().stop()
        except Exception:
            pass
        self.destroy()
