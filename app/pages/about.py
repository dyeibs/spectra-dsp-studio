"""
About Us - who built this, and a short account of what it does.

Also doubles as the opening slide for the defence: the panel on the right states the project
in the examiner's own terms, so the first page they see already answers "what is it".
"""
from __future__ import annotations

import customtkinter as ctk

from .. import theme as T
from .base import Page

GROUP = "GROUP 6"
SECTION = "CPE07A"
SUBJECT = "DIGISIG"
PROFESSOR = "Engr. Rodante Santos"

MEMBERS = (("Leader", "Elideros, John Ver D."),
           ("Member", "Banal, Ryan P."),
           ("Member", "Batac, Jayson V."),
           ("Member", "Manuyag, Darwin F."))

WHAT_IT_DOES = (
    ("Audio Studio",
     "Load, record or generate a signal, filter it, and remove noise from it — with the "
     "before and after shown on the same axes."),
    ("Audio Analysis",
     "The frequency content of the signal and how it changes over time, plus the pitch and "
     "MFCC features a speech recogniser would use."),
    ("Image Restore",
     "Takes a blurry, grainy or flat image and makes it clear again, then measures the "
     "recovery with PSNR and SSIM against the untouched original. Shows the 2-D spectrum, "
     "notches out stripe patterns, and finds edges with seven detectors."),
    ("Video Restore",
     "The same idea across time: noise is independent between frames while the picture is "
     "not, so averaging each still pixel over time removes grain without smearing motion. "
     "Edge detection runs "
     "on the cleaned frames and can be exported as a clip."),
    ("Report & Export",
     "Everything measured during the session, written out as CSV, PDF, WAV or PNG."),
)

TECHNIQUES = (
    "Sampling, quantisation and aliasing",
    "FIR and IIR filter design (5 IIR families, 4 band types)",
    "DFT, power spectral density and the STFT spectrogram",
    "13 noise-reduction methods — spectral, adaptive and wavelet",
    "MFCC speech features and autocorrelation pitch tracking",
    "2-D filtering and Wiener deconvolution for image restoration",
    "2-D FFT and frequency-domain notch filtering of periodic interference",
    "Edge detection: Sobel, Prewitt, Scharr, Roberts, Laplacian, LoG, Canny",
    "Objective quality metrics: SNR, segmental SNR, PSNR, SSIM, THD",
)


class AboutPage(Page):
    title = "About Us"
    subtitle = "The team behind SPECTRA"
    icon = ""
    nav_name = "About Us"
    topics = ()                     # nothing on this page depends on the session

    def build(self):
        page = self.scroll()

        row = self.row(page)
        self._build_team(row)
        self._build_project(row)

        lower = self.row(page)
        self._build_techniques(lower)
        self._build_stack(lower)

    # ----------------------------------------------------------------- panels
    def _build_team(self, parent):
        card = T.card(parent)
        card.pack(side="left", fill="both", expand=True, padx=(0, 5))

        ctk.CTkLabel(card, text=GROUP, text_color=T.TEAL,
                     font=T.font(34, "bold", T.NUM_FONT)).pack(pady=(26, 4))
        ctk.CTkLabel(card, text="DIGITAL SIGNAL PROCESSING  ·  FINAL PROJECT",
                     text_color=T.ACCENT, font=T.font(10, "bold")).pack(pady=(0, 18))

        for role, name in MEMBERS:
            line = ctk.CTkFrame(card, fg_color="transparent")
            line.pack(fill="x", padx=28, pady=5)
            ctk.CTkLabel(line, text=role, text_color=T.ACCENT if role == "Leader" else T.FAINT,
                         width=62, anchor="w",
                         font=T.font(11, "bold" if role == "Leader" else "normal")).pack(
                side="left")
            ctk.CTkLabel(line, text=name, text_color=T.TEXT, anchor="w",
                         font=T.font(14, "bold" if role == "Leader" else "normal")).pack(
                side="left")

        T.divider(card, pady=(20, 14), padx=28)

        for label, value in (("Section", SECTION), ("Subject", SUBJECT),
                             ("Professor", PROFESSOR)):
            line = ctk.CTkFrame(card, fg_color="transparent")
            line.pack(fill="x", padx=28, pady=4)
            ctk.CTkLabel(line, text=label, text_color=T.FAINT, width=72, anchor="w",
                         font=T.font(11)).pack(side="left")
            ctk.CTkLabel(line, text=value, text_color=T.TEAL, anchor="w",
                         font=T.font(13, "bold")).pack(side="left")
        ctk.CTkLabel(card, text="", height=18).pack()

    def _build_project(self, parent):
        card = T.card(parent)
        card.pack(side="left", fill="both", expand=True, padx=(5, 0))

        ctk.CTkLabel(card, text="S P E C T R A", text_color=T.TEXT,
                     font=T.font(26, "bold", T.NUM_FONT)).pack(pady=(26, 2))
        ctk.CTkLabel(card, text="Signal Processing, Enhancement, Characterisation,\n"
                               "Transform & Restoration Application",
                     text_color=T.MUTED, justify="center",
                     font=T.font(11)).pack(pady=(0, 16))

        intro = T.hint(card,
                       "An interactive workbench for audio, speech, image and video signals. "
                       "Every page does the same job in a different domain: take a signal "
                       "that has something wrong with it, fix it, and measure how much "
                       "better it got.", color=T.TEXT, size=12)
        intro.pack(anchor="w", fill="x", padx=24, pady=(0, 16))

        for name, text in WHAT_IT_DOES:
            line = ctk.CTkFrame(card, fg_color="transparent")
            line.pack(fill="x", padx=24, pady=3)
            ctk.CTkLabel(line, text=name, text_color=T.ACCENT, width=118, anchor="nw",
                         font=T.font(11, "bold")).pack(side="left")
            T.hint(line, text).pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(card, text="", height=18).pack()

    def _build_techniques(self, parent):
        card = T.card(parent)
        card.pack(side="left", fill="both", expand=True, padx=(0, 5))
        T.section_title(card, "DSP techniques implemented",
                        "All written on numpy and scipy — no black-box signal processing "
                        "library was used for the algorithms themselves.")
        for item in TECHNIQUES:
            line = ctk.CTkFrame(card, fg_color="transparent")
            line.pack(fill="x", padx=22, pady=2)
            ctk.CTkLabel(line, text="▸", text_color=T.ACCENT, width=16,
                         font=T.font(11)).pack(side="left")
            T.hint(line, item, color=T.TEXT, size=11.5).pack(side="left", fill="x",
                                                             expand=True)
        ctk.CTkLabel(card, text="", height=14).pack()

    def _build_stack(self, parent):
        card = T.card(parent)
        card.pack(side="left", fill="both", expand=True, padx=(5, 0))
        T.section_title(card, "How to check it works",
                        "The processing is verified against ground truth, not just run.")
        box = T.table_box(card, height=210)
        box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        T.write_lines(box, [
            ("python validate.py", "HEAD"),
            ("", None),
            ("157 automated checks that compare the output of every", None),
            ("algorithm against a known correct answer, for example:", None),
            ("", None),
            ("  a generated tone really is at the frequency asked for", "GOOD"),
            ("  noise lands at the requested SNR within 0.1 dB", "GOOD"),
            ("  quantisation SNR matches 6.02B + 1.76 dB", "GOOD"),
            ("  a 3 kHz tone decimated by 4 folds to exactly 1 kHz", "GOOD"),
            ("  every denoiser improves the noise it is built for", "GOOD"),
            ("  image restoration raises both PSNR and SSIM", "GOOD"),
            ("  every edge detector traces a known outline exactly", "GOOD"),
            ("", None),
            ("Built with Python, numpy, scipy, OpenCV, matplotlib", "MUTED"),
            ("and CustomTkinter.", "MUTED"),
        ])
