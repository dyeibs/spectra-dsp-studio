from __future__ import annotations

import logging
import os
import sys
import traceback

LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-18s %(message)s"


def configure_logging() -> None:
    """Log to the console and to logs/spectra.log.

    The file matters during a demonstration: if something misbehaves in front of an audience
    the traceback is already on disk rather than lost with the console window.
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "spectra.log"),
                                            encoding="utf-8"))
    except OSError:
        pass                                  # a read-only folder must not stop the app
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def check_dependencies() -> list:
    """Return the import names of any missing third-party packages."""
    required = {"numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
                "customtkinter": "customtkinter", "PIL": "Pillow", "cv2": "opencv-python",
                "soundfile": "soundfile", "sounddevice": "sounddevice", "pywt": "PyWavelets",
                "reportlab": "reportlab"}
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def main() -> int:
    configure_logging()
    log = logging.getLogger("spectra")

    missing = check_dependencies()
    if missing:
        print("SPECTRA cannot start - these packages are missing:\n")
        for package in missing:
            print(f"    {package}")
        print("\nInstall them with:\n    pip install " + " ".join(missing))
        return 1

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app.main_window import SpectraApp

    try:
        app = SpectraApp()
    except Exception:
        log.critical("SPECTRA failed to start:\n%s", traceback.format_exc())
        return 2

    log.info("SPECTRA ready")
    app.mainloop()
    log.info("SPECTRA closed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
