"""
2-D digital signal processing: image enhancement, filtering and restoration.

The same ideas as the 1-D chain, one dimension up - convolution becomes a 2-D kernel, the DFT
becomes the 2-D DFT, and denoising splits into spatial-domain and frequency-domain families.
Restoration (deblurring) is the part with no 1-D counterpart in this project: it inverts a
known degradation rather than merely suppressing an additive one.

Images are carried as float64 arrays in [0, 255], greyscale (H, W) or colour (H, W, 3) in RGB
order.  Keeping them floating point between stages avoids the rounding losses that would
otherwise accumulate over a multi-step pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from . import metrics as _metrics

NOISE_KINDS = ("Gaussian", "Salt & pepper", "Speckle (multiplicative)", "Poisson (shot)",
               "Uniform", "Periodic (sinusoidal)")

SPATIAL_FILTERS = ("Mean (box)", "Gaussian", "Median", "Bilateral", "Non-local means",
                   "Unsharp mask", "Laplacian sharpen", "Min (erosion)", "Max (dilation)")

EDGE_OPERATORS = ("Sobel", "Prewitt", "Scharr", "Laplacian", "Laplacian of Gaussian", "Canny",
                  "Roberts cross")

FREQ_FILTERS = ("Ideal low-pass", "Ideal high-pass", "Butterworth low-pass",
                "Butterworth high-pass", "Gaussian low-pass", "Gaussian high-pass",
                "Band-pass", "Band-stop (notch)")

RESTORATION = ("Wiener deconvolution", "Inverse filter", "Richardson-Lucy",
               "Constrained least squares")

BLUR_KINDS = ("Motion blur", "Gaussian blur", "Defocus (disc)")

HISTOGRAM_OPS = ("Histogram equalisation", "CLAHE (adaptive)", "Gamma correction",
                 "Contrast stretch", "Logarithmic", "Negative")

EPS = 1e-12


# ----------------------------------------------------------------- I/O and conversion
def load_image(path: str, as_gray: bool = False) -> np.ndarray:
    """Read an image as float64 in [0, 255].  cv2.imdecode is used rather than imread so that
    non-ASCII paths (common on Windows user folders) still work."""
    raw = np.fromfile(path, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(raw, flag)
    if img is None:
        raise ValueError(f"Could not read an image from {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64)


def save_image(path: str, img: np.ndarray) -> None:
    out = to_uint8(img)
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, out)
    if not ok:
        raise ValueError(f"Could not encode an image for {path}")
    buf.tofile(path)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Luminance conversion using the ITU-R BT.601 weights, which match perceived brightness
    far better than a plain channel average."""
    if img.ndim == 2:
        return img.astype(np.float64)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float64)


def is_colour(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] == 3


def _per_channel(img: np.ndarray, fn):
    """Apply a greyscale operation to each colour channel independently."""
    if not is_colour(img):
        return fn(img)
    return np.stack([fn(img[..., c]) for c in range(img.shape[2])], axis=2)


def resize_max(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Cap the long edge.  Large photographs make the interactive filters feel sluggish, and
    nothing in the demonstration depends on full resolution."""
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img
    out = cv2.resize(to_uint8(img), (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return out.astype(np.float64)


# ----------------------------------------------------------------- degradation
def add_noise(img: np.ndarray, kind: str = "Gaussian", amount: float = 20.0,
              density: float = 0.05, freq: float = 0.12, seed: int | None = None) -> np.ndarray:
    """Corrupt an image with one of NOISE_KINDS.

    `amount` is the noise standard deviation in grey levels for the additive kinds, the
    multiplicative spread for speckle, and the amplitude for the periodic pattern.
    `density` is the fraction of pixels replaced for salt & pepper.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(img, dtype=np.float64)

    if kind == "Gaussian":
        return x + rng.normal(0.0, amount, x.shape)
    if kind == "Uniform":
        return x + rng.uniform(-amount, amount, x.shape)
    if kind == "Salt & pepper":
        y = x.copy()
        mask = rng.random(x.shape[:2])
        half = density / 2.0
        y[mask < half] = 0.0
        y[mask > 1.0 - half] = 255.0
        return y
    if kind == "Speckle (multiplicative)":
        return x * (1.0 + rng.normal(0.0, amount / 100.0, x.shape))
    if kind == "Poisson (shot)":
        # Scale so `amount` controls the photon count: fewer photons, more shot noise.
        scale = max(255.0 / max(amount, 1e-3), 1e-3)
        return rng.poisson(np.clip(x, 0, None) / 255.0 * scale) / scale * 255.0
    if kind == "Periodic (sinusoidal)":
        h, w = x.shape[:2]
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pattern = amount * np.sin(2 * np.pi * freq * (xx + yy))
        return x + (pattern[..., None] if is_colour(x) else pattern)
    raise ValueError("Unknown image noise kind: " + str(kind))


def motion_psf(length: int = 15, angle: float = 0.0) -> np.ndarray:
    """Line point-spread function - models camera or subject movement during exposure."""
    length = max(int(length), 1)
    psf = np.zeros((length, length))
    psf[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), angle, 1.0)
    psf = cv2.warpAffine(psf, m, (length, length))
    return psf / max(psf.sum(), EPS)


def gaussian_psf(size: int = 15, sigma: float = 3.0) -> np.ndarray:
    size = max(int(size) | 1, 3)
    k = cv2.getGaussianKernel(size, sigma)
    return (k @ k.T) / max(float((k @ k.T).sum()), EPS)


def disc_psf(radius: int = 7) -> np.ndarray:
    """Circular aperture PSF - the classic out-of-focus blur."""
    r = max(int(radius), 1)
    size = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    psf = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float64)
    return psf / max(psf.sum(), EPS)


def make_psf(kind: str, size: int = 15, angle: float = 0.0, sigma: float = 3.0) -> np.ndarray:
    if kind == "Motion blur":
        return motion_psf(size, angle)
    if kind == "Gaussian blur":
        return gaussian_psf(size, sigma)
    if kind == "Defocus (disc)":
        return disc_psf(max(size // 2, 1))
    raise ValueError("Unknown blur kind: " + str(kind))


def blur(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve with a PSF - the forward degradation model g = h * f that restoration inverts."""
    return _per_channel(img, lambda ch: cv2.filter2D(ch, -1, psf, borderType=cv2.BORDER_REFLECT))


# ----------------------------------------------------------------- spatial filtering
def spatial_filter(img: np.ndarray, kind: str = "Gaussian", size: int = 5, sigma: float = 1.5,
                   strength: float = 1.0, sigma_colour: float = 50.0) -> np.ndarray:
    """Apply one of SPATIAL_FILTERS.

    The instructive contrast here is linear vs non-linear: mean and Gaussian blur noise *and*
    edges alike, while median, bilateral and non-local means are explicitly built to leave
    edges intact - median by picking an existing value rather than averaging, bilateral by
    weighting on intensity as well as distance.
    """
    k = max(int(size) | 1, 3)
    x = np.asarray(img, dtype=np.float64)

    if kind == "Mean (box)":
        return _per_channel(x, lambda ch: cv2.blur(ch, (k, k)))
    if kind == "Gaussian":
        return _per_channel(x, lambda ch: cv2.GaussianBlur(ch, (k, k), sigma))
    if kind == "Median":
        return to_uint8(x) if k < 3 else _per_channel(
            x, lambda ch: cv2.medianBlur(to_uint8(ch), k).astype(np.float64))
    if kind == "Bilateral":
        return _per_channel(
            x, lambda ch: cv2.bilateralFilter(ch.astype(np.float32), k, sigma_colour,
                                              max(sigma, 1.0)).astype(np.float64))
    if kind == "Non-local means":
        src = to_uint8(x)
        if is_colour(src):
            out = cv2.fastNlMeansDenoisingColored(src, None, strength * 10, strength * 10, 7, 21)
        else:
            out = cv2.fastNlMeansDenoising(src, None, strength * 10, 7, 21)
        return out.astype(np.float64)
    if kind == "Unsharp mask":
        blurred = _per_channel(x, lambda ch: cv2.GaussianBlur(ch, (k, k), sigma))
        return x + strength * (x - blurred)          # add back the detail the blur removed
    if kind == "Laplacian sharpen":
        lap = _per_channel(x, lambda ch: cv2.Laplacian(ch, cv2.CV_64F, ksize=3))
        return x - strength * lap
    if kind == "Min (erosion)":
        return _per_channel(x, lambda ch: ndimage.minimum_filter(ch, size=k))
    if kind == "Max (dilation)":
        return _per_channel(x, lambda ch: ndimage.maximum_filter(ch, size=k))
    raise ValueError("Unknown spatial filter: " + str(kind))


def edge_detect(img: np.ndarray, operator: str = "Sobel", ksize: int = 3,
                low: float = 50.0, high: float = 150.0, sigma: float = 1.5) -> np.ndarray:
    """Gradient / edge maps.  All operators run on luminance, since edges are a structural
    property and colour channels would only triple the work for the same contours."""
    g = to_gray(img)
    k = max(int(ksize) | 1, 3)

    if operator == "Sobel":
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=k)
        return np.hypot(gx, gy)
    if operator == "Scharr":
        return np.hypot(cv2.Scharr(g, cv2.CV_64F, 1, 0), cv2.Scharr(g, cv2.CV_64F, 0, 1))
    if operator == "Prewitt":
        kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
        return np.hypot(cv2.filter2D(g, cv2.CV_64F, kx), cv2.filter2D(g, cv2.CV_64F, kx.T))
    if operator == "Roberts cross":
        k1 = np.array([[1, 0], [0, -1]], dtype=np.float64)
        k2 = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        return np.hypot(cv2.filter2D(g, cv2.CV_64F, k1), cv2.filter2D(g, cv2.CV_64F, k2))
    if operator == "Laplacian":
        return np.abs(cv2.Laplacian(g, cv2.CV_64F, ksize=k))
    if operator == "Laplacian of Gaussian":
        return np.abs(ndimage.gaussian_laplace(g, sigma))
    if operator == "Canny":
        return cv2.Canny(to_uint8(g), float(low), float(high)).astype(np.float64)
    raise ValueError("Unknown edge operator: " + str(operator))


# ----------------------------------------------------------------- frequency domain
def fft2_spectrum(img: np.ndarray, log: bool = True, shift: bool = True):
    """Centred 2-D DFT magnitude of the luminance.

    The zero-frequency term is moved to the centre and a log is taken, without which the DC
    component is so much larger than everything else that the display is a single bright dot.
    """
    g = to_gray(img)
    F = np.fft.fft2(g)
    if shift:
        F = np.fft.fftshift(F)
    mag = np.abs(F)
    return 20.0 * np.log10(mag + 1.0) if log else mag


def _distance_grid(shape):
    """Radial distance from the centred DC term, in pixels."""
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def frequency_mask(shape, kind: str = "Gaussian low-pass", cutoff: float = 40.0,
                   order: int = 2, bandwidth: float = 20.0,
                   notch_points: list | None = None) -> np.ndarray:
    """Build a centred transfer function H(u, v) for FREQ_FILTERS.

    The three low-pass shapes are the standard teaching trio: the ideal filter has the
    sharpest cutoff but its abrupt edge is a sinc in the spatial domain, which produces
    visible ringing; Butterworth softens that with an order-controlled roll-off; Gaussian has
    no ringing at all because a Gaussian transforms to a Gaussian.
    """
    d = _distance_grid(shape)
    d0 = max(float(cutoff), 1e-6)

    if kind == "Ideal low-pass":
        return (d <= d0).astype(np.float64)
    if kind == "Ideal high-pass":
        return (d > d0).astype(np.float64)
    if kind == "Butterworth low-pass":
        return 1.0 / (1.0 + (d / d0) ** (2 * max(int(order), 1)))
    if kind == "Butterworth high-pass":
        return 1.0 / (1.0 + (d0 / np.maximum(d, EPS)) ** (2 * max(int(order), 1)))
    if kind == "Gaussian low-pass":
        return np.exp(-(d ** 2) / (2.0 * d0 ** 2))
    if kind == "Gaussian high-pass":
        return 1.0 - np.exp(-(d ** 2) / (2.0 * d0 ** 2))
    if kind == "Band-pass":
        return np.exp(-(((d ** 2 - d0 ** 2) / np.maximum(d * max(bandwidth, 1.0), EPS)) ** 2))
    if kind == "Band-stop (notch)":
        if notch_points:
            # Reject specific interference peaks, plus their conjugates for a real result.
            h, w = shape
            mask = np.ones(shape)
            for (py, px) in notch_points:
                for (qy, qx) in ((py, px), (h - py, w - px)):
                    yy, xx = np.ogrid[:h, :w]
                    mask *= 1.0 - np.exp(-(((yy - qy) ** 2 + (xx - qx) ** 2) /
                                           (2.0 * max(bandwidth, 1.0) ** 2)))
            return mask
        return 1.0 - np.exp(-(((d ** 2 - d0 ** 2) /
                               np.maximum(d * max(bandwidth, 1.0), EPS)) ** 2))
    raise ValueError("Unknown frequency filter: " + str(kind))


def frequency_filter(img: np.ndarray, kind: str = "Gaussian low-pass", cutoff: float = 40.0,
                     order: int = 2, bandwidth: float = 20.0,
                     notch_points: list | None = None):
    """Filter in the frequency domain: FFT, multiply by H, inverse FFT.

    Returns (filtered image, the mask that was applied) so the UI can show the transfer
    function next to its effect - the whole point of working in this domain is that the
    filter is something you can look at.
    """
    def run(ch):
        F = np.fft.fftshift(np.fft.fft2(ch))
        return np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))

    mask = frequency_mask(img.shape[:2], kind, cutoff, order, bandwidth, notch_points)
    return _per_channel(np.asarray(img, dtype=np.float64), run), mask


def detect_periodic_peaks(img: np.ndarray, count: int = 4, exclude_radius: int = 10):
    """Find the brightest off-centre spectrum peaks - the signature of periodic interference,
    and the coordinates a notch filter needs in order to remove it."""
    spec = fft2_spectrum(img, log=True)
    h, w = spec.shape
    work = spec.copy()
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    work[((yy - cy) ** 2 + (xx - cx) ** 2) <= exclude_radius ** 2] = -np.inf
    peaks = []
    for _ in range(max(int(count), 0)):
        idx = int(np.argmax(work))
        py, px = divmod(idx, w)
        if not np.isfinite(work[py, px]):
            break
        peaks.append((py, px))
        work[max(py - 5, 0):py + 6, max(px - 5, 0):px + 6] = -np.inf
    return peaks


# ----------------------------------------------------------------- restoration
def _freq_restore(img: np.ndarray, psf: np.ndarray, transfer):
    """Run a frequency-domain restoration with reflective padding.

    The FFT treats an image as periodic, so the right edge convolves with the left one.  Real
    blur does no such thing, and the mismatch shows up as heavy ringing along every border -
    enough to cost several dB of PSNR even when the restoration itself is exact.  Padding the
    image symmetrically by the kernel size pushes that wraparound out into the margin, which
    is then cropped away.

    `transfer(H)` returns the multiplier applied to the spectrum, given the padded OTF.
    """
    from scipy import fft as sfft

    x = np.asarray(img, dtype=np.float64)
    pad = max(psf.shape) + 2
    # Rounded up to a size the FFT handles fast.  A 768x512 photo padded by 17 is 802 wide,
    # and 802 = 2 x 401 (a prime): the transform took 1.5 s.  The extra margin is more of the
    # same symmetric padding and is cropped away, so the result is unchanged.
    shape = (sfft.next_fast_len(x.shape[0] + 2 * pad), sfft.next_fast_len(x.shape[1] + 2 * pad))
    H = _psf_otf(psf, shape)
    mult = transfer(H)

    def run(ch):
        h, w = ch.shape
        wide = np.pad(ch, ((pad, shape[0] - h - pad), (pad, shape[1] - w - pad)),
                      mode="symmetric")
        out = np.real(sfft.ifft2(sfft.fft2(wide, workers=-1) * mult, workers=-1))
        return out[pad:pad + h, pad:pad + w]

    return _per_channel(x, run)


def wiener_deconvolution(img: np.ndarray, psf: np.ndarray, k: float = 0.01) -> np.ndarray:
    """Wiener deconvolution: F = G · H* / (|H|² + K).

    A plain inverse filter divides by H, which explodes wherever H is near zero and so
    amplifies noise catastrophically.  The K term is the noise-to-signal power ratio, and it
    is exactly what bounds that division - larger K means a safer but blurrier restoration.
    """
    return _freq_restore(img, psf,
                         lambda H: np.conj(H) / (np.abs(H) ** 2 + max(k, EPS)))


def inverse_filter(img: np.ndarray, psf: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Direct inverse filter with a magnitude threshold.

    Included mainly to show *why* Wiener is needed: without the threshold this is unusable on
    any real (noisy) image, and even with it the result is visibly noisier.
    """
    return _freq_restore(
        img, psf, lambda H: 1.0 / np.where(np.abs(H) < threshold, threshold, H))


def richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Richardson-Lucy iterative deconvolution.

    Maximum-likelihood under Poisson statistics, so it suits photon-limited images and keeps
    the result non-negative - something the linear methods cannot promise.  More iterations
    sharpen further but progressively amplify noise, so the count is a real trade-off.
    """
    def run(ch):
        obs = np.maximum(ch, EPS)
        est = np.full_like(obs, float(obs.mean()))
        flipped = psf[::-1, ::-1]
        for _ in range(max(int(iterations), 1)):
            conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
            est = est * cv2.filter2D(obs / np.maximum(conv, EPS), -1, flipped,
                                     borderType=cv2.BORDER_REFLECT)
            est = np.clip(est, 0.0, 1e6)
        return est

    return _per_channel(np.asarray(img, dtype=np.float64), run)


def constrained_least_squares(img: np.ndarray, psf: np.ndarray, gamma: float = 0.03) -> np.ndarray:
    """CLS restoration: F = G · H* / (|H|² + γ|P|²), where P is the Laplacian.

    Instead of Wiener's statistical K it penalises roughness directly, so it needs no noise
    estimate - only a smoothness weight γ.

    Note that γ is *not* on the same scale as Wiener's K: K is a noise-to-signal power ratio,
    while γ weights |P|², the Laplacian's response, which is small at low frequencies and
    large at high ones. γ therefore needs to be roughly an order of magnitude larger than a
    comparable K - around 0.03 here, and higher still for a wide defocus kernel.
    """
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

    def transfer(H):
        P = _psf_otf(laplacian, H.shape)
        return np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2)

    return _freq_restore(img, psf, transfer)


def _psf_otf(psf: np.ndarray, shape) -> np.ndarray:
    """Zero-pad a PSF to the image size and transform it, with the kernel centre rolled to the
    origin so the restoration does not come out spatially shifted."""
    pad = np.zeros(shape, dtype=np.float64)
    h, w = psf.shape
    pad[:h, :w] = psf
    pad = np.roll(pad, (-(h // 2), -(w // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


#: Sensible starting value of the regularisation parameter for each restoration method.
#: They are different quantities (a noise ratio, a magnitude floor, a smoothness weight), so
#: one shared default would make at least one of the three behave badly.
DEFAULT_REGULARISATION = {"Wiener deconvolution": 0.01, "Inverse filter": 0.05,
                          "Richardson-Lucy": 0.0, "Constrained least squares": 0.03}


def restore(img: np.ndarray, method: str, psf: np.ndarray, k: float = 0.01,
            iterations: int = 20) -> np.ndarray:
    if method == "Wiener deconvolution":
        return wiener_deconvolution(img, psf, k)
    if method == "Inverse filter":
        return inverse_filter(img, psf, max(k, 0.01))
    if method == "Richardson-Lucy":
        return richardson_lucy(img, psf, iterations)
    if method == "Constrained least squares":
        return constrained_least_squares(img, psf, k)
    raise ValueError("Unknown restoration method: " + str(method))


# ----------------------------------------------------------------- histogram / contrast
def histogram_op(img: np.ndarray, op: str = "Histogram equalisation", gamma: float = 1.0,
                 clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Point operations on intensity.

    For colour images, equalisation runs on the Y channel of YCrCb rather than on R, G and B
    separately - equalising channels independently shifts their relative levels and visibly
    distorts the colours.
    """
    x = np.asarray(img, dtype=np.float64)

    if op == "Histogram equalisation":
        return _on_luma(x, lambda y: cv2.equalizeHist(to_uint8(y)).astype(np.float64))
    if op == "CLAHE (adaptive)":
        clahe = cv2.createCLAHE(clipLimit=max(clip_limit, 0.01),
                                tileGridSize=(max(int(tile), 1), max(int(tile), 1)))
        return _on_luma(x, lambda y: clahe.apply(to_uint8(y)).astype(np.float64))
    if op == "Gamma correction":
        return 255.0 * np.power(np.clip(x, 0, 255) / 255.0, max(gamma, 1e-3))
    if op == "Contrast stretch":
        lo, hi = np.percentile(x, 2.0), np.percentile(x, 98.0)
        return np.clip((x - lo) * 255.0 / max(hi - lo, EPS), 0, 255)
    if op == "Logarithmic":
        return 255.0 * np.log1p(np.clip(x, 0, 255)) / np.log1p(255.0)
    if op == "Negative":
        return 255.0 - np.clip(x, 0, 255)
    raise ValueError("Unknown histogram operation: " + str(op))


def _on_luma(img, fn):
    if not is_colour(img):
        return fn(img)
    ycc = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = to_uint8(fn(ycc[..., 0].astype(np.float64)))
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float64)


def histogram(img: np.ndarray, bins: int = 256):
    """(bin centres, counts) of the luminance - the distribution the point operations reshape."""
    g = to_gray(img)
    counts, edges = np.histogram(np.clip(g, 0, 255), bins=bins, range=(0, 255))
    return (edges[:-1] + edges[1:]) / 2.0, counts


def morphology(img: np.ndarray, op: str = "Opening", size: int = 3,
               iterations: int = 1) -> np.ndarray:
    """Morphological cleanup.  Opening removes small bright specks, closing fills small dark
    holes - the standard pairing after thresholding a noisy image."""
    ops = {"Erosion": cv2.MORPH_ERODE, "Dilation": cv2.MORPH_DILATE, "Opening": cv2.MORPH_OPEN,
           "Closing": cv2.MORPH_CLOSE, "Gradient": cv2.MORPH_GRADIENT,
           "Top hat": cv2.MORPH_TOPHAT, "Black hat": cv2.MORPH_BLACKHAT}
    if op not in ops:
        raise ValueError("Unknown morphological operation: " + str(op))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(size) | 1, 3),) * 2)
    return cv2.morphologyEx(to_uint8(img), ops[op], k,
                            iterations=max(int(iterations), 1)).astype(np.float64)


MORPH_OPS = ("Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top hat", "Black hat")


# ----------------------------------------------------------------- reporting
def compare(reference: np.ndarray, test: np.ndarray) -> dict:
    """MSE / RMSE / PSNR / SSIM between two images, computed on luminance."""
    return _metrics.image_report(to_gray(reference), to_gray(test), data_range=255.0)


def summary(img: np.ndarray) -> dict:
    g = to_gray(img)
    return {"Width": int(img.shape[1]), "Height": int(img.shape[0]),
            "Channels": int(img.shape[2]) if is_colour(img) else 1,
            "Mean intensity": float(np.mean(g)), "Std deviation": float(np.std(g)),
            "Min": float(np.min(g)), "Max": float(np.max(g)),
            "Entropy (bits)": _entropy(g)}


def _entropy(g: np.ndarray) -> float:
    """Shannon entropy of the intensity histogram - an information-theoretic read on how much
    detail survived a processing step."""
    counts, _ = np.histogram(np.clip(g, 0, 255), bins=256, range=(0, 255))
    p = counts.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    return float(-np.sum(p * np.log2(p)))
