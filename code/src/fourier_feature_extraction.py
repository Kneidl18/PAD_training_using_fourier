import numpy as np

# -------------------- helper functions --------------------

def _fft2_shifted(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (magnitude_spectrum, phase_spectrum) of the centered 2D FFT."""
    f_transform = np.fft.fft2(image) # Compute the 2D Fast Fourier Transform of the image
    f_shift = np.fft.fftshift(f_transform) # Shift zero-frequency component to the center
    return np.abs(f_shift), np.angle(f_shift)

def _dist_from_center(shape: tuple[int, int]) -> np.ndarray:
    """Compute the Euclidean distance of each pixel from the image center."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2 # Coordinates of the image center
    Y, X = np.ogrid[:rows, :cols] # Create open grids for row (Y) and column (X) coords
    return np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2) # Euclidean distance

def _radial_edges(max_radius: float, n_bands: int) -> np.ndarray:
    """Generate evenly spaced radial boundaries (edges) from the center
    to the maximum radius for the given number of bands."""
    return np.linspace(0.0, max_radius, n_bands + 1)

def _band_mask(dist: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    """Create a boolean mask selecting all pixels whose distance
    from the center lies within the specified radial band."""
    return (dist >= r_min) & (dist < r_max)

def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalization of the vector with a small epsilon for numerical stability."""
    return vec / (np.linalg.norm(vec) + eps)

def circular_mean(angles: np.ndarray) -> float:
    """Compute the circular mean of angles (radians) in [-pi, pi).

    -> Used because phase is a circular quantity (from -pi to pi)
    (a simple arithmetic mean would produce incorrect results)."""
    sum_sin = np.sum(np.sin(angles))
    sum_cos = np.sum(np.cos(angles))
    return float(np.arctan2(sum_sin, sum_cos))


# -------------------- main feature extractor --------------------


def _extract_magnitude_features(power_spectrum, dist, max_radius, n_bands_magnitude, log_compression):
    """Extracts magnitude features from the power spectrum."""
    mag_edges = _radial_edges(max_radius, n_bands_magnitude)
    magnitude_features = []
    for i in range(n_bands_magnitude):
        r_min, r_max = mag_edges[i], mag_edges[i + 1]
        mask = _band_mask(dist, r_min, r_max)
        if np.any(mask):
            band_energy = np.sum(power_spectrum[mask])
        else:
            band_energy = 0.0
        magnitude_features.append(band_energy)

    magnitude_features = np.asarray(magnitude_features, dtype=float)
    if log_compression:
        magnitude_features = np.log1p(magnitude_features)

    return magnitude_features


def _extract_phase_features(phase_spectrum, dist, max_radius, n_bands_phase):
    """Extracts phase features from the phase spectrum."""
    phase_edges = _radial_edges(max_radius, n_bands_phase)
    phase_features = []
    for i in range(n_bands_phase):
        r_min, r_max = phase_edges[i], phase_edges[i + 1]
        mask = _band_mask(dist, r_min, r_max)
        if np.any(mask):
            band_phase = circular_mean(phase_spectrum[mask])
        else:
            band_phase = 0.0
        phase_features.append(band_phase)

    return np.asarray(phase_features, dtype=float)

def extract_fourier_features(
    image: np.ndarray,
    n_bands_magnitude: int = 30,
    n_bands_phase: int = 30,
    log_compression: bool = True,
    normalize: bool = True,
):
    """
    Extract a Fourier-based feature vector (magnitude and phase) from a 2D image.

    Parameters
    ----------
    image : np.ndarray
        Input 2D grayscale image.
    n_bands_magnitude : int
        Number of concentric radial bands for magnitude-energy features.
    n_bands_phase : int
        Number of concentric radial bands for phase (circular mean) features.
    log_compression : bool
        If True, log-compresses the magnitude-energy features.
    normalize : bool
        If True, L2-normalizes the resulting feature vectors.

    Returns
    -------
    magnitude_features : np.ndarray
        Shape: (n_bands_magnitude,) with log-energy per radial band.
    phase_features : np.ndarray
        Shape: (n_bands_phase,) with circular-mean phase per band.
    """

    # FFT (centered), obtain magnitude and phase spectra
    magnitude_spectrum, phase_spectrum = _fft2_shifted(image)

    dist = _dist_from_center(image.shape)
    max_radius = float(np.max(dist))

    power_spectrum = np.square(magnitude_spectrum)
    magnitude_features = _extract_magnitude_features(power_spectrum, dist, max_radius, n_bands_magnitude, log_compression)
    phase_features = _extract_phase_features(phase_spectrum, dist, max_radius, n_bands_phase)

    if normalize:
        magnitude_features = _l2_normalize(magnitude_features)
        phase_features = _l2_normalize(phase_features)

    return magnitude_features, phase_features

from functools import partial
from typing import Callable, List, Optional

def get_feature_extractor(feature_mode: str, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a partial function for the specified feature extraction mode."""
    if feature_mode == "magnitude":
        return partial(lambda img: extract_fourier_features(img, **kwargs)[0])
    elif feature_mode == "phase":
        return partial(lambda img: extract_fourier_features(img, **kwargs)[1])
    elif feature_mode == "combined":
        return partial(lambda img: np.concatenate(extract_fourier_features(img, **kwargs)))
    else:
        raise ValueError(f"Invalid feature_mode: {feature_mode}")


def normalize_task2_thirds(thirds: Optional[List[int]]) -> List[int]:
    """Normalize and validate the Task 2 thirds selection."""
    if not thirds:
        return [1, 2, 3]

    unique_sorted = sorted(set(thirds))
    invalid = [third for third in unique_sorted if third not in (1, 2, 3)]
    if invalid:
        raise ValueError(f"Invalid thirds selection: {invalid}. Expected values in [1, 2, 3].")

    return unique_sorted


def format_task2_thirds(thirds: Optional[List[int]]) -> str:
    """Format the Task 2 thirds selection for display."""
    return "".join(str(third) for third in normalize_task2_thirds(thirds))


def _get_third_indices(third_num: int) -> List[int]:
    """Return the feature indices for a specific magnitude third."""
    if third_num == 1:
        return list(range(0, 10))
    if third_num == 2:
        return list(range(10, 20))
    if third_num == 3:
        return list(range(20, 30))
    raise ValueError(f"Invalid third number: {third_num}")


def _get_combined_thirds_indices(thirds: Optional[List[int]]) -> List[int]:
    """Return combined indices for multiple magnitude thirds."""
    indices: List[int] = []
    for third_num in normalize_task2_thirds(thirds):
        indices.extend(_get_third_indices(third_num))
    return indices


def make_thirds_feature_extractor(thirds: Optional[List[int]]) -> Callable[[np.ndarray], np.ndarray]:
    """Create a magnitude feature extractor limited to selected thirds."""
    indices = _get_combined_thirds_indices(thirds)

    def feature_extractor(image: np.ndarray) -> np.ndarray:
        magnitude_features, _ = extract_fourier_features(
            image,
            n_bands_magnitude=30,
            n_bands_phase=0,
        )
        return magnitude_features[indices]

    return feature_extractor
