"""
Spectral analysis.
"""

import numpy as N


def transform(ts, xs):
    """
    Calculate the discrete Fourier transform of a signal.

    Parameters:
      ts: Times.
      xs: Signal values.

    Returns:
      The transformed frequencies and amplitudes.

    Note:
      The times must be equally spaced.
    """

    try:
        dt = ts[1] - ts[0]
    except:
        raise Exception('At least two time points required.')

    freqs = N.fft.fftshift(N.fft.fftfreq(len(xs), d=dt))
    # We use the inverse transform here to get the sign convention we need.
    spectrum = N.fft.fftshift(N.fft.ifft(xs))

    return freqs, spectrum


def find_peak(freqs, amps):
    """
    Find the tallest peak (by magnitude) of a complex spectrum.

    Parameters:
      freqs: Frequencies.
      amps: Spectrum values.

    Returns:
      Index and frequency of tallest peak.
    """

    idx = N.argmax(abs(amps))

    return idx, freqs[idx]


def interpolate(xs, freqs, amps, freq_window, factor):
    """
    Interpolate a discrete Fourier spectrum.

    This kind of interpolation is typically performed by zero-padding the
    signal, but that is wasteful in space. This function is instead wasteful in
    time, applying the discrete Fourier transform formula directly (and not
    seeing the usual FFT speedup). However, if only a small region is to be
    interpolated, this can be a worthwhile compromise.

    Parameters:
      xs: Signal values.
      freqs: Frequencies.
      amps: Spectrum values.
      freq_window: Tuple containing the frequencies between which the
                   interpolation should happen.
      factor: Scaling factor by which the density of points is to be increased.

    Returns:
      The interpolated frequencies and amplitudes.

    Note:
      The inputs must already be fftshifted, so that freqs is monotonically
      increasing.

    Note:
      The spacing between the freqs doesn't have to be consistent (it's
      possible to apply this interpolation several times), but the first and
      last values have to be those from the original transform.
    """

    assert freqs[0] <= freq_window[0] , 'Invalid frequency window: {}, {}.'.format(freqs[0], freq_window[0])
    assert freq_window[0] < freq_window[1], 'Invalid frequency window: {}, {}.'.format(freq_window[0], freq_window[1])
    assert freq_window[1] <= freqs[-1], 'Invalid frequency window: {}, {}.'.format(freq_window[1], freqs[-1])

    # Indices such that we bound freq_window as closely as possible.
    idx_min, idx_max = N.argmax(freqs > freq_window[0]) - 1, N.argmin(freqs < freq_window[1])
    # Indices before which we insert the new elements.
    idxs = N.repeat(N.arange(idx_min, idx_max) + 1, factor - 1)

    # Values to be inserted.
    new_freqs = freqs[idxs - 1] + N.tile(N.arange(factor - 1) + 1, idx_max - idx_min) * (freqs[idxs] - freqs[idxs - 1]) / factor
    new_amps = N.empty_like(new_freqs, dtype=complex)

    for i, freq in enumerate(new_freqs):
        # Find the fractional frequency index, taking into account the fftshift.
        k = (len(xs) - 1) * (freq - freqs[0]) / (freqs[-1] - freqs[0]) - len(xs) // 2
        new_amps[i] = sum(xs * N.exp(2j * N.pi * k * N.arange(len(xs)) / len(xs))) / len(xs)

    return N.insert(freqs, idxs, new_freqs), N.insert(amps, idxs, new_amps)
