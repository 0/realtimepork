"""
Convenience functions for plotting the generated data.
"""

import matplotlib.pyplot as plt


def plot_sa(ts, sas, out_path, *, x_label=None, y_label=None):
    """
    Plot the complex survival amplitude as a function of time.

    Parameters:
      ts: Times.
      sas: Vector of complex values corresponding to ts.
      out_path: Path to the file where the image should be written. Extension
                determines the image format (e.g. pdf, png).
    """

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(ts, sas.real, color='black', label='Real')
    ax.plot(ts, sas.imag, color='black', linestyle='dashed', label='Imaginary')

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.legend()

    fig.savefig(out_path, bbox_inches='tight', transparent=True)

    plt.close(fig)


def plot_spectrum(freqs, amps, out_path, *, magnitude=False, freq_window=None, x_label=None, y_label=None):
    """
    Plot the complex spectrum as a function of frequency.

    Parameters:
      freqs: Frequencies.
      amps: Vector of complex values corresponding to freqs.
      out_path: Path to the file where the image should be written. Extension
                determines the image format (e.g. pdf, png).
      magnitude: Whether to plot the magnitude of amps.
      freq_window: Tuple describing the range of frequencies to display.
    """

    fig = plt.figure()
    ax = fig.gca()

    if magnitude:
        ax.plot(freqs, abs(amps), color='black', label='Magnitude')
    else:
        ax.plot(freqs, amps.real, color='black', label='Real')
        ax.plot(freqs, amps.imag, color='black', linestyle='dashed', label='Imaginary')

    if freq_window is not None:
        ax.set_xlim(freq_window)

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.legend()

    fig.savefig(out_path, bbox_inches='tight', transparent=True)

    plt.close(fig)
