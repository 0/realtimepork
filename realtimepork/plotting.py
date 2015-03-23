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
