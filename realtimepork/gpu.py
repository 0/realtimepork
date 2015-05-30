"""
GPU utilities.
"""

from functools import wraps
from math import ceil

# Load everything we need in this module from PyCUDA (but don't autoinit until
# requested).
try:
    from pycuda.tools import DeviceData
except ImportError:
    _pycuda_available = False
else:
    _pycuda_available = True


# Is this thing on?
_enabled = False


class PyCUDAMissingError(Exception):
    pass


def _require_pycuda(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _pycuda_available:
            raise PyCUDAMissingError('Unable to load PyCUDA.')

        return f(*args, **kwargs)

    return wrapper


@_require_pycuda
def enable():
    """
    Initialize the GPU machinery.
    """

    global _enabled

    if _enabled:
        return

    import pycuda.autoinit

    _enabled = True


def is_enabled():
    """
    Check whether the GPU is available and initialized.
    """

    return _enabled


@_require_pycuda
def carve_array(xn, yn):
    """
    Determine the best grid and block sizes given the input size.

    Parameters:
      xn: Size in the x direction (shorter stride).
      yn: Size in the y direction (longer stride).

    Returns:
      Grid size tuple, block size tuple.
    """

    dev = DeviceData()

    # Align with the warp size in the x direction and use what remains for the
    # y direction.
    x_threads = dev.warp_size
    y_threads = dev.max_threads // x_threads

    assert x_threads * y_threads <= dev.max_threads

    x_blocks = int(ceil(xn / x_threads))
    y_blocks = int(ceil(yn / y_threads))

    return (x_blocks, y_blocks), (x_threads, y_threads, 1)
