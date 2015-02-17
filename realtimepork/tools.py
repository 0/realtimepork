"""
Assorted tools.
"""

import numpy as N


def meshgrid(*args, indexing='ij', sparse=True, copy=False, **kwargs):
    """
    Wrapper for NumPy's meshgrid with useful defaults.
    """

    return N.meshgrid(*args, indexing=indexing, sparse=sparse, copy=copy, **kwargs)


class SignedSqrt:
    """
    Continuous square roots for a trajectory in the complex plane.

    Find the square root of complex values with the correct sign (on the
    correct sheet of the surface) for a trajectory that is determined
    iteratively.
    """

    def __init__(self):
        self.factor = None
        self.sign = None

    def __call__(self, v):
        """
        Parameters:
          v: Next NumPy array in the trajectory.

        Returns:
          Square root of v with the correct sign.
        """

        s = N.sign(v.imag)

        # NumPy's sign returns 0 for 0, but its sqrt is continuous from above
        # along the negative reals, so we consider the real line to be in the
        # upper half of the complex plane.
        s[s == 0] = 1

        if self.factor is None:
            self.factor = N.ones_like(v, dtype=int)
        else:
            self.factor *= -1 * (2 * N.logical_and(N.sign(v.real) < 0, self.sign != s) - 1)

        self.sign = s

        return self.factor * N.sqrt(v)


def signed_sqrt(vs):
    """
    Continuous square roots for a trajectory in the complex plane.

    Find the square root of complex values with the correct sign (on the
    correct sheet of the surface) for a trajectory that is known in its
    entirety.

    Parameters:
      vs: NumPy vector of complex numbers.

    Returns:
      Square roots of the elements of vs with the correct signs.

    Note:
      If vs has more than one axis, the outermost axis (0) is assumed to be
      along the time dimension, so this can be applied to multiple trajectories
      simultaneously.
    """

    if vs.size == 0:
        return vs

    signs_imag = N.sign(vs.imag)
    # NumPy's sign returns 0 for 0, but its sqrt is continuous from above along
    # the negative reals, so we consider the real line to be in the upper half
    # of the complex plane.
    signs_imag[signs_imag == 0] = 1
    signs_imag_rolled = N.roll(signs_imag, 1, axis=0)
    # Never flip the first sign.
    signs_imag_rolled[0] = signs_imag[0]

    # Make a vector of booleans, where each element is False iff it corresponds
    # to an appropriate sign change.
    transitions = N.logical_or(N.sign(vs.real) >= 0, signs_imag == signs_imag_rolled)

    # Convert the booleans to 1 and -1 and use the cumulative product to
    # "carry" each occurrence of -1 until the next one.
    return N.cumprod(2 * transitions - 1, axis=0) * N.sqrt(vs)
