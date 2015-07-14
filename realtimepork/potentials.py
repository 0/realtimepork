"""
Example potential functions.
"""

import numpy as np


def free_particle():
    """
    Free particle (flat) potential.
    """

    def potential_f(q) -> 'kJ/mol':
        # Keep the shape of the input.
        return 0 * q

    def force_f(q) -> 'kJ/nm mol':
        # Keep the shape of the input.
        return 0 * q

    def hessian_f(q) -> 'kJ/nm^2 mol':
        # Keep the shape of the input.
        return 0 * q

    return potential_f, force_f, hessian_f


def harmonic(k=None, m=None, omega=None):
    """
    Harmonic potential relative to the origin.

    Note:
      Either k or (m and omega) must be specified.

    Parameters:
      k: Spring constant (kJ/nm^2 mol).
      m: Mass of particle (g/mol).
      omega: Angular frequency of oscillator (1/ps).
    """

    if k is not None:
        force_constant = k  # kJ/nm^2 mol
    elif m is not None and omega is not None:
        force_constant = m * omega * omega  # kJ/nm^2 mol
    else:
        assert False, 'Must provide either k or (m and omega).'

    def potential_f(q) -> 'kJ/mol':
        return 0.5 * force_constant * q * q

    def force_f(q) -> 'kJ/nm mol':
        return -force_constant * q

    def hessian_f(q) -> 'kJ/nm^2 mol':
        # Keep the shape of the input.
        return force_constant * np.ones_like(q)

    return potential_f, force_f, hessian_f


def quartic(a, b, c):
    """
    Quartic potential of the form a x^2 + b x^3 + c x^4.

    Parameters:
      a: (kJ/nm^2 mol).
      b: (kJ/nm^3 mol).
      c: (kJ/nm^4 mol).
    """

    def potential_f(q) -> 'kJ/mol':
        q2 = q * q  # nm^2

        return (a + b * q + c * q2) * q2

    def force_f(q) -> 'kJ/nm mol':
        q2 = q * q  # nm^2

        return (-2. * a - 3. * b * q - 4. * c * q2) * q

    def hessian_f(q) -> 'kJ/nm^2 mol':
        return 2. * a + 6. * (b + 2. * c * q) * q

    return potential_f, force_f, hessian_f


def double_well(depth, width):
    """
    Double well potential.

    Parameters:
      depth: Height of barrier relative to minima (kJ/mol).
      width: Distance of minima from zero (nm).
    """

    w2 = width * width  # nm^2
    a = -2. * depth / w2  # kJ/nm^2 mol
    b = depth / (w2 * w2)  # kJ/nm^4 mol

    def potential_f(q) -> 'kJ/mol':
        q2 = q * q  # nm^2

        return (a + b * q2) * q2

    def force_f(q) -> 'kJ/nm mol':
        return (-2. * a - 4. * b * q * q) * q

    def hessian_f(q) -> 'kJ/nm^2 mol':
        return 2. * a + 12. * b * q * q

    return potential_f, force_f, hessian_f
