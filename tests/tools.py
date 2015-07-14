"""
Tools for testing.
"""

import numpy as np

from realtimepork.constants import KB, ME, HBAR
from realtimepork.tools import meshgrid, signed_sqrt


# Test parameters for a harmonic oscillator.
harmonic_parameters = {
    'mass': ME,  # g/mol
    'omega': 1. * KB / HBAR,  # 1/ps
    'init_p_max': 1e-2,  # g nm/ps mol
    'init_q_max': 150.,  # nm
    'gamma': 0.018,  # 1/nm^2
}


def harmonic_hk(mass, omega, gamma, ts, shape) -> '1':
    """
    Herman-Kluk prefactor for a harmonic oscillator.

    Parameters:
      mass: Mass of particle (g/mol).
      omega: Angular frequency of oscillator (1/ps).
      gamma: Coherent state width (1/nm^2).
      ts: NumPy array of times for which to find the prefactor (ps).
      shape: NumPy shape tuple for the array of initial conditions.
    """

    x = mass * omega / (HBAR * gamma)  # 1

    result = signed_sqrt(np.cos(omega * ts) - 0.5j * (x + 1. / x) * np.sin(omega * ts))
    # The result is independent of initial conditions for a harmonic oscillator.
    result.resize((len(result), 1, 1))
    return np.tile(result, shape)


def harmonic_trajectory(mass, omega, gamma, ts, init_ps, init_qs) -> '(g nm/ps mol, nm, 1, kJ ps/mol)':
    """
    Analytically-determined time-propagated quantities for a harmonic
    oscillator trajectory.

    Parameters:
      mass: Mass of particle (g/mol).
      omega: Angular frequency of oscillator (1/ps).
      gamma: Coherent state width (1/nm^2).
      ts: NumPy array of times at which to evaluate the trajectory (ps).
      init_ps: NumPy array of initial momenta (g nm/ps mol).
      init_qs: NumPy array of initial positions (nm).

    Returns:
      Momenta, positions, Herman-Kluk prefactors, classical actions.
    """

    mesh_ts, mesh_ps, mesh_qs = meshgrid(ts, init_ps, init_qs)
    shape = np.broadcast(init_ps, init_qs).shape

    c = np.cos(omega * mesh_ts)
    s = np.sin(omega * mesh_ts)

    ps = mesh_ps * c - mass * omega * mesh_qs * s  # g nm/ps mol
    qs = mesh_ps * s / (mass * omega) + mesh_qs * c  # nm

    if gamma is not None:
        Rs = harmonic_hk(mass, omega, gamma, mesh_ts, shape)
    else:
        Rs = None

    Ss = 0.5 * ((mesh_ps * mesh_ps / (mass * omega) - mass * omega * mesh_qs * mesh_qs) * c - 2. * mesh_ps * mesh_qs * s) * s  # kJ ps/mol

    return ps, qs, Rs, Ss


# Test parameters for a double well.
double_well_parameters = [
    # Stay on one side (starting with negative momentum).
    {
        'mass': 0.001,  # g/mol
        'depth': 1e-5,  # kJ/mol
        'width': 10.,  # nm
        'init_p_max': 1e-5,  # g nm/ps mol
        'init_q_max': 13.  # nm
    },
    # Visit both sides (starting with negative momentum).
    {
        'mass': 0.001,  # g/mol
        'depth': 1e-5,  # kJ/mol
        'width': 10.,  # nm
        'init_p_max': 1e-4,  # g nm/ps mol
        'init_q_max': 13.  # nm
    },
]
