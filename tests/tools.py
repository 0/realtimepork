"""
Tools for testing.
"""

import numpy as N

from realtimepork.constants import KB, ME, HBAR
from realtimepork.tools import meshgrid


# Test parameters for a harmonic oscillator.
harmonic_parameters = {
        'mass': ME, # g/mol
        'omega': 1. * KB / HBAR, # 1/ps
        'init_p_max': 1e-2, # g nm/ps mol
        'init_q_max': 150., # nm
        'gamma': 0.018, # 1/nm^2
        }

def harmonic_trajectory(mass, omega, ts, init_ps, init_qs) -> '(g nm/ps mol, nm, kJ ps/mol)':
    """
    Analytically-determined time-propagated quantities for a harmonic
    oscillator trajectory.

    Parameters:
      mass: Mass of particle (g/mol).
      omega: Angular frequency of oscillator (1/ps).
      ts: NumPy array of times at which to evaluate the trajectory (ps).
      init_ps: NumPy array of initial momenta (g nm/ps mol).
      init_qs: NumPy array of initial positions (nm).

    Returns:
      Momenta, positions, classical actions.
    """

    mesh_ts, mesh_ps, mesh_qs = meshgrid(ts, init_ps, init_qs)
    shape = N.broadcast(init_ps, init_qs).shape

    c = N.cos(omega * mesh_ts)
    s = N.sin(omega * mesh_ts)

    ps = mesh_ps * c - mass * omega * mesh_qs * s # g nm/ps mol
    qs = mesh_ps * s / (mass * omega) + mesh_qs * c # nm

    Ss = 0.5 * ((mesh_ps * mesh_ps / (mass * omega) - mass * omega * mesh_qs * mesh_qs) * c - 2. * mesh_ps * mesh_qs * s) * s # kJ ps/mol

    return ps, qs, Ss


# Test parameters for a double well.
double_well_parameters = [
        # Stay on one side (starting with negative momentum).
        {
            'mass': 0.001, # g/mol
            'depth': 1e-5, # kJ/mol
            'width': 10., # nm
            'init_p_max': 1e-5, # g nm/ps mol
            'init_q_max': 13. # nm
            },
        # Visit both sides (starting with negative momentum).
        {
            'mass': 0.001, # g/mol
            'depth': 1e-5, # kJ/mol
            'width': 10., # nm
            'init_p_max': 1e-4, # g nm/ps mol
            'init_q_max': 13. # nm
            },
        ]
