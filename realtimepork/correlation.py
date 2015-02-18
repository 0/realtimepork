"""
Real-time correlation.
"""

import numpy as N

from warnings import warn

from .constants import HBAR
from .semiclassical import SemiclassicalTrajectory
from .tools import meshgrid


class SurvivalAmplitude:
    """
    Calculate survival amplitude of the ground state using Herman-Kluk SC-IVR.
    """

    def __init__(self, gamma, mass, dt, potential_fs, dp, q_grid, wf, p_max=None, max_steps=None):
        """
        Parameters:
          gamma: Coherent state width (1/nm^2).
          mass: Mass of particle (g/mol).
          dt: Time step (ps).
          potential_fs: Tuple of functions describing the potential
                        (nm -> kJ/mol, nm -> kJ/nm mol, nm -> kJ/nm^2 mol).
          dp: Spacing of momentum grid (g nm/ps mol).
          q_grid: Evenly spaced grid of position points (nm).
          wf: Ground state wavefunction evaluated on the position grid.
          p_max: Range of momentum grid (g nm/ps mol).
          max_steps: Number of steps after which to terminate.
        """

        assert len(q_grid) > 1, 'More than one position grid point required.'
        assert len(wf) == len(q_grid), 'Wavefunction is not on position grid.'

        # Normalize the wavefunction the way we require (with the square root
        # of the volume element included in the wavefunction).
        self._wf = wf / N.sqrt(N.sum(wf * wf)) # 1

        self._gamma = gamma # 1/nm^2
        self._q_grid = q_grid # nm
        self._max_steps = max_steps

        dq = self._q_grid[1] - self._q_grid[0] # nm

        # Use the Nyquist frequency relation to obtain the maximum allowed
        # value for the momentum grid.
        p_max_max = N.pi * HBAR / dq # g nm/ps mol

        if p_max is not None:
            if p_max > p_max_max:
                warn('Given p_max ({}) exceeds recommended maximum ({}).'.format(p_max, p_max_max))

            # Always use the given value if there is one.
            p_max_max = p_max

        # Rounding down to err on the side of caution (we shouldn't exceed
        # p_max_max).
        p_grid_len = 2 * int(p_max_max / dp) + 1
        p_max = dp * (p_grid_len - 1) / 2 # g nm/ps mol
        p_grid = N.linspace(-p_max, p_max, p_grid_len) # g nm/ps mol

        self._cur_step = 0

        # One fewer dq than there are position grid integrations due to the
        # normalization of wf.
        self._C = dp * dq * dq * N.sqrt(self._gamma / N.pi) / (2. * N.pi * HBAR) # 1

        mesh_ps, mesh_qs = meshgrid(p_grid, self._q_grid)
        self._trajs = SemiclassicalTrajectory(self._gamma, mass, dt, potential_fs, mesh_ps, mesh_qs)
        self._transformed_wf0 = self._transform_wf(mesh_ps, mesh_qs).conj() # 1

    def _transform_wf(self, ps, qs) -> '1':
        """
        Perform one of the position integrals.

        Parameters:
          ps: Momentum grid (g nm/ps mol).
          qs: Position grid (nm).
        """

        result = N.zeros(N.broadcast(ps, qs).shape, dtype=complex) # 1

        for q_j, wf_j in zip(self._q_grid, self._wf):
            qdiff = q_j - qs # nm
            result += N.exp(-0.5 * self._gamma * qdiff * qdiff + 1j / HBAR * ps * qdiff) * wf_j

        return result

    def __iter__(self) -> 'SurvivalAmplitude':
        return self

    def __next__(self) -> '(ps, 1)':
        if self._max_steps is not None and self._cur_step >= self._max_steps:
            raise StopIteration

        self._cur_step += 1

        t, ps, qs, Rs, Ss = next(self._trajs)
        consts = Rs * N.exp(1j / HBAR * Ss) # 1
        transformed_wf = self._transform_wf(ps, qs) # 1

        # Perform the final integrals over p and q.
        return t, self._C * N.sum(consts * transformed_wf * self._transformed_wf0)
