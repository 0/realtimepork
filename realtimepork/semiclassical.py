"""
Semiclassical IVR using the Herman-Kluk propagator.
"""

import numpy as N

from .classical import RuthForest, TrajectoryAction
from .constants import HBAR
from .tools import SignedSqrt


class RungeKutta4HermanKluk:
    """
    Specialized RK4 integrator for the needs of the HK prefactor.

    Based on RungeKutta4.
    """

    def __init__(self, mass, dt, init_Mpxs, init_Mqxs):
        """
        Parameters:
          mass: Mass of particle (g/mol).
          dt: Time step (ps).
          init_Mpxs: NumPy array of initial "momenta".
          init_Mqxs: NumPy array of initial "positions".

        Note:
          The quantities Mpx and Mqx are elements of the stability matrix,
          where "x" is either "p" or "q", depending on which column of the
          matrix is being propagated.
        """

        self._mass = mass
        self._dt = dt

        self._Mpxs = init_Mpxs
        self._Mqxs = init_Mqxs

    @property
    def Mpxs(self):
        """
        Current Mpxs.
        """

        return self._Mpxs

    @property
    def Mqxs(self):
        """
        Current Mqxs.
        """

        return self._Mqxs

    def step(self, hessians):
        """
        Parameters:
          hessians: NumPy array containing the second derivative of the
                    potential evaluated at positions corresponding to the
                    times: at the current step, half-way to the next step, and
                    at the next the step (kJ/nm^2 mol).
        """

        # Get the state before the next step.
        result = self._Mpxs, self._Mqxs

        dt = self._dt

        dps1 = -dt * hessians[0] * self._Mqxs
        dqs1 = dt * self._Mpxs / self._mass

        dps2 = -dt * hessians[1] * (self._Mqxs + 0.5 * dqs1)
        dqs2 = dt * (self._Mpxs + 0.5 * dps1) / self._mass

        dps3 = -dt * hessians[1] * (self._Mqxs + 0.5 * dqs2)
        dqs3 = dt * (self._Mpxs + 0.5 * dps2) / self._mass

        dps4 = -dt * hessians[2] * (self._Mqxs + dqs3)
        dqs4 = dt * (self._Mpxs + dps3) / self._mass

        self._Mpxs = self._Mpxs + (dps1 + 2.0 * (dps2 + dps3) + dps4) / 6.0
        self._Mqxs = self._Mqxs + (dqs1 + 2.0 * (dqs2 + dqs3) + dqs4) / 6.0

        return result


class HermanKlukPrefactor:
    """
    Propagation of the Herman-Kluk prefactor.
    """

    def __init__(self, gamma, mass, dt, hessian_f, shape):
        """
        Parameters:
          gamma: Coherent state width (1/nm^2).
          mass: Mass of particle (g/mol).
          dt: Time step (ps).
          hessian_f: Function to calculate second derivate of potential at some
                     position (nm -> kJ/nm^2 mol).
          shape: NumPy shape tuple for the array of initial conditions.
        """
        self._gamma = gamma
        self._dt = dt
        self._hessian_f = hessian_f

        self._ss = SignedSqrt()

        # The stability matrix is initially the identity matrix.
        self._integrator_p = RungeKutta4HermanKluk(mass, dt, N.ones(shape), N.zeros(shape))
        self._integrator_q = RungeKutta4HermanKluk(mass, dt, N.zeros(shape), N.ones(shape))

    def step(self, qs) -> '1':
        """
        Take a step and find the new HK prefactor.

        Parameters:
          qs: NumPy array containing the positions corresponding to the times: at
              the current step, half-way to the next step, and at the next the
              step (nm).
        """

        hessians = self._hessian_f(qs) # kJ/nm^2 mol

        Mpp, Mqp = self._integrator_p.step(hessians)
        Mpq, Mqq = self._integrator_q.step(hessians)

        return self._ss(0.5 * (Mpp + Mqq + 1j * (Mpq / (HBAR * self._gamma) - HBAR * self._gamma * Mqp)))


class SemiclassicalTrajectory:
    """
    Propagation of a classical trajectory in time, keeping track of values of
    interest for semiclassical dynamics.
    """

    def __init__(self, gamma, mass, dt, potential_fs, init_ps, init_qs, max_steps=None):
        """
        Parameters:
          gamma: Coherent state width (1/nm^2).
          mass: Mass of particle (g/mol).
          dt: Time step (ps).
          potential_fs: Tuple of functions describing the potential
                        (nm -> kJ/mol, nm -> kJ/nm mol, nm -> kJ/nm^2 mol).
          init_ps: NumPy array of initial momenta (g nm/ps mol).
          init_qs: NumPy array of initial positions (nm).
          max_steps: Number of steps after which to terminate.

        Note:
          The arrays of initial momenta and positions may be of any shape, as
          long as they can be broadcast together.
        """

        self._max_steps = max_steps

        potential_f, force_f, hessian_f = potential_fs
        shape = N.broadcast(init_ps, init_qs).shape

        # We need the half-step information from the classical integrator, so
        # we will have to step it twice for each "real" step.
        self._classical_integrator = RuthForest(mass, 0.5 * dt, force_f, init_ps, init_qs)
        self._classical_action = TrajectoryAction(mass, dt, potential_f, init_ps, init_qs)
        self._hk_prefactor = HermanKlukPrefactor(gamma, mass, dt, hessian_f, shape)

        self._cur_step = 0

    def __iter__(self) -> 'SemiclassicalTrajectory':
        return self

    def __next__(self) -> '(ps, g nm/ps mol, nm, 1, kJ ps/mol)':
        if self._max_steps is not None and self._cur_step >= self._max_steps:
            raise StopIteration

        self._cur_step += 1

        t, p1, q1 = next(self._classical_integrator)
        _, p2, q2 = next(self._classical_integrator)
        # The integrator has already stepped to the last time, but we don't
        # want it to move anymore until our next step, so we just ask it for
        # the position.
        q3 = self._classical_integrator.qs

        S = self._classical_action.step(p2)
        R = self._hk_prefactor.step(N.array([q1, q2, q3]))

        return t, p1, q1, R, S
