"""
Generation of classical trajectories.
"""

import numpy as N


class ClassicalIntegrator:
    """
    Abstract class for classical integrators.

    Subclasses need to implement a _step method, which should update the state
    variables.
    """

    def __init__(self, mass, dt, force_f, init_ps, init_qs, max_steps=None):
        """
        Parameters:
          mass: Mass of particle (g/mol).
          dt: Time step (ps).
          force_f: Force function (? -> kJ/nm mol).
          init_ps: NumPy array of initial momenta (g nm/ps mol).
          init_qs: NumPy array of initial positions (nm).
          max_steps: Number of steps after which to terminate.

        Note:
          The parameters of force_f depend on the integrator implementation.

        Note:
          The arrays of initial momenta and positions may be of any shape, as
          long as they can be broadcast to the same shape.
        """

        self._mass = mass
        self._dt = dt
        self._force_f = force_f
        self._max_steps = max_steps

        # Expand into the full shape if necessary.
        self._ps = init_ps + N.zeros_like(init_qs)
        self._qs = init_qs + N.zeros_like(init_ps)

        self._cur_step = 0
        self._t = 0.0  # ps

    @property
    def t(self) -> 'ps':
        """
        Current time.
        """

        return self._t

    @property
    def ps(self) -> 'g nm/ps mol':
        """
        Current momenta.
        """

        return self._ps

    @property
    def qs(self) -> 'nm':
        """
        Current positions.
        """

        return self._qs

    def __iter__(self) -> 'ClassicalIntegrator':
        return self

    def __next__(self) -> '(ps, g nm/ps mol, nm)':
        """
        Get the current state and prepare the next one.
        """

        if self._max_steps is not None and self._cur_step >= self._max_steps:
            raise StopIteration

        self._cur_step += 1

        # Get the state before the next step.
        result = self._t, self._ps.copy(), self._qs.copy()

        # We still step even if we expect to hit max_steps, because someone
        # might look at our state before the next call to __next__().
        self._step()
        self._t += self._dt

        return result


class RuthForest(ClassicalIntegrator):
    """
    Fourth-order symplectic integrator due to Ruth and Forest.

    Ref: Etienne Forest, Ronald D. Ruth, Fourth-order symplectic integration,
    Physica D: Nonlinear Phenomena, Volume 43, Issue 1, May 1990, Pages
    105-117, ISSN 0167-2789, http://dx.doi.org/10.1016/0167-2789(90)90019-L.

    Note:
      force_f: nm -> kJ/nm mol
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        x = (N.power(2, 1. / 3.) + N.power(2, -1. / 3.) - 1.) / 6.
        self._d1 = self._d4 = self._dt * (x + 0.5)
        self._c2 = self._c4 = self._dt * (2. * x + 1) / self._mass
        self._d2 = self._d3 = self._dt * (-x)
        self._c3 = self._dt * (-4. * x - 1) / self._mass

        self._last_fs = None  # kJ/nm mol

    def _step(self):
        dt = self._dt  # ps

        if self._last_fs is not None:
            # Don't recalculate it.
            last_fs = self._last_fs  # kJ/nm mol
        else:
            last_fs = self._force_f(self._qs)  # kJ/nm mol

        self._ps += self._d1 * last_fs  # g nm/ps mol
        self._qs += self._c2 * self._ps  # nm
        self._ps += self._d2 * self._force_f(self._qs)
        self._qs += self._c3 * self._ps
        self._ps += self._d3 * self._force_f(self._qs)
        self._qs += self._c4 * self._ps
        fs = self._force_f(self._qs)  # kJ/nm mol
        self._ps += self._d4 * fs

        self._last_fs = fs


# This integrator is only used for comparison in testing and as a model for the
# specialized integrator (RungeKutta4HermanKluk).
class RungeKutta4(ClassicalIntegrator):
    """
    Fourth-order Runge-Kutta integrator.

    Ref: Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.
    Numerical recipes in C. 1992. Cambridge: Cambridge University. (Section
    16.1, pp. 710-713)

    Note:
      force_f: (ps, nm) -> kJ/nm mol
    """

    def _step(self):
        dt = self._dt  # ps

        dps1 = dt * self._force_f(self._t, self._qs)  # g nm/ps mol
        dqs1 = dt * self._ps / self._mass  # nm
        dps2 = dt * self._force_f(self._t + 0.5 * dt, self._qs + 0.5 * dqs1)
        dqs2 = dt * (self._ps + 0.5 * dps1) / self._mass
        dps3 = dt * self._force_f(self._t + 0.5 * dt, self._qs + 0.5 * dqs2)
        dqs3 = dt * (self._ps + 0.5 * dps2) / self._mass
        dps4 = dt * self._force_f(self._t + dt, self._qs + dqs3)
        dqs4 = dt * (self._ps + dps3) / self._mass

        self._ps = self._ps + (dps1 + 2.0 * (dps2 + dps3) + dps4) / 6.0
        self._qs = self._qs + (dqs1 + 2.0 * (dqs2 + dqs3) + dqs4) / 6.0


class TrajectoryAction:
    """
    Action calculator for a classical trajectory.
    """

    def __init__(self, mass, dt, potential_f, init_ps, init_qs):
        """
        Parameters:
          mass: Mass of particle (g/mol).
          dt: Time step (ps).
          potential_f: Potential function (nm -> kJ/mol).
          init_ps: NumPy array of initial momenta (g nm/ps mol).
          init_qs: NumPy array of initial positions (nm).

        Note:
          The arrays of initial momenta and positions may be of any shape, as
          long as they can be broadcast together.
        """
        self._mass = mass
        self._dt = dt

        self._Hs = init_ps * init_ps / (2. * self._mass) + potential_f(init_qs)
        self._Ss = N.zeros(N.broadcast(init_ps, init_qs).shape)  # kJ ps/mol

    @property
    def Ss(self) -> 'kJ ps/mol':
        """
        Current action.
        """

        return self._Ss

    def step(self, ps) -> 'kJ ps/mol':
        """
        Get the current state and prepare the next one.

        Parameters:
          ps: Momenta half-way across the step (g nm/ps mol).
        """

        # Get the state before the next step.
        result = self._Ss

        self._Ss = self._Ss + self._dt * (ps * ps / self._mass - self._Hs)

        return result
