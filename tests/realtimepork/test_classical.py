from unittest import main, TestCase

import numpy as N
from numpy.testing import assert_array_almost_equal

from realtimepork.classical import RungeKutta4, RuthForest, TrajectoryAction
from realtimepork.potentials import double_well, harmonic
from realtimepork.tools import meshgrid

from tests.tools import double_well_parameters, harmonic_parameters as hp, harmonic_trajectory


class ClassicalIntegratorTest:
    def testHarmonicOscillator(self):
        """
        Make sure the integrators can handle a harmonic oscillator.
        """

        ps = N.linspace(-hp['init_p_max'], hp['init_p_max'], 7) # g nm/ps mol
        qs = N.linspace(-hp['init_q_max'], hp['init_q_max'], 11) # nm
        init_ps, init_qs = meshgrid(ps, qs)

        ts = N.linspace(0., 3. * N.pi / hp['omega'], self.num_steps) # ps
        dt = ts[1] - ts[0] # ps

        integrator = self._initialize_integrator(hp['mass'], hp['omega'], dt, init_ps, init_qs)

        calculated_ps = N.empty((len(ts), len(ps), len(qs))) # g nm/ps mol
        calculated_qs = N.empty_like(calculated_ps) # nm

        for i, step in enumerate(integrator):
            _, calculated_ps[i], calculated_qs[i] = step

        expected_ps, expected_qs, _, _ = harmonic_trajectory(hp['mass'], hp['omega'], None, ts, init_ps, init_qs)

        assert_array_almost_equal(calculated_ps, expected_ps)
        assert_array_almost_equal(calculated_qs, expected_qs)


class RuthForestTest(TestCase, ClassicalIntegratorTest):
    num_steps = 900

    def _initialize_integrator(self, mass, omega, dt, init_ps, init_qs):
        _, force_f, _ = harmonic(m=mass, omega=omega)

        return RuthForest(mass, dt, force_f, init_ps, init_qs, max_steps=self.num_steps)


class RungeKutta4Test(TestCase, ClassicalIntegratorTest):
    num_steps = 600

    def _initialize_integrator(self, mass, omega, dt, init_ps, init_qs):
        _, force_f, _ = harmonic(m=mass, omega=omega)
        # Add a dummy time parameter.
        force_f_t = lambda t, q: force_f(q)

        return RungeKutta4(mass, dt, force_f_t, init_ps, init_qs, max_steps=self.num_steps)


class ClassicalIntegratorComparisonTest(TestCase):
    def testAllIntegrators(self):
        """
        Compare the integrators for a less trivial problem.
        """

        for param_idx, params in enumerate(double_well_parameters):
            _, force_f, _ = double_well(depth=params['depth'], width=params['width'])
            # Add a dummy time parameter.
            force_f_t = lambda t, q: force_f(q)
            # Use the same step for all integrators.
            dt = 0.25 # ps
            num_steps = 5000

            integrators = []
            integrators.append(RuthForest(params['mass'], dt, force_f, -params['init_p_max'], params['init_q_max'], max_steps=num_steps))
            integrators.append(RungeKutta4(params['mass'], dt, force_f_t, -params['init_p_max'], params['init_q_max'], max_steps=num_steps))

            for step_idx, steps in enumerate(zip(*integrators)):
                ps = []
                qs = []

                for step in steps:
                    ps.append(step[1])
                    qs.append(step[2])

                # Check all pairs.
                for i in range(len(steps)):
                    for j in range(i + 1, len(steps)):
                        assert_array_almost_equal([ps[i], qs[i]], [ps[j], qs[j]], err_msg='Divergence at param set {}, step {}, pair {}/{}.'.format(param_idx, step_idx, i, j))


class TrajectoryActionTest(TestCase):
    def testHarmonicOscillator(self):
        """
        Check the action for a harmonic oscillator.
        """

        ps = N.linspace(-hp['init_p_max'], hp['init_p_max'], 7) # g nm/ps mol
        qs = N.linspace(-hp['init_q_max'], hp['init_q_max'], 11) # nm
        init_ps, init_qs = meshgrid(ps, qs)

        potential_f, force_f, _ = harmonic(m=hp['mass'], omega=hp['omega'])

        ts = N.linspace(0., 4. * N.pi / hp['omega'], 10000) # ps
        dt = ts[1] - ts[0] # ps

        integrator = RuthForest(hp['mass'], 0.5 * dt, force_f, init_ps, init_qs, max_steps=2*len(ts))
        action = TrajectoryAction(hp['mass'], dt, potential_f, init_ps, init_qs)

        calculated_Ss = N.empty((len(ts), len(ps), len(qs))) # kJ ps/mol

        for i, (_, step2) in enumerate(zip(integrator, integrator)):
            _, ps, _ = step2
            calculated_Ss[i] = action.step(ps)

        _, _, _, expected_Ss = harmonic_trajectory(hp['mass'], hp['omega'], None, ts, init_ps, init_qs)

        assert_array_almost_equal(calculated_Ss, expected_Ss)


if __name__ == '__main__':
    main()
