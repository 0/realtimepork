from unittest import main, TestCase

import numpy as N
from numpy.testing import assert_array_almost_equal

from realtimepork.constants import HBAR
from realtimepork.potentials import harmonic
from realtimepork.semiclassical import HermanKlukPrefactor, SemiclassicalTrajectory
from realtimepork.tools import meshgrid

from tests.tools import harmonic_hk, harmonic_parameters as hp, harmonic_trajectory


class HermanKlukPrefactorTest(TestCase):
    def testHarmonicOscillator(self):
        """
        Check the Herman-Kluk prefactor for a harmonic oscillator.
        """

        ps = N.linspace(-hp['init_p_max'], hp['init_p_max'], 7)  # g nm/ps mol
        qs = N.linspace(-hp['init_q_max'], hp['init_q_max'], 11)  # nm
        init_ps, init_qs = meshgrid(ps, qs)
        shape = N.broadcast(init_ps, init_qs).shape

        ts = N.linspace(0., 4. * N.pi / hp['omega'], 500)  # ps
        dt = ts[1] - ts[0]  # ps
        omts = hp['omega'] * ts  # 1

        _, _, hessian_f = harmonic(m=hp['mass'], omega=hp['omega'])
        hkp = HermanKlukPrefactor(hp['gamma'], hp['mass'], dt, hessian_f, shape)

        calculated = N.empty((len(ts), len(ps), len(qs)), dtype=complex)

        for i in range(len(ts)):
            # The harmonic oscillator Hessian is constant, so we plug in some
            # dummy values here.
            calculated[i] = hkp.step(N.array([1.0, 1.0, 1.0]))

        exact = harmonic_hk(hp['mass'], hp['omega'], hp['gamma'], ts, shape)

        assert_array_almost_equal(calculated, exact)


class SemiclassicalTrajectoryTest(TestCase):
    def testHarmonicOscillator(self):
        """
        Check the semiclassical trajectory for a harmonic oscillator.
        """

        ps = N.linspace(-hp['init_p_max'], hp['init_p_max'], 7)  # g nm/ps mol
        qs = N.linspace(-hp['init_q_max'], hp['init_q_max'], 11)  # nm
        init_ps, init_qs = meshgrid(ps, qs)
        shape = N.broadcast(init_ps, init_qs).shape

        ho_fs = harmonic(m=hp['mass'], omega=hp['omega'])

        ts = N.linspace(0., 4. * N.pi / hp['omega'], 6000)  # ps
        dt = ts[1] - ts[0]  # ps
        omts = hp['omega'] * ts  # 1

        traj = SemiclassicalTrajectory(hp['gamma'], hp['mass'], dt, ho_fs, init_ps, init_qs, max_steps=len(ts))

        calculated_p = N.empty((len(ts), len(ps), len(qs)))  # g nm/ps mol
        calculated_q = N.empty_like(calculated_p)  # nm
        calculated_R = N.empty_like(calculated_p, dtype=complex)  # 1
        calculated_S = N.empty_like(calculated_p)  # kJ ps/mol

        for i, step in enumerate(traj):
            _, calculated_p[i], calculated_q[i], calculated_R[i], calculated_S[i] = step

        exact_ps, exact_qs, exact_Rs, exact_Ss = harmonic_trajectory(hp['mass'], hp['omega'], hp['gamma'], ts, init_ps, init_qs)

        assert_array_almost_equal(calculated_p, exact_ps)
        assert_array_almost_equal(calculated_q, exact_qs)
        assert_array_almost_equal(calculated_R, exact_Rs)
        assert_array_almost_equal(calculated_S, exact_Ss)


if __name__ == '__main__':
    main()
