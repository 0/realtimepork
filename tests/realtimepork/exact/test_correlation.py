from unittest import main, TestCase

import numpy as N
from numpy.testing import assert_array_almost_equal

from realtimepork.constants import HBAR, KB
from realtimepork.exact.correlation import SurvivalAmplitude

from tests.tools import harmonic_parameters as hp


class SurvivalAmplitudeTest(TestCase):
    def testHarmonicOscillator(self):
        """
        Check the harmonic oscillator survival amplitude.
        """

        # Very long steps, because we are not constrained by classical
        # trajectory integrators.
        dt = 12.34 * N.pi / hp['omega'] # ps
        num_steps = 3
        qs = N.linspace(-150., 150., 55) # nm
        # Exact (unnormalized) harmonic oscillator wavefunctions.
        wfn = 3
        herms = N.polynomial.hermite.hermval(N.sqrt(hp['mass'] * hp['omega'] / HBAR) * qs, N.eye(wfn))
        exps = N.exp(-hp['mass'] * hp['omega'] * qs * qs / (2. * HBAR))
        wfs = herms * exps
        energies = HBAR * hp['omega'] * (0.5 + N.arange(wfn)) # kJ/mol

        sa_gen = SurvivalAmplitude(hp['gamma'], dt, qs, wfs, energies, max_steps=num_steps)
        ts = N.empty(num_steps) # ps
        sas = N.empty(num_steps, dtype=complex) # 1

        for i, (t, amp) in enumerate(sa_gen):
            ts[i] = t
            sas[i] = amp

        exact = N.exp(-0.5j * hp['omega'] * ts)

        assert_array_almost_equal(sas, exact)


if __name__ == '__main__':
    main()
