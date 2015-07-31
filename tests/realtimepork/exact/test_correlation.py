from unittest import main, TestCase

import numpy as np
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
        dt = 12.34 * np.pi / hp['omega']  # ps
        num_steps = 3
        qs = np.linspace(-95., 95., 31)  # nm
        wf_qs = np.linspace(-93., 93., 29)  # nm
        # Exact (unnormalized) harmonic oscillator wavefunctions.
        wfn = 3
        herms = np.polynomial.hermite.hermval(np.sqrt(hp['mass'] * hp['omega'] / HBAR) * wf_qs, np.eye(wfn))
        exps = np.exp(-hp['mass'] * hp['omega'] * wf_qs * wf_qs / (2. * HBAR))
        wfs = herms * exps
        energies = HBAR * hp['omega'] * (0.5 + np.arange(wfn))  # kJ/mol

        sa_gen = SurvivalAmplitude(hp['gamma'], dt, qs, wf_qs, wfs, energies, max_steps=num_steps)
        ts = np.empty(num_steps)  # ps
        sas = np.empty(num_steps, dtype=complex)  # 1

        for i, (t, amp) in enumerate(sa_gen):
            ts[i] = t
            sas[i] = amp

        exact = np.exp(-0.5j * hp['omega'] * ts)

        assert_array_almost_equal(sas, exact)


if __name__ == '__main__':
    main()
