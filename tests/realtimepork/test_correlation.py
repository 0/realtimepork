from unittest import main, TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from realtimepork.constants import HBAR
from realtimepork.correlation import SurvivalAmplitude
from realtimepork.potentials import harmonic

from tests.tools import harmonic_parameters as hp


class SurvivalAmplitudeTest(TestCase):
    def testHarmonicOscillator(self):
        """
        Check the harmonic oscillator survival amplitude.
        """

        ho_fs = harmonic(m=hp['mass'], omega=hp['omega'])

        # Very short time.
        dt = 0.001 * np.pi / hp['omega']  # ps
        num_steps = 51
        qs = np.linspace(-110., 110., 11)  # nm
        wf_qs = np.linspace(-110., 110., 37)  # nm
        # Exact (unnormalized) harmonic oscillator wavefunction.
        wf = np.exp(-hp['mass'] * hp['omega'] * wf_qs * wf_qs / (2. * HBAR))

        sa_gen = SurvivalAmplitude(hp['gamma'], hp['mass'], dt, ho_fs, qs, wf_qs, wf, max_steps=num_steps)
        ts = np.empty(num_steps)  # ps
        sas = np.empty(num_steps, dtype=complex)  # 1

        for i, (t, amp) in enumerate(sa_gen):
            ts[i] = t
            sas[i] = amp

        exact = np.exp(-0.5j * hp['omega'] * ts)

        assert_array_almost_equal(sas, exact)


if __name__ == '__main__':
    main()
