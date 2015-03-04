from unittest import main, TestCase

import numpy as N
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
        dt = 0.001 * N.pi / hp['omega'] # ps
        num_steps = 51
        qs = N.linspace(-110., 110., 75) # nm
        # Exact (unnormalized) harmonic oscillator wavefunction.
        wf = N.exp(-hp['mass'] * hp['omega'] * qs * qs / (2. * HBAR))

        sa_gen = SurvivalAmplitude(hp['gamma'], hp['mass'], dt, ho_fs, qs, wf, max_steps=num_steps)
        ts = N.empty(num_steps) # ps
        sas = N.empty(num_steps, dtype=complex) # 1

        for i, (t, amp) in enumerate(sa_gen):
            ts[i] = t
            sas[i] = amp

        exact = N.exp(-0.5j * hp['omega'] * ts)

        assert_array_almost_equal(sas, exact)


if __name__ == '__main__':
    main()
