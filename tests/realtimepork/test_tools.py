from unittest import main, TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from realtimepork.tools import SignedSqrt, signed_sqrt


class SignedSqrtTest(TestCase):
    def _gen_traj(self, periods):
        """
        Generate a triangle wave trajectory that crosses over the real line
        twice per period.
        """

        A = np.linspace(0., 1., 9)[:-1]
        B = np.linspace(1., -1., 5)[:-1]

        R = np.concatenate([A + i for i in range(periods)])
        I = np.concatenate([np.concatenate((B, -B))] * periods)

        trajectory = R + 1j * I
        factors = np.array([1, 1, 1, -1, -1, -1, 1, 1] * periods)

        return trajectory, factors

    def testPositive(self):
        """
        Make sure we don't trigger the flip over the positive real line.
        """
        traj, _ = self._gen_traj(3)
        traj = traj[:, np.newaxis]
        ss = SignedSqrt()
        rooted1 = np.empty_like(traj)
        rooted2 = signed_sqrt(traj)

        for i, v in enumerate(traj):
            rooted1[i] = ss(v)

        assert_array_almost_equal(rooted1, np.sqrt(traj))
        assert_array_equal(rooted1, rooted2)

    def testNegative(self):
        """
        Make sure we do trigger the flip over the negative real line.
        """

        traj, factors = self._gen_traj(3)
        traj = traj[:, np.newaxis]
        factors = factors[:, np.newaxis]
        # Reflect the trajectory over the y-axis.
        traj = -traj.conj()
        ss = SignedSqrt()
        rooted1 = np.empty_like(traj)
        rooted2 = signed_sqrt(traj)

        for i, v in enumerate(traj):
            rooted1[i] = ss(v)

        assert_array_almost_equal(rooted1, factors * np.sqrt(traj))
        assert_array_equal(rooted1, rooted2)

    def testCircle(self):
        """
        Check a circle in the complex plane.
        """

        ts = np.linspace(0., 4. * np.pi, 401)
        traj = (np.cos(ts) + 1j * np.sin(ts))[:, np.newaxis]
        ss = SignedSqrt()
        rooted1 = np.empty_like(traj)
        rooted2 = signed_sqrt(traj)

        for i, v in enumerate(traj):
            rooted1[i] = ss(v)

        factors = np.concatenate((np.ones(101), -np.ones(200), np.ones(100)))[:, np.newaxis]
        # The trigonometric functions will probably not give us exactly zero,
        # so tweak things a bit at the crossings to account for implementation
        # details.
        if traj[100] < 0:
            factors[100] = -1
        if traj[300] < 0:
            factors[300] = -1

        assert_array_almost_equal(rooted1, factors * np.sqrt(traj))
        assert_array_equal(rooted1, rooted2)

    def testMultiple(self):
        """
        Make sure we handle multiple axes properly.
        """

        base_traj, factors = self._gen_traj(3)
        trajs = np.multiply.outer(-base_traj.conj(), np.array([[1., 2.], [3., 4.]]))

        ss = SignedSqrt()
        rooted1 = np.empty_like(trajs)
        rooted2 = signed_sqrt(trajs)

        for i, v in enumerate(trajs):
            rooted1[i] = ss(v)

        assert_array_almost_equal(rooted1, (factors * np.sqrt(trajs).T).T)
        assert_array_equal(rooted1, rooted2)


if __name__ == '__main__':
    main()
