"""
Real-time correlation.
"""

from . import gpu

if gpu.is_enabled():
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray

import numpy as np

from .constants import HBAR
from .gpu import carve_array
from .semiclassical import SemiclassicalTrajectory
from .tools import meshgrid


class ThresholdExceededError(Exception):
    def __init__(self, t, sa):
        self.t = t
        self.sa = sa


class _SurvivalAmplitude:
    """
    Calculate survival amplitude of the ground state using Herman-Kluk SC-IVR.
    """

    # Amplitude after which we assume we've broken down entirely and there's no
    # sense whatsoever in carrying on.
    ABORT_THRESHOLD = 5.

    def __init__(self, gamma, mass, dt, potential_fs, q_grid, wf_q_grid, wf, max_steps=None, p_sigma=None):
        """
        Parameters:
          gamma: Coherent state width (1/nm^2).
          mass: Mass of particle (g/mol).
          dt: Time step (ps).
          potential_fs: Tuple of functions describing the potential
                        (nm -> kJ/mol, nm -> kJ/nm mol, nm -> kJ/nm^2 mol).
          q_grid: Evenly spaced grid of position points (nm).
          wf_q_grid: Evenly spaced grid of position points for the wavefunction
                     (nm).
          wf: Ground state wavefunction evaluated on wf_q_grid.
          max_steps: Number of steps after which to terminate.
          p_sigma: Standard deviation of momentum smoothing (g nm/ps mol).
        """

        assert len(q_grid) > 1, 'More than one position grid point required.'
        assert len(wf_q_grid) > 1, 'More than one position grid point required.'
        assert len(wf) == len(wf_q_grid), 'Wavefunction is not on position grid.'

        # Normalize the wavefunction the way we require (with the square root
        # of the volume element included in the wavefunction).
        self._wf = wf / np.sqrt(np.sum(wf * wf))  # 1

        self._gamma = gamma  # 1/nm^2
        self._wf_q_grid = wf_q_grid  # nm
        self._max_steps = max_steps

        dq = q_grid[1] - q_grid[0]  # nm
        wf_dq = self._wf_q_grid[1] - self._wf_q_grid[0]  # nm
        p_max = np.pi * HBAR / wf_dq  # g nm/ps mol
        p_grid = np.linspace(-p_max, p_max, len(self._wf_q_grid))  # g nm/ps mol
        dp = p_grid[1] - p_grid[0]  # g nm/ps mol

        self._cur_step = 0

        # One fewer wf_dq than there are position grid integrations due to the
        # normalization of wf.
        self._C = dp * dq * wf_dq * np.sqrt(self._gamma / np.pi) / (2. * np.pi * HBAR)  # 1

        self._init(len(p_grid), len(q_grid), len(self._wf_q_grid))

        mesh_ps, mesh_qs = meshgrid(p_grid, q_grid, sparse=False)
        self._trajs = SemiclassicalTrajectory(self._gamma, mass, dt, potential_fs, mesh_ps, mesh_qs)
        self._transformed_wf0 = self._transform_wf(mesh_ps, mesh_qs).conj()  # 1

        if p_sigma is not None:
            var = p_sigma * p_sigma
            self._p_smoothing = np.exp(-0.5 * p_grid * p_grid / var)
        else:
            self._p_smoothing = np.ones_like(p_grid)

    def _init(self, pn, qn, wf_qn):
        pass

    def _transform_wf(self, ps, qs):
        """
        Perform one of the position integrals.

        Parameters:
          ps: Momentum grid (g nm/ps mol).
          qs: Position grid (nm).
        """

        result = np.zeros(np.broadcast(ps, qs).shape, dtype=complex)  # 1

        for q_j, wf_j in zip(self._wf_q_grid, self._wf):
            qdiff = q_j - qs  # nm
            result += np.exp(-0.5 * self._gamma * qdiff * qdiff + 1j / HBAR * ps * qdiff) * wf_j

        return result

    def __iter__(self) -> '_SurvivalAmplitude':
        return self

    def __next__(self) -> '(ps, 1)':
        if self._max_steps is not None and self._cur_step >= self._max_steps:
            raise StopIteration

        self._cur_step += 1

        t, ps, qs, Rs, Ss = next(self._trajs)
        consts = Rs * np.exp(1j / HBAR * Ss)  # 1
        transformed_wf = self._transform_wf(ps, qs)  # 1

        # Perform the final integrals over p and q.
        sa = self._C * np.sum(self._p_smoothing * np.sum(consts * transformed_wf * self._transformed_wf0, axis=1))

        if abs(sa) >= self.ABORT_THRESHOLD:
            raise ThresholdExceededError(t, sa)

        return t, sa


class _SurvivalAmplitudeGPU(_SurvivalAmplitude):
    def _init(self, pn, qn, wf_qn):
        super()._init(pn, qn, wf_qn)

        self._wf_q_grid_gpu = gpuarray.to_gpu(np.ascontiguousarray(self._wf_q_grid))
        self._wf_gpu = gpuarray.to_gpu(np.ascontiguousarray(self._wf))

        mod = SourceModule("""
            __global__ void transform(double *ps, double *qs, double *wf_q_grid, double *wf, double *out_real, double *out_imag) {{
                int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
                int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
                int idx = idx_x + idx_y * {qn};
                double qdiff, prefactor, s, c;

                if (idx_x >= {qn} || idx_y >= {pn})
                    return;

                for (int j = 0; j < {wf_qn}; j++) {{
                    qdiff = wf_q_grid[j] - qs[idx];
                    prefactor = exp({g} * qdiff * qdiff) * wf[j];
                    sincos({h} * ps[idx] * qdiff, &s, &c);

                    out_real[idx] += prefactor * c;
                    out_imag[idx] += prefactor * s;
                }}
            }}
        """.format(g=-0.5 * self._gamma, h=1. / HBAR, pn=pn, qn=qn, wf_qn=wf_qn))
        self._kernel = mod.get_function('transform')
        self._kernel.prepare('PPPPPP')

        self._gpu_grid, self._gpu_block = carve_array(qn, pn)

    def _transform_wf(self, ps, qs):
        result_real_gpu = gpuarray.zeros(np.broadcast(ps, qs).shape, np.double)
        result_imag_gpu = gpuarray.zeros_like(result_real_gpu)

        self._kernel.prepared_call(self._gpu_grid, self._gpu_block,
                                   gpuarray.to_gpu(np.ascontiguousarray(ps)).gpudata,
                                   gpuarray.to_gpu(np.ascontiguousarray(qs)).gpudata,
                                   self._wf_q_grid_gpu.gpudata,
                                   self._wf_gpu.gpudata,
                                   result_real_gpu.gpudata,
                                   result_imag_gpu.gpudata,
                                   )

        return result_real_gpu.get() + 1j * result_imag_gpu.get()


if gpu.is_enabled():
    SurvivalAmplitude = _SurvivalAmplitudeGPU
else:
    SurvivalAmplitude = _SurvivalAmplitude
