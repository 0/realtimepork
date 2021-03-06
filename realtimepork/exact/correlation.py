"""
Real-time correlation with the exact propagator.
"""

from .. import gpu

if gpu.is_enabled():
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray

import numpy as np

from ..constants import HBAR
from ..correlation import _SurvivalAmplitude as _SAHK, ThresholdExceededError
from ..gpu import carve_array
from ..tools import meshgrid


class _SurvivalAmplitude(_SAHK):
    """
    Calculate survival amplitude of the ground state using the exact propagator
    expressed as a truncated sum-over-states.
    """

    def __init__(self, gamma, dt, q_grid, wf_q_grid, wfs, energies, max_steps=None):
        """
        Parameters:
          gamma: Coherent state width (1/nm^2).
          dt: Time step (ps).
          q_grid: Evenly spaced grid of position points (nm).
          wf_q_grid: Evenly spaced grid of position points for the wavefunction
                     (nm).
          wfs: Lowest-energy wavefunctions evaluated on wf_q_grid.
          energies: Energies corresponding to wfs (kJ/mol).
          max_steps: Number of steps after which to terminate.
        """

        assert len(q_grid) > 1, 'More than one position grid point required.'
        assert len(wf_q_grid) > 1, 'More than one position grid point required.'
        assert wfs.shape[1] == len(wf_q_grid), 'Wavefunctions are not on position grid.'

        # Normalize the wavefunctions the way we require (with the square root
        # of the volume element included in the wavefunction).
        self._wfs = (wfs.T / np.sqrt(np.sum(wfs * wfs, axis=1))).T  # 1

        self._gamma = gamma  # 1/nm^2
        self._dt = dt  # ps
        self._q_grid = q_grid  # nm
        self._wf_q_grid = wf_q_grid  # nm
        self._energies = energies  # kJ/mol
        self._max_steps = max_steps

        dq = self._q_grid[1] - self._q_grid[0]  # nm
        wf_dq = self._wf_q_grid[1] - self._wf_q_grid[0]  # nm
        p_max = np.pi * HBAR / wf_dq  # g nm/ps mol
        self._p_grid = np.linspace(-p_max, p_max, len(self._wf_q_grid))  # g nm/ps mol
        dp = self._p_grid[1] - self._p_grid[0]  # g nm/ps mol

        self._t = 0.  # ps
        self._cur_step = 0

        # Two fewer wf_dqs than there are position grid integrations due to the
        # normalization of wfs.
        self._C = dp * dq * wf_dq * np.sqrt(self._gamma / np.pi) / (2. * np.pi * HBAR)  # 1

        self._init(len(self._p_grid), len(self._q_grid), len(self._wf_q_grid))

        self._mesh_ps, self._mesh_qs = meshgrid(self._p_grid, self._q_grid)
        self._transformed_wf0 = self._transform_wf(self._t).conj()  # 1

    def _transform_wf(self, t):
        """
        Perform one of the position integrals.

        Parameters:
          t: Time (ps).
        """

        phases = np.exp(-1j / HBAR * self._energies * t)  # 1
        result = np.zeros(np.broadcast(self._mesh_ps, self._mesh_qs).shape, dtype=complex)  # 1

        for n, phase in enumerate(phases):
            for q_alpha, wf_n_alpha in zip(self._wf_q_grid, self._wfs[n]):
                qdiff = q_alpha - self._mesh_qs  # nm
                prefactor = phase * wf_n_alpha * np.exp(-self._gamma / 2. * qdiff * qdiff + 1j / HBAR * self._mesh_ps * qdiff)

                for wf_0_j, wf_n_j in zip(self._wfs[0], self._wfs[n]):
                    result += prefactor * wf_0_j * wf_n_j

        return result

    def __next__(self) -> '(ps, 1)':
        if self._max_steps is not None and self._cur_step >= self._max_steps:
            raise StopIteration

        self._cur_step += 1

        transformed_wf = self._transform_wf(self._t)  # 1

        # Perform the final integrals over p and q.
        sa = self._C * np.sum(transformed_wf * self._transformed_wf0)

        if abs(sa) >= self.ABORT_THRESHOLD:
            raise ThresholdExceededError(self._t, sa)

        result = self._t, sa

        self._t += self._dt

        return result


class _SurvivalAmplitudeGPU(_SurvivalAmplitude):
    def _init(self, pn, qn, wf_qn):
        super()._init(pn, qn, wf_qn)

        self._p_grid_gpu = gpuarray.to_gpu(np.ascontiguousarray(self._p_grid))
        self._q_grid_gpu = gpuarray.to_gpu(np.ascontiguousarray(self._q_grid))
        self._wf_q_grid_gpu = gpuarray.to_gpu(np.ascontiguousarray(self._wf_q_grid))
        self._wfs_gpu = gpuarray.to_gpu(np.ascontiguousarray(self._wfs))
        self._energies_gpu = gpuarray.to_gpu(np.ascontiguousarray(self._energies))

        mod = SourceModule("""
            __global__ void transform(double *p_grid, double *q_grid, double *wf_q_grid, double *wfs, double *energies, double t, double *out_real, double *out_imag) {{
                int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
                int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
                int idx = idx_x + idx_y * {qn};
                double qdiff, prefactor1, prefactor2, s, c;

                if (idx_x >= {qn} || idx_y >= {pn})
                    return;

                for (int n = 0; n < {wfn}; n++) {{
                    for (int alpha = 0; alpha < {wf_qn}; alpha++) {{
                        qdiff = wf_q_grid[alpha] - q_grid[idx_x];
                        prefactor1 = wfs[alpha + n * {qn}] * exp({g} * qdiff * qdiff);
                        sincos({h} * (p_grid[idx_y] * qdiff - energies[n] * t), &s, &c);

                        for (int j = 0; j < {wf_qn}; j++) {{
                            prefactor2 = prefactor1 * wfs[j] * wfs[j + n * {wf_qn}];

                            out_real[idx] += prefactor2 * c;
                            out_imag[idx] += prefactor2 * s;
                        }}
                    }}
                }}
            }}
        """.format(g=-0.5 * self._gamma, h=1. / HBAR, wfn=len(self._wfs), pn=pn, qn=qn, wf_qn=wf_qn))
        self._kernel = mod.get_function('transform')
        self._kernel.prepare('PPPPPdPP')

        self._gpu_grid, self._gpu_block = carve_array(qn, pn)

    def _transform_wf(self, t):
        result_real_gpu = gpuarray.zeros((len(self._p_grid), len(self._q_grid)), np.double)
        result_imag_gpu = gpuarray.zeros_like(result_real_gpu)

        self._kernel.prepared_call(self._gpu_grid, self._gpu_block,
                                   self._p_grid_gpu.gpudata,
                                   self._q_grid_gpu.gpudata,
                                   self._wf_q_grid_gpu.gpudata,
                                   self._wfs_gpu.gpudata,
                                   self._energies_gpu.gpudata,
                                   t,
                                   result_real_gpu.gpudata,
                                   result_imag_gpu.gpudata,
                                   )

        return result_real_gpu.get() + 1j * result_imag_gpu.get()


if gpu.is_enabled():
    SurvivalAmplitude = _SurvivalAmplitudeGPU
else:
    SurvivalAmplitude = _SurvivalAmplitude
