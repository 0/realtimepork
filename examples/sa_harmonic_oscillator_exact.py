#!/usr/bin/env python3

"""
Harmonic oscillator ground state survival amplitude example using the exact
propagator.
"""

from argparse import ArgumentParser, FileType

import numpy as np


# Parse arguments.
p = ArgumentParser(description='Calculate the HO ground state survival amplitude.')
p_config = p.add_argument_group('configuration')

p_config.add_argument('--gamma', metavar='G', type=float, required=True, help='coherent state width (1/nm^2)')
p_config.add_argument('--q-max', metavar='Q', type=float, help='range of position grid (nm) (default: same as wavefunction)')
p_config.add_argument('--qn', metavar='N', type=int, help='number of position grid points (default: same as wavefunction)')
p_config.add_argument('--dt', metavar='T', type=float, required=True, help='spacing of time grid (ps)')
p_config.add_argument('--steps', metavar='N', type=int, required=True, help='number of real-time steps')
p_config.add_argument('--wfs-in', metavar='FILE', required=True, help='path to wavefunctions')
p_config.add_argument('--energies-in', metavar='FILE', required=True, help='path to energies (K)')

p.add_argument('--gpu', dest='gpu', action='store_true', help='use the GPU')
p.add_argument('--no-gpu', dest='gpu', action='store_false', help="don't use the GPU (default)")

p.add_argument('--sas-out', metavar='FILE', type=FileType('w'), help='path to output values (- for stdout)')
p.add_argument('--sas-plot-out', metavar='FILE', help='path to output survival amplitude plot')
p.add_argument('--sas-spectrum-out', metavar='FILE', help='path to output spectrum plot')

args = p.parse_args()

if len([x for x in [args.q_max, args.qn] if x is not None]) not in [0, 2]:
    from sys import exit

    print('Both --q-max and --qn must be given.')
    exit(1)


# Import now that we know whether to use the GPU.
if args.gpu:
    from realtimepork import gpu

    try:
        gpu.enable()
    except gpu.PyCUDAMissingError:
        from sys import exit

        print('Cannot load PyCUDA for --gpu option!')
        exit(1)

from realtimepork.constants import KB, HBAR
from realtimepork.exact.correlation import SurvivalAmplitude
from realtimepork.spectrum import find_peak, interpolate, transform


gamma = args.gamma  # 1/nm^2
q_max = args.q_max  # nm
qn = args.qn  # 1
dt = args.dt  # ps
steps = args.steps  # 1
wfs_in = args.wfs_in
energies_in = args.energies_in

sas_out = args.sas_out
sas_plot_out = args.sas_plot_out
sas_spectrum_out = args.sas_spectrum_out


# Calculate values.
wfs_in_data = np.loadtxt(wfs_in)
wf_qs = wfs_in_data[:, 0]  # nm
wfs = wfs_in_data[:, 1:].T
energies = np.loadtxt(energies_in, ndmin=1) * KB  # kJ/mol

if q_max is None:
    qs = wf_qs
else:
    qs = np.linspace(-q_max, q_max, qn)  # nm

sa_gen = SurvivalAmplitude(gamma, dt, qs, wf_qs, wfs, energies, max_steps=steps)
ts = np.empty(steps)
sas = np.empty(steps, dtype=complex)

for i, (t, amp) in enumerate(sa_gen):
    ts[i] = t
    sas[i] = amp

    # Output the values as we go.
    if sas_out is not None:
        print('{} {}'.format(ts[i], sas[i]), file=sas_out)


# Obtain the spectrum.
freqs, amps = transform(ts, sas)  # 1/ps, 1
freqs *= 2. * np.pi * HBAR / KB  # K
peak_idx, _ = find_peak(freqs, amps)
freq_window = freqs[peak_idx - 3], freqs[peak_idx + 4]
freqs_interp, amps_interp = interpolate(sas, freqs, amps, freq_window, 64)  # K, 1

print('Peak at {} K.'.format(find_peak(freqs_interp, amps_interp)[1]))


# Output plots.
if sas_plot_out:
    from realtimepork.plotting import plot_sa

    plot_sa(ts, sas, sas_plot_out, x_label='$t / \mathrm{ps}$', y_label='$S_0(t)$')

if sas_spectrum_out:
    from realtimepork.plotting import plot_spectrum

    plot_spectrum(freqs_interp, amps_interp, sas_spectrum_out, magnitude=True,
                  freq_window=freq_window,
                  x_label='$\omega / \mathrm{K}$', y_label='$I(\omega)$')
