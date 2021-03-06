#!/usr/bin/env python3

"""
Harmonic oscillator ground state survival amplitude example.

An oscillator with an angular frequency of x kelvin has a ground state survival
amplitude that oscillates at x/2 kelvin. A natural choice for the gamma
parameter is (mass * omega / hbar), since this leads to unsqueezed coherent
states.
"""

from argparse import ArgumentParser, FileType

import numpy as np


# Parse arguments.
p = ArgumentParser(description='Calculate the HO ground state survival amplitude.')
p_config = p.add_argument_group('configuration')

p_config.add_argument('--mass', metavar='M', type=float, required=True, help='particle mass (electron masses)')
p_config.add_argument('--omega', metavar='W', type=float, required=True, help='angular frequency (K)')
p_config.add_argument('--gamma', metavar='G', type=float, required=True, help='coherent state width (1/nm^2)')
p_config.add_argument('--q-max', metavar='Q', type=float, help='range of position grid (nm) (default: same as wavefunction)')
p_config.add_argument('--qn', metavar='N', type=int, help='number of position grid points (default: same as wavefunction)')
p_config.add_argument('--dt', metavar='T', type=float, required=True, help='spacing of time grid (ps)')
p_config.add_argument('--steps', metavar='N', type=int, required=True, help='number of real-time steps')
p_config.add_argument('--wf-in', metavar='FILE', required=True, help='path to wavefunction')
p_config.add_argument('--p-sigma', metavar='S', type=float, help='standard deviation of momentum smoothing (g nm/ps mol; default: none)')

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

from realtimepork.constants import KB, HBAR, ME
from realtimepork.correlation import SurvivalAmplitude
from realtimepork.potentials import harmonic
from realtimepork.spectrum import find_peak, interpolate, transform


mass = args.mass * ME  # g/mol
omega = args.omega * KB / HBAR  # 1/ps
gamma = args.gamma  # 1/nm^2
q_max = args.q_max  # nm
qn = args.qn  # 1
dt = args.dt  # ps
steps = args.steps  # 1
wf_in = args.wf_in
p_sigma = args.p_sigma  # g nm/ps mol

sas_out = args.sas_out
sas_plot_out = args.sas_plot_out
sas_spectrum_out = args.sas_spectrum_out


# Calculate values.
ho_fs = harmonic(m=mass, omega=omega)

wf_in_data = np.loadtxt(wf_in)
wf_qs = wf_in_data[:, 0]  # nm
wf = wf_in_data[:, 1]

if q_max is None:
    qs = wf_qs
else:
    qs = np.linspace(-q_max, q_max, qn)  # nm

sa_gen = SurvivalAmplitude(gamma, mass, dt, ho_fs, qs, wf_qs, wf, max_steps=steps, p_sigma=p_sigma)
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
