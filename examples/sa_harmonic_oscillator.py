#!/usr/bin/env python3

"""
Harmonic oscillator ground state survival amplitude example.

An oscillator with an angular frequency of x kelvin has a ground state survival
amplitude that oscillates at x/2 kelvin. A natural choice for the gamma
parameter is (mass * omega / hbar), since this leads to unsqueezed coherent
states.
"""

from argparse import ArgumentParser, FileType

import numpy as N


# Parse arguments.
p = ArgumentParser(description='Calculate the HO ground state survival amplitude.')
p_config = p.add_argument_group('configuration')

p_config.add_argument('--mass', metavar='M', type=float, required=True, help='particle mass (electron masses)')
p_config.add_argument('--omega', metavar='W', type=float, required=True, help='angular frequency (K)')
p_config.add_argument('--gamma', metavar='G', type=float, required=True, help='coherent state width (1/nm^2)')
p_config.add_argument('--dp', metavar='P', type=float, required=True, help='spacing of momentum grid (g nm/ps mol)')
p_config.add_argument('--p-max', metavar='P', type=float, help='range of momentum grid (g nm/ps mol)')
p_config.add_argument('--dt', metavar='T', type=float, required=True, help='spacing of time grid (ps)')
p_config.add_argument('--steps', metavar='N', type=int, required=True, help='number of real-time steps')
p_config.add_argument('--wf-in', metavar='FILE', required=True, help='path to wavefunction')

p.add_argument('--gpu', dest='gpu', action='store_true', help='use the GPU')
p.add_argument('--no-gpu', dest='gpu', action='store_false', help="don't use the GPU (default)")

p.add_argument('--sas-out', metavar='FILE', type=FileType('w'), help='path to output values (- for stdout)')
p.add_argument('--sas-plot-out', metavar='FILE', help='path to output survival amplitude plot')
p.add_argument('--sas-spectrum-out', metavar='FILE', help='path to output spectrum plot')

args = p.parse_args()


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


mass = args.mass * ME # g/mol
omega = args.omega * KB / HBAR # 1/ps
gamma = args.gamma # 1/nm^2
dp = args.dp # g nm/ps mol
p_max = args.p_max # g nm/ps mol
dt = args.dt # ps
steps = args.steps # 1
wf_in = args.wf_in

sas_out = args.sas_out
sas_plot_out = args.sas_plot_out
sas_spectrum_out = args.sas_spectrum_out


# Calculate values.
ho_fs = harmonic(m=mass, omega=omega)

wf_in_data = N.loadtxt(wf_in)
qs = wf_in_data[:,0] # nm
wf = wf_in_data[:,1]

sa_gen = SurvivalAmplitude(gamma, mass, dt, ho_fs, dp, qs, wf, p_max=p_max, max_steps=steps)
ts = N.empty(steps)
sas = N.empty(steps, dtype=complex)

for i, (t, amp) in enumerate(sa_gen):
    ts[i] = t
    sas[i] = amp

    # Output the values as we go.
    if sas_out is not None:
        print('{} {}'.format(ts[i], sas[i]), file=sas_out)


# Obtain the spectrum.
freqs, amps = transform(ts, sas) # 1/ps, 1
freqs *= 2. * N.pi * HBAR / KB # K
peak_idx, _ = find_peak(freqs, amps)
freq_window = freqs[peak_idx - 3], freqs[peak_idx + 4]
freqs_interp, amps_interp = interpolate(sas, freqs, amps, freq_window, 64) # K, 1

print('Peak at {} K.'.format(find_peak(freqs_interp, amps_interp)[1]))


# Output plots.
if sas_plot_out:
    from realtimepork.plotting import plot_sa

    plot_sa(ts, sas, sas_plot_out, x_label='$t / \mathrm{ps}$', y_label='$S_0(t)$')

if sas_spectrum_out:
    from realtimepork.plotting import plot_spectrum

    plot_spectrum(freqs_interp, amps_interp, sas_spectrum_out, magnitude=True, freq_window=freq_window, x_label='$\omega / \mathrm{K}$', y_label='$I(\omega)$')
