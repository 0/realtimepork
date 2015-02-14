# realtimepork

> PIGS wavefunctions combined with HK semiclassical propagation.

This is a Python 3 package for finding real-time correlation functions of 1-dimensional quantum systems using PIGS (Path Integral Ground State) and HK (Herman-Kluk).
Any method can be used to obtain the ground state wavefunctions, but the intent is to use PIGS, possibly in the form of numerical matrix multiplication (see [pathintmatmult](https://github.com/0/pathintmatmult)).
The Herman-Kluk (SC-IVR) propagator is used to approximate the quantum real-time propagator.


## PyCUDA (GPU)

It is possible to speed up computation by doing some of it in parallel on a GPU using PyCUDA.
To enable this, call `realtimepork.gpu.enable()` _before_ importing any other modules.
In the examples, this is done with the `--gpu` option.
To select a specific device, use any of the usual PyCUDA methods (e.g. set `CUDA_DEVICE`).


## Tests

Tests are found in the `/tests/` directory and can be run with `nosetests`.


## License

Provided under the terms of the MIT license.
See LICENSE.txt for more information.
