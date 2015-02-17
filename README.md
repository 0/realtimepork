# realtimepork

> PIGS wavefunctions combined with HK semiclassical propagation.

This is a Python 3 package for finding real-time correlation functions of 1-dimensional quantum systems using PIGS (Path Integral Ground State) and HK (Herman-Kluk).
Any method can be used to obtain the ground state wavefunctions, but the intent is to use PIGS, possibly in the form of numerical matrix multiplication (see [pathintmatmult](https://github.com/0/pathintmatmult)).
The Herman-Kluk (SC-IVR) propagator is used to approximate the quantum real-time propagator.


## Tests

Tests are found in the `/tests/` directory and can be run with `nosetests`.


## License

Provided under the terms of the MIT license.
See LICENSE.txt for more information.
