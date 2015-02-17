"""
Assorted tools.
"""

import numpy as N


def meshgrid(*args, indexing='ij', sparse=True, copy=False, **kwargs):
    """
    Wrapper for NumPy's meshgrid with useful defaults.
    """

    return N.meshgrid(*args, indexing=indexing, sparse=sparse, copy=copy, **kwargs)
