r"""
For utils used in more than one module.

The transformation between wave-function bits and integer representation is defined as following:
$$\vec{x}\rightarrow\sum_{i=L}^1(1-x_i)\cdot 2^{L-i}=\boldsymbol{\mathrm{int}}.$$

See ``legacy_utils`` for more information about this transformation.
"""

import time
import numpy as np
from sympy.utilities.iterables import multiset_permutations


def ptime(tz):
    """
    Args:
        tz (float): a time.time() output
    Returns:
         a string with the time it took since the initial counter tz
    """
    tzz = time.time() - tz
    if tzz < 60:
        return "%1.3f sec" % tzz
    elif tzz < 3600:
        return "%1.3f min" % (tzz / 60)
    else:
        return "%1.3f hours" % (tzz / 3600)


def blockex(n, j):
    """
    Generates a basis of ``n`` spins and ``j`` excitations (spins pointing up)

    Args:
        n (int): number of spins in the chain
        j (int): number of excitations in the basis

    Returns:
        np.array: wave functions basis in bits

    """
    h = np.zeros(n)
    h[:j] = 1
    return np.flipud(np.array(list(multiset_permutations(h))).astype(int))


def ordtobit(ord, n):
    r"""
    Integers basis to bits basis (e.g $\boldsymbol{4}\rightarrow\texttt{[0,1,1]}$)

    Args:
        ord (np.array): an array of integers that represents the spins
        n:  number of spins in the chain
    Returns:
         np.array: a list of bits that represents the spins (1 = spin pointing up, 0 = down)
    """
    return 1 - np.array([list(map(int, list(np.binary_repr(i, n)))) for i in ord])


def bittoint(basis):
    r"""
    Basis in bits to integers (e.g $\texttt{[0,1,1]}\rightarrow\boldsymbol{4}$)

    Args:
        basis (np.array):  a list of bits that represents the spins (1 = spin pointing up, 0 = down)
    Returns:
         np.array: an array of integers encoding the wave function
    """
    if basis.ndim == 2:
        return np.sum(np.fliplr(1 - basis) * (2 ** np.arange(basis.shape[1])), axis=1)
    else:
        return np.sum(np.fliplr(1 - basis) * (2 ** np.arange(basis.shape[0])))
