"""
For utils used in more than one module
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
    h = np.zeros(n)
    h[:j] = 1
    return np.flipud(np.array(list(multiset_permutations(h))).astype(int))


def ordtobit(ord, n):
    """
    Integers 'order' to bits

    Args:
        ord (str): a list of integers that represents the spins
        n:  number of spins in the chain
    Returns:
         list: a list of bits that represents the spins (1 = spin pointing up, 0 = down)
    """
    return (1 - np.array([list(map(int, list(np.binary_repr(i, n)))) for i in ord]))


def bittoint2(ordbits):
    """
    Bits 'order' to integers

    Args:
        ordbits (str):  a list of bits that represents the spins (1 = spin pointing up, 0 = down)
    Returns:
         list: a list of integers
    """
    ordbits = np.flipud(ordbits)
    x = np.zeros(ordbits.shape[0])
    x1 = 0
    for i in range(0, ordbits.shape[0]):
        if ordbits.ndim == 2:
            k = 0
            for j in range(0, ordbits.shape[1]):
                if ordbits[i][j] != -1:
                    x[i] = x[i] + ordbits[i][- (j + 1)] * 2 ** j
                else:
                    x[i] = None
        else:
            k = 1
            x1 = x1 + ordbits[-i - 1] * 2 ** i
    xx = [x, x1]
    return np.flip(xx[k].astype(int))

def bittoint(ordbits):
    if ordbits.ndim == 2:
        return np.sum(np.fliplr(1 - ordbits) * (2 ** np.arange(ordbits.shape[1])), axis=1)
    else:
        return np.sum(np.fliplr(1 - ordbits) * (2 ** np.arange(ordbits.shape[0])))
