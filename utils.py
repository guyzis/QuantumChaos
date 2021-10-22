"""
For utils used in more than one module
"""

import time
import numpy as np

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


def ordtobit(ord, n):
    """
    Integers 'order' to bits

    Args:
        ord (str): a list of integers that represents the spins
        n:  number of spins in the chain
    Returns:
         list: a list of bits that represents the spins (1 = spin pointing up, 0 = down)
    """
    return np.array([list(map(int, list(np.binary_repr(i, n)))) for i in ord])


def bittoint(ordbits):
    """
    Bits 'order' to integers

    Args:
        ordbits (str):  a list of bits that represents the spins (1 = spin pointing up, 0 = down)
    Returns:
         list: a list of integers
    """
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
    return xx[k]