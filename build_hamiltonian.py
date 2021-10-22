"""
Utils used to generate the Stark Hamiltonian Matrices
"""

import numpy as np
from scipy import sparse as spr
import time
import sys
import unittest
import os
import shutil
from utils import *


def block0(n):
    """
    Generate the zero magnetization block 'order'

    Args:
        n (int): the number of spins
    Returns:
         the block0 'order' in integers
    """
    t = time.time()
    h = np.zeros(2 ** n)
    l = np.flipud(ordtobit(np.arange(2 ** n), n))
    for i in range(0, n):
        h = h + (l[:, i] - 0.5)
    ord = np.where(h == 0)[0]
    print("block0 time was %s" % ptime(t))
    return ord



def matth0(n, Jz, ord, bc):
    """
    Creates the H_0 matrix

    Args:
        n (int): number of spins in the chain
        Jz (float): zz coupling
        ord (array): an array of integers represents the spins sub-space
        bc (0 or 1): boundary conditions 0 = open, 1 = close
    Returns:
         The matrix H_0
    """
    t = time.time()
    H = spr.dok_matrix((ord.shape[0], ord.shape[0]))
    l = np.flipud(ordtobit(ord, n))
    if bc == 1:
        cc = n
    else:
        cc = n - 1
    for i in range(0, cc):
        for j in range(0, ord.shape[0]):
            H[j, j] = H[j, j] + Jz * (l[j, i] - 0.5) * (l[j, np.mod(i + 1, n)] - 0.5)
    print("\nmatth0 time was: %s" % ptime(t))
    return H



def mattv(n, Jx, ord, c):
    """
    Creates the V matrix

    Args:
        n (int): number of spins in the chain
        Jx (float): xx coupling
        ord (array): an array of integers represents the spins sub-space
        bc (0 or 1): boundary conditions 0 = open, 1 = close
    Returns:
        the matrix V
    """
    t = time.time()
    H = spr.dok_matrix((ord.shape[0], ord.shape[0]))
    l = np.flipud(ordtobit(ord, n))
    if c == 1:
        cc = n
    else:
        cc = n - 1
    for i in range(0, cc):
        for j in range(0, ord.shape[0]):
            l1 = l[j].copy()
            if l[j, i] == 0 and l[j, np.mod(i + 1, n)] == 1:
                l1[i] = 1
                l1[np.mod(i + 1, n)] = 0
                k1 = int(np.argwhere(bittoint(l1) == np.flip(ord)))
                H[k1, j] = H[k1, j] + Jx * 0.5  # * (-1) ** l[j, i]
    print("\nmattv time was: %s" % ptime(t))
    return H

def mattaddjz(H0, n, Jz, ord, c):
    t = time.time()
    H = H0.copy()
    l = np.flipud(ordtobit(ord, n))
    # H = H.tocsr()
    for i in range(0, n - 1 + c):
        H.setdiag(H.diagonal() + Jz * (l[:, i] - 0.5) * (l[:, np.mod(i + 1, n)] - 0.5))
    print("\nmattaddjz time was: %s" % ptime(t))
    return H

def mattaddstark2(H0, n, f, a, ord, c):
    t = time.time()
    H = H0.copy()
    l = np.flipud(ordtobit(ord, n))
    pot_arr = f * np.arange(n) + a * np.arange(n) ** 2 / float(n - 1) ** 2
    pot_arr = pot_arr - np.mean(pot_arr)
    H = H.tocsr()
    if c == 1:
        cc = n
    else:
        cc = n - 1
    for i in range(0, cc):
        H.setdiag(H.diagonal() + pot_arr[i] * (l[:, i] - 0.5))
    if c == 0:
        # H[j, j] = H[j, j] + (f * (n - 1) / 2 - a * ((n - 1) / n) ** 2) * (l[j, n - 1] - 0.5)
        H.setdiag(H.diagonal() + pot_arr[n - 1] * (l[:, n - 1] - 0.5))  # * 0.5
    print("\nmattaddstark2 time was: %s" % ptime(t))
    return H


def matt32(n, Jx, Jz, f, a, ord, c, shift=True):
    t = time.time()
    H = spr.dok_matrix((ord.shape[0], ord.shape[0]))
    l = np.flipud(ordtobit(ord, n))
    pot_arr = f * np.arange(n) + a * np.arange(n) ** 2 / float(n - 1) ** 2
    if shift:
        pot_arr = pot_arr - np.mean(pot_arr)
    if c == 1:
        cc = n
    else:
        cc = n - 1
    for i in range(0, cc):
        for j in range(0, ord.shape[0]):
            # H[j, j] = H[j, j] + (f * i * (1 - 0.5 * int(i == 0)) - a * (i / n) ** 2) * (l[j, i] - 0.5)
            H[j, j] = H[j, j] + pot_arr[i] * (l[j, i] - 0.5)  # * (1 - 0.5 * int(i == 0))
            H[j, j] = H[j, j] + Jz * (l[j, i] - 0.5) * (l[j, np.mod(i + 1, n)] - 0.5)
            l1 = l[j].copy()
            l2 = l[j].copy()
            if l[j, i] == 0 and l[j, np.mod(i + 1, n)] == 1:
                l1[i] = 1
                l1[np.mod(i + 1, n)] = 0
                k1 = int(np.argwhere(bittoint(l1) == np.flip(ord)))
                H[k1, j] = H[k1, j] + Jx * 0.5
            elif l[j, i] == 1 and l[j, np.mod(i + 1, n)] == 0:
                l2[i] = 0
                l2[np.mod(i + 1, n)] = 1
                k2 = int(np.argwhere(bittoint(l2) == np.flip(ord)))
                H[k2, j] = H[k2, j] + Jx * 0.5
    for j in range(0, ord.shape[0]):
        if c == 0:
            # H[j, j] = H[j, j] + (f * (n - 1) / 2 - a * ((n - 1) / n) ** 2) * (l[j, n - 1] - 0.5)
            H[j, j] = H[j, j] + 1 * pot_arr[n - 1] * (l[j, n - 1] - 0.5)
    print("\nmatt32 time was: %s" % ptime(t))
    return H
