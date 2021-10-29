"""
.. include:: build_hamiltonian.md
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
    Generate the zero magnetization block 'order', namely the numbers encoding the wave functions that are in the basis.

    Args:
        n (int): the number of spins
    Returns:
         (np.array): array of numbers encoding the wave functions
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
    Creates the $\hat{H}_0$ matrix for the Magnus expension in the Stark model

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
    Creates the $\hat{V}$ matrix for the Magnus expension in the Stark model

    Args:
        n (int): number of spins in the chain
        Jx (float): xx coupling
        ord (array): an array of integers represents the spins sub-space
        bc (0 or 1): boundary conditions 0 = open, 1 = close
    Returns:
        the matrix $\hat{V}$
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


def matt_add_jz(H0, n, Jz, ord, bc):
    """
    Adds $\hat{S}^z_i\hat{S}^z_{i+1}$ interaction for an existing Hamiltonian

    Args:
        H0 (sparse matrix): existing Hamiltonian
        n (int): number of spins in the chain
        Jz (float): interaction strength
        ord (np.array): array of numbers encoding the wave functions
        bc (0 or 1): boundary conditions 0 = open, 1 = close

    Returns:
        The matrix $\hat{H}_0$ with addition of $zz$ interactions

    """
    t = time.time()
    H = H0.copy()
    l = np.flipud(ordtobit(ord, n))
    # H = H.tocsr()
    for i in range(0, n - 1 + bc):
        H.setdiag(H.diagonal() + Jz * (l[:, i] - 0.5) * (l[:, np.mod(i + 1, n)] - 0.5))
    print("\nmatt_add_jz time was: %s" % ptime(t))
    return H


def matt_add_stark(H0, n, f, a, ord, bc):
    """
    Add a linear field of strength ``f`` to an existing Hamiltonian (stark potential)

    Args:
        H0 (sparse matrix): existing Hamiltonian
        n (int): number of spins in the chain
        f (float): linear potential strength
        a (float): curvature strength
        ord (np.array): array of numbers encoding the wave functions
        bc (0 or 1): boundary conditions 0 = open, 1 = close

    Returns:

    """
    t = time.time()
    H = H0.copy()
    l = np.flipud(ordtobit(ord, n))
    pot_arr = f * np.arange(n) + a * np.arange(n) ** 2 / float(n - 1) ** 2
    pot_arr = pot_arr - np.mean(pot_arr)
    H = H.tocsr()
    if bc == 1:
        cc = n
    else:
        cc = n - 1
    for i in range(0, cc):
        H.setdiag(H.diagonal() + pot_arr[i] * (l[:, i] - 0.5))
    if bc == 0:
        # H[j, j] = H[j, j] + (f * (n - 1) / 2 - a * ((n - 1) / n) ** 2) * (l[j, n - 1] - 0.5)
        H.setdiag(H.diagonal() + pot_arr[n - 1] * (l[:, n - 1] - 0.5))  # * 0.5
    print("\nmatt_add_stark time was: %s" % ptime(t))
    return H


def matt_stark(n, Jx, Jz, f, a, ord, bc, shift=True):
    """
    Generates the Stark Hamiltonian (xxz chain with linear potential)

    Args:
        n (int): number of spins
        Jx (float): xx interaction strength
        Jz (float): zz interaction strength
        f (float): linear potential strength
        a (float): curvature strength
        ord (np.array): array of numbers encoding the wave functions
        bc (0 or 1): boundary conditions 0 = open, 1 = close
        shift (bool): shift the potential to be concentrated in the middle

    Returns:

    """
    t = time.time()
    H = spr.dok_matrix((ord.shape[0], ord.shape[0]))
    l = np.flipud(ordtobit(ord, n))
    pot_arr = f * np.arange(n) + a * np.arange(n) ** 2 / float(n - 1) ** 2
    if shift:
        pot_arr = pot_arr - np.mean(pot_arr)
    if bc == 1:
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
        if bc == 0:
            # H[j, j] = H[j, j] + (f * (n - 1) / 2 - a * ((n - 1) / n) ** 2) * (l[j, n - 1] - 0.5)
            H[j, j] = H[j, j] + 1 * pot_arr[n - 1] * (l[j, n - 1] - 0.5)
    print("\nmatt_stark time was: %s" % ptime(t))
    return H


def matt_add_imp(H0, imp, h, l, n=0):
    r"""
    Add a magentic impurity at a specific site of the chain $h\hat{S}^z_{\textrm{imp}}$

    Args:
        H0 (sparse matrix): existing Hamiltonian
        imp (int): impurity location
        h (float): impurity strength
        l (np.array): array of wave functions, encoded in bits formation

    Returns:

    """
    if l.ndim == 1:
        l = np.flipud(ordtobit(l, n))
    tz = time.time()
    H = H0.copy()
    H.setdiag(H.diagonal() + h * (l[:, imp] - 0.5))
    print("\nmatt_add_imp time was: %s" % ptime(tz))
    return H.todok()


def matt0sz(i, l):
    """
    Generates the operator $\hat{S}^z_i$

    Args:
        i (int): location of the operator
        l (np.array): array of wave functions, encoded in bits formation

    Returns:
        (sparse matrix): $\hat{S}^z_i$

    """
    h = np.zeros(l.shape[0])
    h = h + (l[:, i] - 0.5)
    return spr.dia_matrix((h, 0), shape=(l.shape[0], l.shape[0]))


def matt3sz(n, i, ord):
    """
    Generate the operator $\hat{S}^z_i$

    Args:
        n (int): number of spins
        i (int): location of the operator
        ord (np.array): array of numbers encoding the wave functions

    Returns:
        (sparse matrix): $\hat{S}^z_i$

    """
    if i < 0:
        raise ValueError('matt3sz got an input i out of the chain')
    h = np.zeros(ord.shape[0])
    l = np.flipud(ordtobit(ord, n))
    h = h + (l[:, i] - 0.5)
    return spr.dia_matrix((h, 0), shape=(ord.shape[0], ord.shape[0]))
