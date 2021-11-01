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


def matt0(n, Jx, Jz, basis, bc):
    """
    Generates the XXZ Hamiltonian

    Args:
        n (int): number of spins
        Jx (float): xx interaction strength
        Jz (float): zz interaction strength
        basis (np.array): array of bits encoding the wave functions
        bc (0 or 1): boundary conditions 0 = open, 1 = close

    Returns:
        sparse matrix: xxz chain
    """
    t = time.time()
    H = spr.dok_matrix((basis.shape[0], basis.shape[0]))
    cc = n - 1 + bc
    for i in range(0, cc):
        for j in range(0, basis.shape[0]):
            H[j, j] = H[j, j] + Jz * (basis[j, i] - 0.5) * (basis[j, np.mod(i + 1, n)] - 0.5)
            l1 = basis[j].copy()
            l2 = basis[j].copy()
            if basis[j, i] == 0 and basis[j, np.mod(i + 1, n)] == 1:
                l1[i] = 1
                l1[np.mod(i + 1, n)] = 0
                k1 = int(np.argwhere((l1 == basis).all(axis=1))[0])
                H[k1, j] = H[k1, j] + Jx * 0.5
            elif basis[j, i] == 1 and basis[j, np.mod(i + 1, n)] == 0:
                l2[i] = 0
                l2[np.mod(i + 1, n)] = 1
                k2 = int(np.argwhere((l2 == basis).all(axis=1))[0])
                H[k2, j] = H[k2, j] + Jx * 0.5
    print("\nmatt0 time was: %s" % ptime(t))
    return H


def matt_add_jz(H0, n, Jz, basis, bc):
    """
    Adds $\hat{S}^z_i\hat{S}^z_{i+1}$ interaction for an existing Hamiltonian

    Args:
        H0 (sparse matrix): existing Hamiltonian
        n (int): number of spins in the chain
        Jz (float): interaction strength
        basis (np.array): array of numbers encoding the wave functions
        bc (0 or 1): boundary conditions 0 = open, 1 = close

    Returns:
        sparse matrix: The matrix $\hat{H}_0$ with addition of $zz$ interactions

    """
    t = time.time()
    H = H0.copy()
    # H = H.tocsr()
    for i in range(0, n - 1 + bc):
        H.setdiag(H.diagonal() + Jz * (basis[:, i] - 0.5) * (basis[:, np.mod(i + 1, n)] - 0.5))
    print("\nmatt_add_jz time was: %s" % ptime(t))
    return H


def matt_add_stark(H0, n, f, a, basis, bc):
    r"""
    Add a linear field of strength ``f`` to an existing Hamiltonian (stark potential).
    $$\hat{H}_0\rightarrow\hat{H}_0 + \sum_{j=1}^{L}\left(\gamma j+\alpha j^{2}/L^{2}\right) \hat{S}_{j}^{z}$$

    Args:
        H0 (sparse matrix): existing Hamiltonian
        n (int): number of spins in the chain
        f (float): linear potential strength
        a (float): curvature strength
        basis (np.array): array of numbers encoding the wave functions
        bc (0 or 1): boundary conditions 0 = open, 1 = close

    Returns:
        sparse matrix: The initial Hamiltonian plus the Stark potential term

    """
    t = time.time()
    H = H0.copy()
    pot_arr = f * np.arange(n) + a * np.arange(n) ** 2 / float(n - 1) ** 2
    # H = H.tocsr()
    if bc == 1:
        cc = n
    else:
        cc = n - 1
    for i in range(0, cc):
        H.setdiag(H.diagonal() + pot_arr[i] * (basis[:, i] - 0.5))
    if bc == 0:
        # H[j, j] = H[j, j] + (f * (n - 1) / 2 - a * ((n - 1) / n) ** 2) * (basis[j, n - 1] - 0.5)
        H.setdiag(H.diagonal() + pot_arr[n - 1] * (basis[:, n - 1] - 0.5))  # * 0.5
    print("\nmatt_add_stark time was: %s" % ptime(t))
    return H


def matt_add_imp(H0, imp, h, basis, n=0):
    r"""
    Add a magentic impurity at a specific site of the chain $h\hat{S}^z_{\textrm{imp}}$

    Args:
        H0 (sparse matrix): existing Hamiltonian
        imp (int): impurity location
        h (float): impurity strength
        basis (np.array): array of wave functions, encoded in bits formation

    Returns:
        sparse matrix: The initial Hamiltonian plus the impurity term

    """
    tz = time.time()
    H = H0.copy()
    H.setdiag(H.diagonal() + h * (basis[:, imp] - 0.5))
    print("\nmatt_add_imp time was: %s" % ptime(tz))
    return H.todok()


def matt0sz(i, basis):
    """
    Generates the operator $\hat{S}^z_i$

    Args:
        i (int): location of the operator
        basis (np.array): array of wave functions, encoded in bits formation

    Returns:
        sparse matrix: $\hat{S}^z_i$

    """
    h = np.zeros(basis.shape[0])
    h = h + (basis[:, i] - 0.5)
    return spr.dia_matrix((h, 0), shape=(basis.shape[0], basis.shape[0]))
