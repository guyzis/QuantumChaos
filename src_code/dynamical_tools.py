"""
.. include:: dynamics.md
"""
import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp
from scipy import sparse as spr
from scipy.sparse import linalg as las
import scipy.interpolate as intr
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.linalg import expm, sinm, cosm
from scipy import optimize
from scipy import stats
import os
from sympy.utilities.iterables import multiset_permutations
import h5py
import sympy as sym
import itertools
from sympy.core import sympify
import sys

from utils import *
from build_hamiltonian import *


def entropy(v, n, basis):
    """
    Calculates the entanglement entropy of a vector ``v``

    Args:
        v (np.array): vector
        n (int): number of spins in the chain
        basis (np.array): array of numbers encoding the wave functions

    Returns:
        (float): entanglment entropy

    """
    k = int(n / 2)
    basis1 = bittoint(basis[:, :k])
    basis2 = bittoint(basis[:, k:])
    m = spr.csc_matrix((v, (basis1, basis2)), shape=(2 ** k, 2 ** k))
    s = np.trim_zeros(sp.linalg.svdvals(m.A)) ** 2
    s = s / np.sum(s)
    return - np.sum(s * np.log(s))


def evol_kryl(H, v, dt, T):
    r"""
    Calculates the time evolution of a vector using Krylov sub-space
    $$\vec{v}(t) = e^{-i\hat{H}t}\vec{v}$$

    Args:
        H (sparse matrix): Hamiltonian
        v (np.array): starting vector
        dt (float): time interval
        T (int): number of steps

    Returns:
        (np.array): vector

    """
    for i in range(0, T):
        v = las.expm_multiply(-1j * H * dt, v)  # this loop is necessary for the convergence of expm_multiply
    return v


def msd(H, n, basis, t, k, dt, seed=False):
    r"""
    Calculates the spins mean square displacement after a spin flip in the middle of the chain
    $$G_{n}\left(t\right)=\frac{1}{\mathcal{D}}\textrm{Tr}\left[\hat{S}_{n}^{z}\left(t\right)\hat{S}_{L/2}^{z}\right]$$
    $$x^{2}\left(t\right)=\sum_{n}n^{2}\left(G_{n}\left(t\right)-G_{n}\left(0\right)\right)$$
    Args:
        H (sparse matrix): Hamiltonian
        n (int): number of spins in the chain
        basis (np.array): array of numbers encoding the wave functions
        t (float): ending time of the calculation
        k (int): number of point to sample in time
        dt (float): time intervals for the Krylov time evolution
        seed (bool): fix the seed

    Returns:

    """
    tz1 = time.time()
    tt = np.linspace(0, t, k)
    rng = range(int(- n / 2) + 1, int(n / 2) + 1)
    grprofile = np.zeros([k, len(rng)])
    i = int(n / 2) - 1  # warning!!!
    op = matt0sz(i, basis)
    op2_dic = {}
    if seed:
        np.random.seed(1)
    v = np.random.normal(0, 1, basis.shape[0]) + 1j * np.random.normal(0, 1, basis.shape[0])
    psi = v / la.norm(v)
    psi2 = op.dot(psi)
    mss = np.zeros(k)
    for l in range(0, k):
        ms = 0
        for j in range(0, len(rng)):
            r = rng[j]
            if r not in op2_dic:
                op2 = matt0sz(i + r, basis)
                op2_dic[r] = op2.copy()
            else:
                op2 = op2_dic[r].copy()
            grt = np.real(np.dot(np.conj(psi), op2.dot(psi2.T)))
            ms = ms + r ** 2 * grt
            grprofile[l, j] = grt
        if l < k - 1:
            T = np.floor((tt[l + 1] - tt[l]) / dt).astype(int)  # number of steps in the section
            dt2 = (tt[l + 1] - tt[l]) / T  # fine tuning dt so it will actually fit the number of steps
            psi = evol_kryl(H, psi, dt2, T)
            psi2 = evol_kryl(H, psi2, dt2, T)
        mss[l] = ms
    print("L = %i, k = %i ,one haar measure time: %s" % (n, k, ptime(tz1)))
    return (tt, mss, grprofile)


def entropy_vs_time(H, n, basis, t, k, dt):
    """
    Calculate the entanglement entropy


    Args:
        H (sparse matrix): Hamiltonian
        n (int): number of spins in the chain
        basis (np.array): array of numbers encoding the wave functions
        t (float): ending time of the calculation
        k (int): number of point to sample in time
        dt (float): time intervals for the Krylov time evolution

    Returns:
        (np.array): [time array, entangelment entropy array]

    """
    tz0 = time.time()
    tt = np.linspace(0, t, k)
    ent = np.zeros(k)
    psi = np.zeros(basis.shape[0])
    psi[np.random.randint(0, basis.shape[0])] = 1
    for l in range(0, k):
        ent[l] = entropy(psi, n, basis)
        if l < k - 1:
            T = np.floor((tt[l + 1] - tt[l]) / dt).astype(int)  # number of steps in the section
            dt2 = (tt[l + 1] - tt[l]) / T  # fine tuning dt so it will actually fit the number of steps
            psi = evol_kryl(H, psi, dt2, T)
    print("L = %i, k = %i ,entropy measure time: %s" % (n, k, ptime(tz0)))
    return np.array([tt, ent])
