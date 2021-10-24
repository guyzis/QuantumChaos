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


def entropy(v, n, ord):
    """

    Args:
        v:
        n:
        ord:

    Returns:

    """
    l = ordtobit(ord, n)
    k = int(n / 2)
    l1 = bittoint(l[:, :k])
    l2 = bittoint(l[:, k:])
    m = spr.csc_matrix((v, (l1, l2)), shape=(2 ** k, 2 ** k))
    s = np.trim_zeros(sp.linalg.svdvals(m.A)) ** 2
    s = s / np.sum(s)
    return - np.sum(s * np.log(s))


def evolkryl3(H, v, dt, T):
    """

    Args:
        H:
        v:
        dt:
        T:

    Returns:

    """
    for i in range(0, T):
        v = las.expm_multiply(-1j * H * dt, v)
    return v


def msd99(H, n, ord, t, k, dt, seed=False, neel=False):
    """

    Args:
        H:
        n:
        ord:
        t:
        k:
        dt:
        seed:
        neel:

    Returns:

    """
    tz1 = time.time()
    tt = np.linspace(0, t, k)
    rng = range(int(- n / 2) + 1, int(n / 2) + 1)
    grprofile = np.zeros([k, len(rng)])
    i = int(n / 2) - 1  # warning!!!
    op = matt3sz(n, i, ord)
    if seed:
        np.random.seed(1)
    v = np.random.normal(0, 1, ord.shape[0]) + 1j * np.random.normal(0, 1, ord.shape[0])
    if neel:
        v = np.zeros(ord.shape[0])
        nl = np.mod(np.arange(n) + 1, 2)
        v[int(np.argwhere(bittoint(nl) == np.flip(block0(n))))] = 1
    psi = v / la.norm(v)
    psi2 = op.dot(psi)
    mss = np.zeros(k)
    for l in range(0, k):
        ms = 0
        for j in range(0, len(rng)):
            r = rng[j]
            op2 = matt3sz(n, i + r, ord)
            grt = np.real(np.dot(np.conj(psi), op2.dot(psi2.T)))
            ms = ms + r ** 2 * grt
            grprofile[l, j] = grt
        if l < k - 1:
            T = np.floor((tt[l + 1] - tt[l]) / dt).astype(int)  # number of steps in the section
            dt2 = (tt[l + 1] - tt[l]) / T  # fine tuning dt so it will actually fit the number of steps
            psi = evolkryl3(H, psi, dt2, T)
            psi2 = evolkryl3(H, psi2, dt2, T)
        mss[l] = ms
    print("L = %i, k = %i ,one haar measure time: %s" % (n, k, ptime(tz1)))
    return (tt, mss, grprofile)


def msd99ent(H, n, ord, t, k, dt):
    """

    Args:
        H:
        n:
        ord:
        t:
        k:
        dt:

    Returns:

    """
    tz0 = time.time()
    tt = np.linspace(0, t, k)
    ent = np.zeros(k)
    psi = np.zeros(ord.shape[0])
    psi[np.random.randint(0, ord.shape[0])] = 1
    for l in range(0, k):
        ent[l] = entropy(psi, n, ord)
        if l < k - 1:
            T = np.floor((tt[l + 1] - tt[l]) / dt).astype(int)  # number of steps in the section
            dt2 = (tt[l + 1] - tt[l]) / T  # fine tuning dt so it will actually fit the number of steps
            psi = evolkryl3(H, psi, dt2, T)
    print("L = %i, k = %i ,entropy measure time: %s" % (n, k, ptime(tz0)))
    return np.array([tt, ent])