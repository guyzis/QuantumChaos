r"""
Old versions of some utils, that have been replaced with a more efficient src_code.
This module can be used for benchmarking with new utils.

The main idea behind this module is that matrices are generating using kronecker product, this is way less efficient
but a bit easier to grasp then the methods used in build_hamiltonian module.

For example, a term can be built as follow:
$$\hat{S}^z = \mathbb{I}_{2^i}\otimes \hat{S}^z \otimes \mathbb{I}_{2^{L-i-1}},$$
where $L$ is the chain length, and $\mathbb{I}_N$ is the identity matrix of dim $N$.

Some functions are built to do the
"""

import numpy as np
import time
from scipy import sparse as spr
from utils import *

# Defining useful matrices
Sz = np.dot(1 / 2, np.array([[1, 0], [0, -1]], dtype=np.complex_))
Sx = np.dot(1 / 2, np.array([[0, 1], [1, 0]], dtype=np.complex_))
Sy = np.dot(1j / 2, np.array([[0, -1], [1, 0]], dtype=np.complex_))
I2 = np.array([[1, 0], [0, 1]], dtype=np.complex_)
S0 = I2
S1 = 2 * Sx
S2 = 2 * Sy
S3 = 2 * Sz
Sigma = [S0, S1, S2, S3]
Sp = (Sx + 1j * Sy)
Sm = (Sx - 1j * Sy)


def outr(a, b):
    r"""
    Kronecker product, equivalent to np.kron but much slower

    Args:
        a (matrix): $\hat{a}$
        b (matrix): $\hat{b}$

    Returns:
        (matrix): $\hat{a}\otimes \hat{b}$

    """
    c = np.zeros([a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]], dtype=np.complex_)
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[1]):
            for ii in range(0, b.shape[0]):
                for jj in range(0, b.shape[1]):
                    c[i * b.shape[0] + ii][j * b.shape[1] + jj] = a[i][j] * b[ii][jj]
    return c


def outr1d(a, b):
    """
    Equivalent to np.kron but only for diagonal matrices

    Args:
        a (matrix): $\hat{a}$
        b (matrix): $\hat{b}$

    Returns:
        (matrix): $\hat{a}\otimes \hat{b}$

    """
    c = np.zeros([a.shape[0] * b.shape[0]], dtype=np.complex_)
    for i in range(0, a.shape[0]):
        for j in range(0, b.shape[0]):
            c[i * b.shape[0] + j] = a[i] * b[j]
    return c


# equiv to tensordot with subspace
def outrsub(a, b, ord):
    c = np.zeros([ord.shape[0], ord.shape[0]], dtype=np.complex_)
    for i in range(0, ord.shape[0]):
        for j in range(0, ord.shape[0]):
            c[i][j] = a[int(ord[i] / b.shape[0])][int(ord[j] / b.shape[0])] * b[np.mod(ord[i], b.shape[0])][
                np.mod(ord[j], b.shape[0])]
    return c


# equiv to outrsub but much faster
def outrsub2(a, b, ord):
    c = np.zeros([ord.shape[0], ord.shape[0]], dtype=np.complex_)
    orda = np.floor_divide(ord, b.shape[0])
    ordb = np.mod(ord, b.shape[0])
    a = a[:, orda][orda]
    b = b[:, ordb][ordb]
    c = np.multiply(a, b)
    return c


# outrsub just for diagonal matrices
def outrsub21d(a, b, ord):
    # c = np.zeros([ord.shape[0], ord.shape[0]], dtype=np.complex_)
    orda = np.floor_divide(ord, b.shape[0])
    ordb = np.mod(ord, b.shape[0])
    a = a[orda, orda]
    b = b[ordb, ordb]
    return np.diag(np.multiply(a, b))


# generate an 2^n dim matrix representing matrix a in place i in the full hilbert space
def outerr(a, i, n):
    if i > 0 and i + 1 < n:
        c = outr(outr(np.identity((np.power(2, i))), a), np.identity((np.power(2, n - i - 1))))
    if i == 0:
        c = outr(a, np.identity((np.power(2, n - 1))))
    if i + 1 == n:
        c = outr(np.identity(np.power(2, i)), a)
    return c


# outerr but faster
def outerrspr(a, i, n):
    if i > 0 and i + 1 < n:
        c = spr.kron(spr.kron(np.identity((np.power(2, i))), a).toarray(),
                     np.identity((np.power(2, n - i - 1)))).toarray()
    if i == 0:
        c = spr.kron(a, np.identity((np.power(2, n - 1)))).toarray()
    if i + 1 == n:
        c = spr.kron(np.identity(np.power(2, i)), a).toarray()
    return c


# outerr for diagonal matrices
def outerr1d(a, i, n):
    if i > 0 and i + 1 < n:
        c = outr1d(outr1d(np.ones((np.power(2, i))), a), np.ones((np.power(2, n - i - 1))))
    if i == 0:
        c = outr1d(a, np.ones((np.power(2, n - 1))))
    if i + 1 == n:
        c = outr1d(np.ones(np.power(2, i)), a)
    return c


# equiv to outerr but with subspace
def outerrsub(a, b, i, n, ord):
    if i > 0 and i + 1 < n:
        c = outrsub(outr(np.identity(np.power(2, i)), outr(a, b)), np.identity((np.power(2, n - i - 2))), ord)
    if i == 0:
        c = outrsub(outr(a, b), np.identity((np.power(2, n - 2))), ord)
    if i + 1 == n:
        c = outrsub(np.identity(np.power(2, i - 1)), outr(a, b), ord)
    return c


# equiv to outerrsub but faster since using outersub2 and spr.kron
def outerrsub2(a, b, i, n, ord):
    if i > 0 and i + 1 < n:
        c = outrsub2(spr.kron(np.identity(np.power(2, i)), spr.kron(a, b).toarray()).toarray(),
                     np.identity((np.power(2, n - i - 2))), ord)
    if i == 0:
        c = outrsub2(spr.kron(a, b).toarray(), np.identity((np.power(2, n - 2))), ord)
    if i + 1 == n:
        c = outrsub2(np.identity(np.power(2, i - 1)), spr.kron(a, b).toarray(), ord)
    if i == -1:
        c = outrsub2(spr.kron(a, np.identity(2 ** (n - 2))).toarray(), b, ord)
    return c


# outerrsub2 for diagonal matrices
def outerrsub21d(a, b, i, n, ord):
    if i > 0 and i + 1 < n:
        c = outrsub21d(spr.kron(np.identity(np.power(2, i)), spr.kron(a, b).toarray()).toarray(),
                       np.identity((np.power(2, n - i - 2))), ord)
    if i == 0:
        c = outrsub21d(spr.kron(a, b).toarray(), np.identity((np.power(2, n - 2))), ord)
    if i + 1 == n:
        c = outrsub21d(np.identity(np.power(2, i - 1)), spr.kron(a, b).toarray(), ord)
    return c


# the same as outerr, not in use
def outrr(a, i, n):
    c = I2
    for j in range(0, n - 1):
        if i != 0:
            if j != i - 1:
                c = outr(c, I2)
            else:
                c = outr(c, a)
        else:
            if j == 0:
                c = outr(a, c)
            else:
                c = outr(c, I2)
    return c


# generate xxz hamiltonian
def xxz(n, Jx, Jz):
    H = np.zeros([np.power(2, n), np.power(2, n)], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * np.dot(outerr(Sx, i, n), outerr(Sx, i + 1, n)) + Jx * np.dot(outerr(Sy, i, n),
                                                                                  outerr(Sy, i + 1, n)) + Jz * np.dot(
            outerr(Sz, i, n), outerr(Sz, i + 1, n))
    return H


# generate xxz hamiltonian faster
def xxz2(n, Jx, Jz):
    t = time.time()
    H = np.zeros([np.power(2, n), np.power(2, n)], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * np.dot(outerrspr(Sx, i, n), outerrspr(Sx, i + 1, n)) + Jx * np.dot(outerrspr(Sy, i, n),
                                                                                        outerrspr(Sy, i + 1,
                                                                                                  n)) + Jz * np.dot(
            outerrspr(Sz, i, n), outerrspr(Sz, i + 1, n))
    print("xxz time was: ", time.time() - t)
    return H


# generate xxz hamiltonian with uniform random magentic field on Z axis
def xxzrand(n, Jx, Jz, h):
    H = np.zeros([np.power(2, n), np.power(2, n)], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * np.dot(outerrspr(Sx, i, n), outerrspr(Sx, i + 1, n)) + Jx * np.dot(outerrspr(Sy, i, n),
                                                                                        outerrspr(Sy, i + 1,
                                                                                                  n)) + Jz * np.dot(
            outerrspr(Sz, i, n), outerrspr(Sz, i + 1, n)) + h * np.random.uniform(-1, 1, 1) * outerrspr(Sz, i, n)
    H = H + h * np.random.uniform(-1, 1, 1) * outerrspr(Sz, n - 1, n)
    return H


# add uniform magnetic field on Z axis
def xxzaddrand(H, n, Jx, Jz, h):
    for i in range(0, n):
        H = H + h * np.random.uniform(-1, 1, 1) * outerrspr(Sz, i, n)
    return H


# generate xxz hamiltonian with closed b.c
def xxzc(n, Jx, Jz):
    H = np.zeros([np.power(2, n), np.power(2, n)], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * np.dot(outerr(Sx, i, n), outerr(Sx, i + 1, n)) + Jx * np.dot(outerr(Sy, i, n),
                                                                                  outerr(Sy, i + 1, n)) + Jz * np.dot(
            outerr(Sz, i, n), outerr(Sz, i + 1, n))
    H = H + Jx * np.dot(outerr(Sx, n - 1, n), outerr(Sx, 0, n)) + Jx * np.dot(outerr(Sy, n - 1, n),
                                                                              outerr(Sy, 0, n)) + Jz * np.dot(
        outerr(Sz, n - 1, n), outerr(Sz, 0, n))
    return H


# same as xxc but uses outerrspr which is faster than outerr
def xxzc2(n, Jx, Jz):
    H = np.zeros([np.power(2, n), np.power(2, n)], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * np.dot(outerrspr(Sx, i, n), outerrspr(Sx, i + 1, n)) + Jx * np.dot(outerrspr(Sy, i, n),
                                                                                        outerrspr(Sy, i + 1,
                                                                                                  n)) + Jz * np.dot(
            outerrspr(Sz, i, n), outerrspr(Sz, i + 1, n))
    H = H + Jx * np.dot(outerrspr(Sx, n - 1, n), outerrspr(Sx, 0, n)) + \
        Jx * np.dot(outerrspr(Sy, n - 1, n), outerrspr(Sy, 0, n)) + \
        Jz * np.dot(outerrspr(Sz, n - 1, n), outerrspr(Sz, 0, n))
    return H


# return a subspace of H by the order a
def subspace(H, a):
    h = np.zeros([a.shape[0], a.shape[0]], dtype=np.complex_)
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[0]):
            h[i, j] = H[a[i], a[j]]
    return h


# generate xxz block0 hamiltonian
def xxzblock0(n, Jx, Jz, ord):
    if n % 2 != 0:
        print("There's no zero magentization block if the number of spins is odd")
    H = np.zeros([ord.shape[0], ord.shape[0]], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * outerrsub(Sx, Sx, i, n, ord) + Jx * outerrsub(Sy, Sy, i, n, ord) + Jz * outerrsub(Sz, Sz, i, n,
                                                                                                       ord)
    return H


# generate xxz block0 hamiltonian with f linear potential and a curvature
def xxzblock0stark(n, Jx, Jz, f, a, ord, c):
    t = time.time()
    if ord.ndim == 2:
        ord = bittoint(ord)
    H = np.zeros([ord.shape[0], ord.shape[0]], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * outerrsub2(Sx, Sx, i, n, ord) + Jx * outerrsub2(Sy, Sy, i, n, ord) + Jz * outerrsub21d(Sz, Sz, i,
                                                                                                            n, ord) + \
            (f * i + a * (i / (n - 1)) ** 2) * outerrsub21d(Sz, I2, i, n, ord)
    H = H + (f * (n - 1) + a * ((n - 1) / (n - 1)) ** 2) * outerrsub21d(I2, Sz, n - 1, n, ord)
    if c == 1:
        H = H + Jx * (outerrsub2(Sx, Sx, - 1, n, ord) + outerrsub2(Sy, Sy, - 1, n, ord)) + \
            Jz * outerrsub2(Sz, Sz, - 1, n, ord)
    print("normal stark time was: ", time.time() - t)
    return H


def xxzblock0addstark(H, n, f, a, ord):
    t = time.time()
    if ord.ndim == 2:
        ord = bittoint(ord)
    for i in range(0, n - 1):
        H += (f * i + a * (i / (n - 1)) ** 2) * outerrsub21d(Sz, I2, i, n, ord)
    H += (f * (n - 1) + a * ((n - 1) / (n - 1)) ** 2) * outerrsub21d(I2, Sz, n - 1, n, ord)
    print("normal addstark time was: ", time.time() - t)
    return H


# generate xxz block0 hamiltonian with impurity at imp
def xxzblock0imp(n, Jx, Jz, imp, h, ord, c):
    t = time.time()
    if ord.ndim == 2:
        ord = bittoint(ord)
    H = np.zeros([ord.shape[0], ord.shape[0]], dtype=np.complex_)
    for i in range(0, n - 1):
        H = H + Jx * outerrsub2(Sx, Sx, i, n, ord) + Jx * outerrsub2(Sy, Sy, i, n, ord) + Jz * outerrsub21d(Sz, Sz, i,
                                                                                                            n, ord) + \
            h * int(imp == i) * outerrsub21d(Sz, I2, i, n, ord)
    H = H + h * int(imp == n - 1) * outerrsub21d(I2, Sz, n - 1, n, ord)
    if c == 1:
        # H = H + subspace(Jx * np.dot(outerrspr(Sx, n - 1, n), outerrspr(Sx, 0, n)) + \
        #     Jx * np.dot(outerrspr(Sy, n - 1, n), outerrspr(Sy, 0, n)) + \
        #     Jz * np.dot(outerrspr(Sz, n - 1, n), outerrspr(Sz, 0, n)), ord)
        H = H + Jx * (outerrsub2(Sx, Sx, - 1, n, ord) + outerrsub2(Sy, Sy, - 1, n, ord)) + \
            Jz * outerrsub2(Sz, Sz, - 1, n, ord)
    print("normal xxzimp0 time was: ", time.time() - t)
    return H


# generate xxz block0 hamiltonian with impurity at imp
def xxzblock0addimp(H, n, imp, h, ord):
    t = time.time()
    if ord.ndim == 2:
        ord = bittoint(ord)
    if imp != n - 1:
        H += h * outerrsub21d(Sz, I2, imp, n, ord)
    else:
        H += h * outerrsub21d(I2, Sz, n - 1, n, ord)
    print("normal xxzaddimp0 time was: ", time.time() - t)
    return H
