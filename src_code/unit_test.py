"""
Test module, to use it run
``python -m unittest -v src_code.unit_test.Test``
"""
import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)
os.chdir(dname)

from build_hamiltonian import *
from legacy_utils import *
from dynamical_tools import *
from static_tools import *
from numpy import linalg as la


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialize variables, creates XXZ matrices for benchmarking, diagonalize the matrices.
        """
        print('----------------------\n Test setUp\n----------------------')

        cls.n = np.random.randint(10, 14)
        cls.jx = np.random.uniform(0.5, 3)
        cls.jz = np.random.uniform(0.5, 3)
        cls.basis = blockex(cls.n, cls.n // 2)
        cls.bc = np.random.choice([0, 1])
        cls.f = np.random.uniform(0.5, 3)
        cls.a = np.random.uniform(0.1, 1)
        cls.imp = np.random.randint(0, cls.n)
        cls.h_imp = np.random.uniform(0.1, 2)
        print('Initial variables')
        print(f'n = {cls.n}, jx = {cls.jx:.2f}, jz = {cls.jz:.2f}, bc = {cls.bc}')
        print(f'f = {cls.f:.2f}, a = {cls.a:.2f}')
        print(f'imp = {cls.imp}, h_imp = {cls.h_imp:.2f}')

        cls.h1 = matt0(cls.n, cls.jx, cls.jz, basis=cls.basis, bc=cls.bc)
        cls.h2 = xxz_block_stark(cls.n, cls.jx, cls.jz, f=0, a=0, basis=cls.basis, bc=cls.bc)
        cls.h1_imp = matt_add_imp(cls.h1.copy(), cls.imp, cls.h_imp, cls.basis, n=cls.n)
        cls.h2_imp = xxz_block_add_imp(cls.h2.copy(), cls.n, cls.imp, cls.h_imp, cls.basis)

        cls.E, cls.V = la.eigh(cls.h1_imp.A)

        print('----------------------\n done setUp\n----------------------')

    def test_xxz(self):
        """
        Benchmark the basic XXZ matrices
        """
        np.testing.assert_allclose(self.h1.A, self.h2,
                                   err_msg='matt0 and xxzblock0stark do not generate the same matrix')

    def test_stark(self):
        """
        Benchmark the `Stark` potentials
        """
        h1_stark = matt_add_stark(self.h1.copy(), self.n, self.f, self.a, self.basis, self.bc)
        h2_stark = xxz_block_add_stark(self.h2.copy(), self.n, self.f, self.a, self.basis)
        np.testing.assert_allclose(h1_stark.A, h2_stark,
                                   err_msg='matt_add_stark and xxzblock0addstark do not generate the same matrix')

    def test_impurity(self):
        """
        Benchmark the addition of impurity
        """
        np.testing.assert_allclose(self.h1_imp.A, self.h2_imp,
                                   err_msg='matt_add_imp and xxzblock0addimp do not generate the same matrix')

    def test_matt0sz(self):
        i = np.random.randint(self.n)
        sz1 = matt0sz(i, self.basis).A
        sz2 = np.real(outerrsub21d(Sz, I2, i, self.n, bittoint(self.basis)))
        np.testing.assert_allclose(sz1.diagonal(), sz2.diagonal())

    def test_msd(self):
        """
        Make sure msd functions runs properly and return a tuple
        """
        x = msd(self.h1, self.n, self.basis, t=5, k=10, dt=0.1)
        self.assertIsInstance(x, tuple, msg='msd does not return a tuple')

    def test_rfold(self):
        """
        Make sure rfold is runs properly and returns a float
        """
        r = rfold(self.E)
        self.assertIsInstance(r, np.float_, msg='r is not a float')

    def test_diag_elements(self):
        """
        Make sure diag_elements runs properly and returns a tuple
        """
        x = diag_elements([self.E, self.V], self.n, self.basis)
        self.assertIsInstance(x, tuple, msg='diag_elements does not return a tuple')

    def test_offdiag(self):
        """
        Make sure offdiag runs properly and returns a tuple
        """
        x = offdiag([self.E, self.V], self.n, self.basis)
        self.assertIsInstance(x, tuple, msg='offdiag does not return a tuple')

    def test_offdiag_dist(self):
        """
        Make sure offdiag_dist runs properly and returns an np.array
        """
        x = offdiag_dist([self.E, self.V], self.n, self.basis)
        self.assertIsInstance(x, np.ndarray, msg='offdiag does not return an np.array')

    @classmethod
    def tearDownClass(cls):
        print('----------------------\n Test tearDown\n----------------------')
        print('----------------------\n done tearDown\n----------------------')
