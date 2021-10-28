"""
Test module, to use it run
``python -m unittest -v code.unit_test.Test``
"""
import os
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
sys.path.append(dname)
os.chdir(dname)

from build_hamiltonian import *
from legacy_utils import *
from numpy import linalg as la

class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('----------------------\n Test setUp\n----------------------')

        cls.n = np.random.choice([8, 10, 12, 14])
        cls.jx = np.random.uniform(0.5, 3)
        cls.jz = np.random.uniform(0.5, 3)
        cls.ord = block0(cls.n)
        cls.bc = np.random.choice([0, 1])
        cls.f = np.random.uniform(0.5, 3)
        cls.a = np.random.uniform(0.1, 1)
        cls.imp = np.random.randint(0, cls.n)
        cls.h_imp = np.random.uniform(0.1, 2)
        print('Initial variables')
        print('n = {n}, jx = {jx:.2f}, jz = {jz:.2f}, bc = {bc}'.format(n=cls.n, jx=cls.jx, jz=cls.jz, bc=cls.bc))
        print('f = {f:.2f}, a = {a:.2f}'.format(f=cls.f, a=cls.a))
        print('imp = {imp}, h_imp = {h_imp:.2f}'.format(imp=cls.imp, h_imp=cls.h_imp))



        cls.h1 = matt_stark(cls.n, cls.jx, cls.jz, f=0,  a=0, ord=cls.ord, bc=cls.bc, shift=False)
        cls.h2 = xxzblock0stark(cls.n, cls.jx, cls.jz, f=0, a=0, ord=cls.ord, c=cls.bc)

        print('----------------------\n done setUp\n----------------------')

    def test_xxz(self):
        xxz_bool = np.allclose(self.h1.A, self.h2)
        print('test_xxz = ?', xxz_bool)
        self.assertTrue(xxz_bool, msg='matt_stark and xxzblock0stark do not generate the same matrix')

    def test_stark(self):
        h1_stark = mattaddstark(self.h1.copy(), self.n, self.f, self.a, self.ord, self.bc)
        h2_stark = xxzblock0addstark(self.h2.copy(), self.n, self.f, self.a, self.ord, self.bc)
        stark_bool = np.allclose(h1_stark.A, h2_stark)
        print('test_stark = ?', stark_bool)
        self.assertTrue(stark_bool, msg='mattaddstark and xxzblock0addstark do not generate the same matrix')

    def test_impurity(self):
        h1_imp = mattaddimp3(self.h1.copy(), self.imp, self.h_imp, self.ord, n=self.n)
        h2_imp = xxzblock0addimp(self.h2.copy(), self.n, self.imp, self.h_imp, self.ord)
        imp_bool = np.allclose(h1_imp.A, h2_imp)
        print('test_stark = ?', imp_bool)
        self.assertTrue(imp_bool, msg='mattaddimp and xxzblock0addimp do not generate the same matrix')

    @classmethod
    def tearDownClass(cls):
        print('----------------------\n Test tearDown\n----------------------')
        print('----------------------\n done tearDown\n----------------------')