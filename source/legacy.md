Old versions of some utils, that have been replaced with a more efficient ``src_code``.
This module is used for benchmarking, and educational purposes.

The main idea behind this module is that matrices are generating using Kronecker product  (denoted by $\otimes$), this is much less efficient but a bit easier to grasp than the methods used in ``build_hamiltonian`` module.

For example, a term can be built as follow:
$$\hat{S}^z_i = \mathbb{I}_{2^i}\otimes \hat{S}^z \otimes \mathbb{I}_{2^{L-i-1}},$$
where $L$ is the chain length, and $\mathbb{I}_N$ is the identity matrix of dim $N$.

Using this method the output is the matrix representation on the 'computational basis',  for example for two spins ($L=2$):
$$\{\left|\uparrow\uparrow\right\rangle, \left|\uparrow\downarrow\right\rangle, \left|\downarrow\uparrow\right\rangle, \left|\downarrow\downarrow\right\rangle\}.$$

If we want to use only a sub-sector of the $N=2^L$ matrix we need to code this basis into ordered numbers, to do so we use the following transformation
$$\left|\downarrow\uparrow\right\rangle=\texttt{[0,1]}=\sum_{i=L}^1(1-x_i)\cdot 2^{L-i}=0\cdot2^0+1\cdot2^1=\boldsymbol{2}.$$

For example if we want to take only the zero magnetization sector (where the same number of spins point up and down) out of the following matrix
$$\hat{S}_{1}^{z}=\mathbb{I}_{2}\otimes\hat{S}^{z}=\left(\begin{array}{cccc}
0.5 & 0 & 0 & 0\\
0 & -0.5 & 0 & 0\\
0 & 0 & 0.5 & 0\\
0 & 0 & 0 & -0.5
\end{array}\right)$$

We need to cut the following entries
$$\{\left|\uparrow\downarrow\right\rangle, \left|\downarrow\uparrow\right\rangle\}=\{\boldsymbol{1}, \boldsymbol{2}\},$$

so the matrix representation in this basis would be

$$\hat{S}_{1,\mathrm{basis}}^{z}=\left(\begin{array}{cc}
-0.5 & 0\\
0 & 0.5
\end{array}\right)$$

