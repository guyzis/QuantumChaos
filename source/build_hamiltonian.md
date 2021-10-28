This module uses to build matrix representations of Hamiltonians.

The main logic behind the code is to represent the wavefunctions at an easy computational basis (and code it as binary numbers):
$$\left|\psi\right\rangle=\left|\uparrow\downarrow\downarrow\right\rangle=\texttt{[1,0,0]}=\sum_{i=2}^02^i\cdot x=\boldsymbol{4}.$$
We start building the Hamiltonian matrix by generating this basis (either in basis of 10 or binary numbers array). If we take for example three spins ($L=3$) and one excitation ($N=1$), we get the following basis:
$$\left\{\left|\downarrow\downarrow\uparrow\right\rangle  , \left|\downarrow\uparrow\downarrow\right\rangle  , \left|\uparrow\downarrow\downarrow\right\rangle\right\},\space \left\{\boldsymbol{1},\boldsymbol{2},\boldsymbol{4}\right\} $$

 Now we can see how a term in the Hamiltonian operates on this wavefunction
$$\hat{S}^-_0\hat{S}^+_1\left|\uparrow\downarrow\downarrow\right\rangle=\left|\downarrow\uparrow\downarrow\right\rangle$$
Then we could match that output with the origin $\boldsymbol{4}\leftrightarrow\boldsymbol{2}$, so the matrix representation of this term in this basis is:


$$\hat{S}_{0}^{-}\hat{S}_{1}^{+}=\begin{array}{c|ccc}
\psi & \left|\downarrow\downarrow\uparrow\right\rangle  & \left|\downarrow\uparrow\downarrow\right\rangle  & \left|\uparrow\downarrow\downarrow\right\rangle \\
\hline \left\langle \downarrow\downarrow\uparrow\right| & 0 & 0 & 0\\
\left\langle \downarrow\uparrow\downarrow\right| & 0 & 0 & 1\\
\left\langle \uparrow\downarrow\downarrow\right| & 0 & 0 & 0
\end{array}=\begin{array}{c|ccc}
\psi & \boldsymbol{1} & \boldsymbol{2} & \boldsymbol{4}\\
\hline \boldsymbol{1} & 0 & 0 & 0\\
\boldsymbol{2} & 0 & 0 & 1\\
\boldsymbol{4} & 0 & 0 & 0
\end{array}=\left(\begin{array}{ccc}
0 & 0 & 0\\
0 & 0 & 1\\
0 & 0 & 0
\end{array}\right)$$

All matrices generated are at the size of the basis.


