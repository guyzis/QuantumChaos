This repo can be used for two purposes:

1. Generate Magnus expanssions expressions (or builders) of an arbitrary time dependent Hamiltonian
2. Generate Magnus expension Hamiltonian (i.e matrices) from that builders.

The code was originally written to solve the following Hamiltonian $$\hat{H}\left(t\right)=\hat{H}_{0}+e^{-i\eta t}\hat{V}+e^{i\eta t}\hat{V}^{\dagger},$$ hence will **definitely need modifications** to generate expressions for other time-dependent Hamiltonian.

``magnus_generator_classic`` was written to generate more arbitrary time-dependent Hamiltonians, while ``magnus_generator`` and ``magnus_generator_parts`` were written specifically for the Hamiltonian above.

To generate the effective Hamiltonian (i.e matrix) one could use the ``build_heff`` module. ``stark_utils`` module is used for generating the Stark Hamiltonian matrix, and is used for testing purposes.

Docs can be found in this ``_build/html/index.html``