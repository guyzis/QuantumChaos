This repo can be used for two purposes:

1. Generate Magnus expanssions expressions (or builders) of an arbitrary time dependent Hamiltonian
2. Generate Magnus expension Hamiltonian (i.e matrices) from that builders.

The code was originally written to solve the following Hamiltonian <img src="http://www.sciweavers.org/tex2img.php?eq=%5Chat%7BH%7D%5Cleft%28t%5Cright%29%3D%5Chat%7BH%7D_%7B0%7D%2Be%5E%7B-i%5Ceta%20t%7D%5Chat%7BV%7D%2Be%5E%7Bi%5Ceta%20t%7D%5Chat%7BV%7D%5E%7B%5Cdagger%7D%2C&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\hat{H}\left(t\right)=\hat{H}_{0}+e^{-i\eta t}\hat{V}+e^{i\eta t}\hat{V}^{\dagger}," width="229" height="22" />$$\hat{H}\left(t\right)=\hat{H}_{0}+e^{-i\eta t}\hat{V}+e^{i\eta t}\hat{V}^{\dagger},$$ hence will **definitely need modifications** to generate expressions for other time-dependent Hamiltonian.

``magnus_generator_classic`` was written to generate more arbitrary time-dependent Hamiltonians, while ``magnus_generator`` and ``magnus_generator_parts`` were written specifically for the Hamiltonian above.

To generate the effective Hamiltonian (i.e matrix) one could use the ``build_heff`` module. ``stark_utils`` module is used for generating the Stark Hamiltonian matrix, and is used for testing purposes.

Docs can be found in this ``_build/html/index.html``