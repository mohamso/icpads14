Effective Multi-GPU Communication Using Multiple CUDA Streams and Threads
===================================================================================
An online repository containing the source code for the paper called “An Effective Multi-GPU Communication Scheme Using Multiple CUDA Streams and GPUDirect” submitted to the SC '14 conference. 

The repository is divided as follows: 2 and 4. The former is a directory containing the source code for the two GPU implementations and the latter contains the source code for the four GPU implementations.

The source code can be compiled by running “make” from the src directory found inside each main directory. Code comments with thorough description of the code can be found at the top of each file.

Repeating the Experiments
=========================
All codes accept the following parameters:

./<app> global_domain_x global_domain_y global_domain_z iterations CUDA_block_size_x CUDA_block_size_y CUDA_block_size_z

We recommend the following block sizes for Fermi:

CUDA_block_size_x: 32
CUDA_block_size_y: 4
CUDA_block_size_z: 1 

We recommend the following block sizes for Kepler:

CUDA_block_size_x: 64
CUDA_block_size_y: 4
CUDA_block_size_z: 1 

In order to repeat the synchronous baseline experiment with a global problem size of 256^3 from the paper on Fermi, we recommend starting the application in following way:

../bin/sync 256 256 256 100 32 4 1

and the following parameters are recommended for Kepler:

../bin/async 256 256 256 100 64 4 1
