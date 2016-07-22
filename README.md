# Gaussian Process regression on GPU
## About
This is a rough implementation of Gaussian Process regression for GPUs using CUDA. The original implementation was from Carl Edward Rasmussen in Matlab. Some fine details include understanding CUDA dynamic parallelism and the cuBLAS device API (along with memory considerations), compiling with the right arch flags (35), linking with cublas library. Optimization of hyperparameters is done via conjugate gradient descent and the covariance function used throughout is the exponential period covariance function since this was developed for application on daily temperature data. 

## Requirements
* NVIDIA GPU with compute capability >= 3.5 for CUBLAS Dynamic Parallelism
* CUDA/Toolkit version >= 7.0
* cuBLAS libraries

## Notes
A caveat is that LU decomposition is performed everywhere instead of Cholesky decomposition because a device side API for cuSolver library or any other library does not exist for Cholesky decomposition at this point in time (notice that `<potrf/s/i>` functions have been replaced with `<getrf/s/i>` versions). If your covaraince matrix is not _nice_, the two methods are not equivalent in terms of inverse. LU decomposition is not as numerically stable as Cholesky decomposition and is considerably slower. 

## Contact
tnybny@gmail.com
