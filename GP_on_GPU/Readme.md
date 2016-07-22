Relevant files:

- gpMain: main function for reading data, calling functions for training, prediction, writing results

- header.h: main header file linking all functions

- minimize.c: Conjugate Gradient Descent

- minimizep.c: parallel(multi-threaded) version - UNDER CONSTRUCTION

- covPNoise.c: function for calculating covariance matrix and also derivatives wrt each hyperparameter

- gpchange.c: functions for max. likelihood

- utils.c: generic math functions relevant to gp

To properly run code, 

- Make sure you have the following installed on your machine: BLAS, LAPACK (if you are using a mac, they should technically already be a part of your machine, so find their location using locate command)

- Point directories to proper files in Makefile

- Build code using the following commands: make clean make install make

- To run, ./a.out

Input: ~10k time series - 1 for each gridbox (read from data folder)

