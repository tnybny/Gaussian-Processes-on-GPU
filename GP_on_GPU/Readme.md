Relevant files:

- gpMain.cu: main function for reading data, calling functions for training, prediction, writing results

- gp.cu: meat of Gaussian Process regression

- minimize.cu: Conjugate Gradient Descent

- covPNoise.cu: function for calculating covariance matrix and also derivatives wrt each hyperparameter

- gpchange.cu: functions for max. likelihood

- utils.cu: generic math functions relevant to gp

- io.cpp: reading input and writing intermediate results

- all of the above's associated header files
