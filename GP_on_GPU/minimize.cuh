#ifndef MINIMIZE_CUH
#define MINIMIZE_CUH

__device__ void minimize(double *X, int nX, double *Y, int nY, int length, int red, double *lh0, int nh, int numthreads, int method, int n);

#endif
