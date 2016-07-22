#ifndef UTILS_CUH
#define UTILS_CUH

__device__ double* myrealloc(int oldsize, int newsize, double* old);
__device__ double minu(double a, double b);
__device__ double maxu(double a, double b);

#endif
