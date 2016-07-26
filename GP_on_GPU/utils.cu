#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


__device__ double* myrealloc(int oldsize, int newsize, double* old)
{
    double * newT = (double *) malloc (newsize * sizeof(double));

    for(int i = 0; i < oldsize; i++)
        newT[i] = old[i];

    free(old);
    return newT;
}

__device__ double minu(double a,double b){if(a < b) return a;else return b;}
__device__ double maxu(double a,double b){if(a > b) return a;else return b;}
