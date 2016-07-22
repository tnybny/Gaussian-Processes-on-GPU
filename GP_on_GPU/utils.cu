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
/*void ztest(double *P, double *X, int nX, double *mean, int nMean, double *sigma, int nSigma, int tail){
  int i,j = 0,k = 0;
  for(i = 0; i < nX; i++){
    if(nMean > 1) j = i;
    if(nSigma > 1) k = i;
    switch(tail){
    case 0:
      P[i] = 2*gsl_cdf_ugaussian_Q(fabs((X[i] - mean[j])/sigma[k]));
      break;
    case 1:
      P[i] = gsl_cdf_ugaussian_Q(fabs((X[i] - mean[j])/sigma[k]));
      break;
    case -1:
      P[i] = gsl_cdf_ugaussian_P(fabs((X[i] - mean[j])/sigma[k]));
      break;
    default:
      fprintf(stderr,"tail must be 0 (both) 1 (right) or -1 (left)\n");
      return;
    }
  }
}*/
