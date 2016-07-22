#ifndef GPCHANGE_CUH
#define GPCHANGE_CUH

/* GPChange related function routines */
__device__ void computeMLLChol(double *mll, double *dmll, double *X, double *Y, int nX, int nY, double *lh, int nh, char order);

#endif
