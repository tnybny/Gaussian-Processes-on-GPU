#ifndef GP_CUH
#define GP_CUH

/* main GP realted routine*/
__device__ void gp(double *lh, double *X, int nX, double *Y, double Xtr, double *mean, double *variance, int idx);

#endif