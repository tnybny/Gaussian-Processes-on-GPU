#ifndef COVPNOISE_CUH
#define COVPNOISE_CUH

/* covariance function related routines */
__device__ int retNumParams();
__device__ void covVec(double *X, int nX, double *cV, double *lh);
__device__ void dcovVec(double *X, int nX, double *dcV, double *lh, int n);
__device__ void covMat(double *X, int nX, double *cM, double *lh);
__device__ void covMatSp(double *X, int nX, double *cM, double *lh);
__device__ void dcovMat(double *X, int nX, double *dcM, double *lh, int n);
__device__ void covMatCross(double *X, double *Y, int nX, int nY, double *cM, double *lh);
__device__ void dcovMatCross(double *X, double *Y, int nX, int nY, double *dcM, double *lh, int n);

#endif
