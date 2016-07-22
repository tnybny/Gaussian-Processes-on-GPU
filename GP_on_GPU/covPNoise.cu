#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define PI 3.141592
/* 
 *  Return number of hyperparameters
 */
__device__ int retNumParams(){
	return 4;
}

/*
 * Compute the covariance matrix in column oriented vector format using X
 * The output matrix is upper triangular
 * Output matrix formula k(x,z) = sf^2 * exp( -2*sin^2( pi*||x-z||/p )/ell^2 )	
 */
__device__ void covMat(double *X, int nX, double *cM, double *lh){
	double l = exp(lh[0]);
	double sf2 = exp(2 * lh[1]);
	double p = exp(lh[2]);
	double sn2 = exp(2 * lh[3]);
	int i, j;
	for(i = 0; i < nX; i++){
		for(j = i; j < nX; j++){
			double _int = fabs(X[i] - X[j]);
			cM[i + j * nX] = sf2 * exp(-2 * pow(sin((PI * _int) / p), 2) / pow(l, 2));
			if(j == i) cM[i + j * nX] += sn2;
		}

	}
}


/*
 * Compute the covariance matrix in column oriented vector format using X
 * The output matrix is upper triangular
 * Output matrix formula k(x,z) = sf^2 * exp( -2*sin^2( pi*||x-z||/p )/ell^2 )
 * same as above except doesn't add sn2 to main diagonal	
 * Divides all elements by sn2 (when noise is high) and then adds an identity matrix
 */
__device__ void covMatSp(double *X, int nX, double *cM, double *lh){
	double l = exp(lh[0]);
	double sf2 = exp(2 * lh[1]);
	double p = exp(lh[2]);
	double sn2 = exp(2 * lh[3]);
	for(int i = 0; i < nX; i++)
	{
		for(int j = i; j < nX; j++)
		{
			double _int = fabs(X[i] - X[j]);
			cM[i + j * nX] = sf2 * exp(-2 * pow(sin((PI * _int) / p), 2) / pow(l, 2)) / sn2;
			if (j == i)
			{
				cM[i + j * nX] += 1;
			}
		}
	}
}

/*
 * Compute the derivative of the covariance matrix in column oriented vector format using X with respect to hyperparameter #n
 * The output matrix is upper triangular
 */
__device__ void dcovMat(double *X, int nX, double *dcM, double *lh, int n){
	double l = exp(lh[0]);
	double sf2 = exp(2*lh[1]);
	double p = exp(lh[2]);
	double sn2 = exp(2*lh[3]);
	int i, j;
	double tmp100;
	for(i = 0; i < nX; i++)
	{
		for(j = i; j < nX; j++)
		{
			double _int = fabs(X[i]-X[j]);
			switch(n){
				case 1://log(l)
					dcM[i+j*nX] = (sf2*exp(-2*pow(sin((PI*_int)/p),2)/pow(l,2)))*4*(pow(sin((PI*_int)/p),2)/pow(l,2));
					break;
				case 2://log(sf)
					dcM[i+j*nX] = 2*sf2*exp(-2*pow(sin((PI*_int)/p),2)/pow(l,2));
					break;
				case 3://log(p)
					//	dcM[i+j*nX] = sf2*exp(-2*pow(sin((PI*_int)/p),2)/pow(l,2))*(4*sin((PI*_int)/p))*((PI*_int)/(pow(p,2)*pow(l,2)))*(cos((PI*_int)/p));
					tmp100 = (PI*_int)/p;
					dcM[i+j*nX]= ((sf2*4*tmp100)/(pow(l,2)))*exp((-2*pow(sin(tmp100),2))/pow(l,2))*sin(tmp100)*cos(tmp100);
					break;
				case 4://log(sn)
					if(i == j)
						dcM[i+j*nX] = 2*sn2;
					else
						dcM[i+j*nX] = 0;
					break;
				default:
					return;
			}
		}
	}
}

/* 
 * Compute element wise cross-covariance matrix between a vector of training observations, X and a vector test observations, Y.
 * The output matrix cM is a nX x nY matrix in column oriented format
 */

__device__ void covMatCross(double *X, double *Y, int nX, int nY, double *cM, double *lh){
	double l = exp(lh[0]);
	double sf2 = exp(2 * lh[1]);
	double p = exp(lh[2]);
	int i, j;
	for(j = 0; j < nY; j++)
	{
		for(i = 0; i < nX; i++)
		{
			double _int = fabs(X[i] - Y[j]);
			cM[i + j * nX] = sf2 * exp(-2 * pow(sin((PI * _int) / p), 2) / pow(l, 2));
		}
	}
}