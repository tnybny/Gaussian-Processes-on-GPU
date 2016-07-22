#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "covPNoise.cuh"

#define O CblasColMajor
#define S CblasUpper

#define PI 3.141592

double *_cV;
double **_dcV;
double *_Y;
int _nh;

typedef struct gpthread_st{
	unsigned int thread_id;
	unsigned int start;
	unsigned int end;
	double *mll;
	double *dmll;
	double *ld;
	int nX;
} gpthread_st;

/*
 * Compute Marginal Log Likelihood for inputs X and outputs Y
 * using hyperparameters specified in lh (CHOLESKY METHOD)
 * Also computes the derivatives w.r.t hyperparameters
 * X is a nX x 1 vector
 * Y is a nY x nX matrix where each row is a set of inputs
 * order specifies if Y is column oriented (order = 'C') or row oriented (order = 'R')
 */

__device__ void computeMLLChol(double *mll, double *dmll, double *X, double *Y, int nX, int nY, double *lh, int nh, char order){
	int i, j, k;
	
	double * cM = (double *)malloc(sizeof(double) * nX * nX);
	if(cM == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	covMat(X, nX, cM, lh);
	
	double ** dcM = (double **) malloc(sizeof(double *) * nh);
	if (dcM == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	//printf("Computing cholesky based MLL\n");
	for(i = 0; i < nh; i++) {
		dcM[i] = (double *) malloc(sizeof(double) * nX * nX);
		if(dcM[i] == nullptr)
		{
			printf("could not allcoate memory\n");
			return;
		}
		dcovMat(X, nX, dcM[i], lh, i + 1);
	}

	double negloglik = 0;
	double * dnegloglik = (double *) malloc(nh * sizeof(double));
	if (dnegloglik == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	for(i = 0; i < nh; i++) dnegloglik[i] = 0;

	double *l, ld, *y;
	l = (double *) malloc(sizeof(double) * nX);
	if (l == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	y = (double *) malloc(sizeof(double) * nX);
	if (y == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	int mY = 1;

	// compute the Cholesky factorization of a real symmetric positive definite matrix
	//dpotrf_(&U, &nX, cM, &nX, &info);
	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	
	if (status != CUBLAS_STATUS_SUCCESS)
                  return;
	
	int * piv = (int *) malloc(nX * sizeof(int));
	if (piv == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	int * inf = (int *) malloc(sizeof(int));
	if (inf == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	double ** Aarray = (double **)malloc(sizeof(double *));
	if (Aarray == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	*Aarray = cM;
	
	status = cublasDgetrfBatched(hdl, nX, Aarray, nX, piv, inf, 1);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		return;
	
	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	double * cMi = (double *) malloc(sizeof(double) * nX * nX);
	//cblas_dcopy(nX*nX,cM,1,cMi,1);
	status = cublasCreate_v2(&hdl);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

    status = cublasDcopy(hdl, nX * nX, cM, 1, cMi, 1);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		return;
	
	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	//compute the inverse of a real symmetric positive definite matrix A using the Cholesky factorization A = U**T*U or A = L*L**T computed by DPOTRF
	//dpotri_(&U, &nX, cMi, &nX, &info);
	double * carr = (double *) malloc(sizeof(double) * nX * nX);
	if (carr == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	double ** Carray = (double **)malloc(sizeof(double *));
	if (Carray == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	*Carray = carr;
	
	double ** cAarray = (double **)malloc(sizeof(double *));
	if (cAarray == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	*cAarray = cMi;
	
	status = cublasCreate_v2(&hdl);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDgetriBatched(hdl, nX, cAarray, nX, piv, Carray, nX, inf, 1);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}
	// copy Carray[0] into cMi
	status = cublasCreate_v2(&hdl);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;
	
	status = cublasDcopy(hdl, nX * nX, carr, 1, cMi, 1);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		return;
	
	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	free(carr);
	free(cAarray);
	
	double ** conAarray = (double **)malloc(sizeof(double *)); 
	if (conAarray == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	*conAarray = cM;
	
	double ** larray = (double **)malloc(sizeof(double *));
	if (larray == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	for(i = 0; i < nY; i++){
		//solve cM*l = Y[i] using dpotrs
		if(order == 'C')
		{
			//cblas_dcopy(nX,&(Y[i]),nY,l,1);
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			status = cublasDcopy(hdl, nX, &(Y[i]), nY, l, 1);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}
		}
		else
		{
			//cblas_dcopy(nX,&(Y[i*nX]),1,l,1);
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			status = cublasDcopy(hdl, nX, &(Y[i * nX]), 1, l, 1);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}
		}
		//solve a system of linear equations A*X = B with a symmetric positive definite matrix A using the Cholesky factorization A = U**T*U or A = L*L**T computed by DPOTRF
		//dpotrs_(&U, &nX, &mY, cM, &nX, l, &nX, &info);
		status = cublasCreate_v2(&hdl);

		if (status != CUBLAS_STATUS_SUCCESS)
			return;

		*larray = l;

		status = cublasDgetrsBatched(hdl, CUBLAS_OP_N, nX, mY, conAarray, nX, piv, larray, nX, inf, 1);
		
		if (status != CUBLAS_STATUS_SUCCESS)
			return;

		if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
		{
			printf("cublasDestroy failed\n");
			return;
		}
		
		free(piv);
		free(inf);
		//negloglik += cblas_ddot(nX,l,1,&(Y[i]),nY);
		double * ret = (double *)malloc(sizeof(double));
		if (ret == nullptr)
		{
			printf("could not allocate memory\n");
			return;
		}

		status = cublasCreate_v2(&hdl);

		if (status != CUBLAS_STATUS_SUCCESS)
			return;

		status = cublasDdot(hdl, nX, l, 1, &(Y[i]), nY, ret);
		
		if (status != CUBLAS_STATUS_SUCCESS)
			return;
		
		if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
		{
			printf("cublasDestroy failed\n");
			return;
		}

		negloglik += *ret;

		double * al = (double *)malloc(sizeof(double));
		if (al == nullptr)
		{
			printf("could not allocate memory\n");
			return;
		}

		double * be = (double *)malloc(sizeof(double));
		if (be == nullptr)
		{
			printf("could not allocate memory\n");
			return;
		}
		
		*al = 1,
		*be = 0;
		
		for(j = 0; j < nh; j++){
			//cblas_dsymv(CblasColMajor,CblasUpper,nX,1,dcM[j],nX,l,1,0,y,1);
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			status = cublasDsymv(hdl, CUBLAS_FILL_MODE_UPPER, nX, al, dcM[j], nX, l, 1, be, y, 1);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			//dnegloglik[j] += cblas_ddot(nX,l,1,y,1);
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			status = cublasDdot(hdl, nX, l, 1, y, 1, ret);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

            negloglik += *ret;
		}
		free(ret);
		free(al);
		free(be);
	}

	free(l);
	free(larray);
	free(y);

	//compute ld using cholesky decomposition of cM
	ld = 0;
	for(i = 0; i < nX; i++) ld += log(cM[i + i * nX]);

	ld *= 2;
	negloglik = 0.5 * (negloglik + nY * ld + nY * nX * log(2 * PI));
	*mll = negloglik;

	//compute tr(K^{-1}dk)
	for(i = 0; i < nX; i++){
		for(j = 0; j < nX; j++){
			for(k = 0; k < nh; k++){
				if(j >= i){
					dnegloglik[k] -= dcM[k][i + j * nX] * cMi[i + j * nX];
				}else{
					dnegloglik[k] -= dcM[k][j + i * nX] * cMi[j + i * nX];
				}
			}
		}
	}
	free(cM);
	free(Aarray);
	free(cMi);

	for (i = 0; i < nh; i++) free(dcM[i]);

	free(dcM);

	for(i = 0; i < nh; i++) {
		dmll[i] = -0.5 * dnegloglik[i];
	}  

	free(dnegloglik);
}
