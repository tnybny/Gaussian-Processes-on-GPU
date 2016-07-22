#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "covPNoise.cuh"

__device__ void gp(double *lh, double *X, int nX, double *Y, double Xtr, double *mean, double *variance, int idx){
	int i, j;
	int ncols = 1;
	//Formula: Predicted mean = transpose(KTT1) * (cMi) * y
	double * KT1T = (double *)malloc(nX * sizeof(double));
	if (KT1T == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	
	double * cM = (double *)malloc(nX * nX * sizeof(double));
	if (cM == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	// calculate full nX by nX covariance matrix and covariance matrix of new X vs. all others
	// in cM and KT1T respectively
	covMatCross(X, &Xtr, nX, 1, KT1T, lh);
	covMatSp(X, nX, cM, lh);
	// L = chol(cM) 
	// alpha = solve_chol(L, y-m)/sl;
	// L = chol(cM) - we do this using dpotrf 
	//Step 1: Cholesky Factorization - L value is stored in cM itself 
	//dpotrf_(&U, &nX, cM, &nX, &info); 

	// allocations for Dgetrf
	int * piv = (int *)malloc(nX * sizeof(int));
	if (piv == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	int * inf = (int *)malloc(sizeof(int));
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

	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDgetrfBatched(hdl, nX, Aarray, nX, piv, inf, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	double sl = exp(2 * lh[3]);
	double * alpha = (double *)malloc(sizeof(double) * nX);
	if (alpha == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	status = cublasCreate_v2(&hdl);
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	//use dpotrs to do solve_chol(cM,y)
	//cblas_dcopy(nX, Y, 1, alpha, 1);
	status = cublasDcopy(hdl, nX, Y, 1, alpha, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	//dpotrs_(&U, &nX, &ncols, cM, &nX, alpha, &nX, &info);
	double ** conAarray = (double **)malloc(sizeof(double *));
	if (conAarray == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	*conAarray = cM;
	const double **aconst = (const double **)conAarray;
	double ** Barray = (double **)malloc(sizeof(double *));
	if (Barray == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	*Barray = alpha;

	status = cublasCreate_v2(&hdl);
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDgetrsBatched(hdl, CUBLAS_OP_N, nX, ncols, aconst, nX, piv, Barray, nX, inf, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;
	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	for (i = 0; i < nX; i++){
		alpha[i] = alpha[i] / sl;
	}

	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, 1, nX, 1, KT1T, 1, alpha, nX, 0, mean, 1);
	double * ret = (double *)malloc(sizeof(double));
	if (ret == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	// dynamically allocate scalar parameters too for cuBLAS
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

	*al = 1;
	*be = 0;

	status = cublasCreate_v2(&hdl);
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, nX, al, KT1T, 1, alpha, nX, be, ret, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	free(piv);
	free(inf);
	free(alpha);
	free(Barray);
	free(KT1T);
	mean[idx] = *ret;
	free(ret);

	//printf("INSIDE GP: PREDICTED MEAN = %.4lf\n", *mean); 
	////////// PREDICTED MEAN COMPUTATION COMPLETE

	//Begin predicted variance computation
	double * sW = (double *)malloc(nX * sizeof(double));
	if (sW == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	for (i = 0; i < nX; i++){
		sW[i] = (double)(1 / sqrt(sl));
	}

	double * KT1T1 = (double *)malloc(sizeof(double));
	if (KT1T1 == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	// calculate KT1T1 as self-covariance of new X
	covMatCross(&Xtr, &Xtr, 1, 1, KT1T1, lh);

	// Transpose L (stored in cM) to get LT
	double * LT = (double *)malloc(nX * nX * sizeof(double));
	if (LT == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	for (i = 0; i < nX; i++){
		for (j = 0; j < nX; j++){
			LT[j + i * nX] = cM[i + j * nX];
		}
	}

	free(cM);
	free(Aarray);
	//(repmat(sW,1,length(nX)).*KT1T)
	for (i = 0; i < nX; i++){
		KT1T[i] = sW[i] * KT1T[i];
	}
	free(sW);

	//Just try to solve using L'-1 *%* repmatSWKs;
	//dgetrf_(&nX, &nX, LT, &nX, IPIV, &info);
	double ** Aarray2 = (double **)malloc(sizeof(double *));
	if (Aarray2 == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	*Aarray2 = LT;
	int * piv2 = (int *)malloc(nX * sizeof(int));
	if (piv2 == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	int * inf2 = (int *)malloc(sizeof(int));
	if (inf2 == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}

	status = cublasCreate_v2(&hdl);
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDgetrfBatched(hdl, nX, Aarray2, nX, piv2, inf2, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	//dgetri_(&nX, LT, &nX, IPIV, WORK, &LWORK, &info);
	double * carr = (double *)malloc(sizeof(double) * nX * nX);
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
	const double **aconst2 = (const double **)Aarray2;

	status = cublasCreate_v2(&hdl);
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDgetriBatched(hdl, nX, aconst2, nX, piv2, Carray, nX, inf2, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	free(piv2);
	free(inf2);

	// copy Carray[0] = carr into LT 
	status = cublasCreate_v2(&hdl);
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDcopy(hdl, nX * nX, carr, 1, LT, 1);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;
	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	free(carr);
	free(Carray);

	double * Vee = (double *)malloc(nX * sizeof(double));
	if (Vee == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nX, 1, nX, 1, LT, nX, KT1T, nX, 0, Vee, nX);

	status = cublasCreate_v2(&hdl);
	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	status = cublasDgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N, nX, 1, nX, al, LT, nX, KT1T, nX, be, Vee, nX);

	if (status != CUBLAS_STATUS_SUCCESS)
		return;

	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	free(LT);
	free(Aarray2);
	free(al);
	free(be);

	variance[idx] = (double)0;

	// sum(V.*V,1)
	for (i = 0; i < nX; i++){
		variance[idx] = variance[idx] + Vee[i] * Vee[i];
	}
	free(Vee);
	variance[idx] = KT1T1[0] - variance[idx];
	free(KT1T1);

	// Add noise to variance
	double noise = exp(2 * lh[3]);
	variance[idx] = variance[idx] + noise;
	if (variance[idx] < 0) variance[idx] = 0;
}