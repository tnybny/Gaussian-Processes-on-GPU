#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "headers.h"
#include "utils.cuh"
#include "gp.cuh"
#include "minimize.cuh"

#define MAX_DAYS 12784

__global__ void predict(double * d_X, double * d_Y, double * d_variance, double * d_mean, int nte, int nX, double * d_lh)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id == 0)
	{
		double * XTmp = (double *)malloc((nX + 1) * sizeof(double));
		if (XTmp == nullptr)
		{
			printf("could not allocate memory\n");
			return;
		}

		double * YTmp = (double *)malloc((nX + 1) * sizeof(double));
		if (YTmp == nullptr)
		{
			printf("could not allocate memory\n");
			return;
		}

		int length = 100;
		int red = 1;
		int nH = 4;

		minimize(d_X, nX, d_Y, 1, length, red, d_lh, nH, 5, 1, 10);

		cublasHandle_t hadl;
		cublasStatus_t status = cublasCreate_v2(&hadl);

		if (status != CUBLAS_STATUS_SUCCESS)
			return;
		for (int i = nX; i < (nX + 1); i++){ //i < (nX + nte)
			
			//cblas_dcopy(i + 1, d_X, 1, XTmp, 1);
			status = cublasDcopy(hadl, i + 1, d_X, 1, XTmp, 1);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			//cblas_dcopy(i + 1, d_Y, 1, YTmp, 1);
			status = cublasDcopy(hadl, i + 1, d_Y, 1, YTmp, 1);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			printf(" New X = %.6lf , New Y = %.6lf\n", XTmp[i], YTmp[i]);
			
			gp(d_lh, XTmp, i, YTmp, XTmp[i], d_mean, d_variance, i - nX);

			printf(" predicted mean = %.6lf , predicted variance = %.6lf\n", d_mean[i - nX], d_variance[i - nX]);
			
			XTmp = myrealloc(i + 1, i + 2, XTmp);
			if (XTmp == nullptr)
			{
				printf("could not allocate memory\n");
				return;
			}
			
			YTmp = myrealloc(i + 1, i + 2, YTmp);
			if (YTmp == nullptr)
			{
				printf("could not allocate memory\n");
				return;
			}
		}
		cublasDestroy_v2(hadl);
		free(XTmp);
		free(YTmp);
	}
}

int main(){
	// initialize a matrix for number of rows
	int nrows = 731;
	int ncols = 1;
	int nH = 4;
	int nte = 365;
	int nX = nrows;	
	char delim = ',';
	char order = 'C';	
	int ntotal = nrows + nte;
	double z = 2.236;
	double * lh0 = (double *)malloc(nH * sizeof(double));
	if (lh0 == nullptr)
	{
		printf("could not allocate memory\n");
		return -1;
	}

	double * h_Y = (double *)malloc((nrows + nte) * ncols * sizeof(double));
	if (h_Y == nullptr)
	{
		printf("could not allocate memory\n");
		return -1;
	}
	double * h_X = (double *)malloc((nrows + nte) * ncols * sizeof(double));
	if (h_X == nullptr)
	{
		printf("could not allocate memory\n");
		return -1;
	}
	double * h_lh = (double *)malloc(nH * sizeof(double));
	if (h_lh == nullptr)
	{
		printf("could not allocate memory\n");
		return -1;
	}

	FILE *file = fopen("C:/Users/bramach2/Documents/GP_on_GPU/ncep1.STemp.1.csv", "r");
	readmatrix(file, h_Y, ntotal, ncols, order, &delim);
	fclose(file);

	int i;
	//initialize X and divide Y into training and testing
	for(i = 0; i < ntotal; i++){
		h_X[i] = i + 1;
	}

	// Build the covariance matrix for training data - initial hyperparameters
	lh0[0] = log(0.09);//log(l)
	lh0[1] = log(20.0); //log(sf2)
	lh0[2] = log(365.0);//log(p)
	lh0[3] = log(5.0); //log(sn2)

	//minimize(h_X, nX, h_Y, 1, length, red, lh0, h_lh, nH, 5, 1, 10);

	h_lh[0] = lh0[0];
	h_lh[1] = lh0[1];
	h_lh[2] = lh0[2];
	h_lh[3] = lh0[3];

	//UNCOMMENT NEXT 4 LINES TO IGNORE MINIMIZE
	/*
	h_lh[0] = -0.2967;
	h_lh[1] = -0.1452;
	h_lh[2] = 5.8965;
	h_lh[3] = -1.1027;
	*/

	// choose GPU device
	if (cudaSetDevice(1) != cudaSuccess)
	{
		printf("Error setting GPU device\n");
		return -1;
	}
	// increase per thread heap size to 64MB
	if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 * 1024 * 1024) != cudaSuccess)
	{
		printf("Error setting device limit\n");
		return -1;
	}

	// cudaMalloc
	double * h_variance = (double *)malloc(nte * sizeof(double));
	if (h_variance == nullptr)
	{
		printf("could not allocate memory\n");
		return -1;
	}
	double * h_mean = (double *)malloc(nte * sizeof(double));
	if (h_mean == nullptr)
	{
		printf("could not allocate memory\n");
		return -1;
	}

	double * d_X;
	double * d_Y; 
	double * d_variance;
	double * d_mean;
	double * d_lh;

	if (cudaMalloc(&d_X, ntotal * ncols * sizeof(double)) != cudaSuccess)
	{
		printf("cudaMalloc failed\n");
		return -1;
	}
	if(cudaMalloc(&d_Y, ntotal * ncols * sizeof(double)) != cudaSuccess)
	{
		printf("cudaMalloc failed\n");
		return -1;
	}
	if(cudaMalloc(&d_variance, nte * sizeof(double)) != cudaSuccess)
	{
		printf("cudaMalloc failed\n");
		return -1;
	}
	if (cudaMalloc(&d_mean, nte * sizeof(double)) != cudaSuccess)
	{
		printf("cudaMalloc failed\n");
		return -1;
	}
	if (cudaMalloc(&d_lh, nH * sizeof(double)) != cudaSuccess)
	{
		printf("cudaMalloc failed\n");
		return -1;
	}

	// cudaMemcpy
	if(cudaMemcpy(d_X, h_X, ntotal * ncols * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("cudaMemcpy failed\n");
		return -1;
	}
	if(cudaMemcpy(d_Y, h_Y, ntotal * ncols * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("cudaMemcpy failed\n");
		return -1;
	}
	/*if (cudaMemcpy(d_lh, h_lh, nH * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("cudaMemcpy failed\n");
		return -1;
	}*/

	// kernel call
	predict<<<1, 1>>>(d_X, d_Y, d_variance, d_mean, nte, nX, d_lh);

	cudaError_t error;

	if ((error = cudaGetLastError()) != cudaSuccess)
	{
		printf("Kernel execution error!\n");
		return -1;
	}

	// cudaMemcpy
	if(cudaMemcpy(h_variance, d_variance, nte * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("cudaMemcpy failed\n");
		return -1;
	}
	if (cudaMemcpy(h_mean, d_mean, nte * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("cudaMemcpy failed\n");
		return -1;
	}
	printf(" predicted mean = %.6lf , predicted variance = %.6lf\n", h_mean[0], h_variance[0]);
	double max, min;
	for(i = 0; i < nte; i++)
	{
		max = h_mean[i] + z * sqrt(h_variance[i]);
		min = h_mean[i] - z * sqrt(h_variance[i]);
		/*if(h_Y[nX + i] > max){
			printf("HOT: PREDICTED MEAN = %.4lf \tPREDICTED VARIANCE = %.4lf \tMAX = %.4lf \tMIN = %.4lf\n", h_mean[i], h_variance[i], max, min);
		}
		else if(h_Y[nX + i] < min){
			printf("COLD: PREDICTED MEAN = %.4lf \tPREDICTED VARIANCE = %.4lf \tMAX = %.4lf \tMIN = %.4lf\n", h_mean[i], h_variance[i], max, min);
		}
		else{	
			printf("NONE: PREDICTED MEAN = %.4lf \tPREDICTED VARIANCE = %.4lf \tMAX = %.4lf \tMIN = %.4lf\n", h_mean[i], h_variance[i], max, min);
		}*/
	}

	// cudaFree
	if(cudaFree(d_X) != cudaSuccess)
	{
		printf("cudaFree failed\n");
		return -1;
	}
	if(cudaFree(d_Y) != cudaSuccess)
	{
		printf("cudaFree failed\n");
		return -1;
	}
	if(cudaFree(d_mean) != cudaSuccess)
	{
		printf("cudaFree failed\n");
		return -1;
	}
	if(cudaFree(d_variance) != cudaSuccess)
	{
		printf("cudaFree failed\n");
		return -1;
	}
	if(cudaFree(d_lh) != cudaSuccess)
	{
		printf("cudaFree failed\n");
		return -1;
	}

	free(h_X);
	free(h_Y);
	free(lh0);
	free(h_lh);
	free(h_mean);
	free(h_variance);
}
