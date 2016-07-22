#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cublas_v2.h>
#include "utils.cuh"
#include "gpchange.cuh"

/* Minimize the marginal or conditional log likelihood of given data 
 * as a function of the hyper-parameters using conjugate gradient descent.
 *
 * The current values of the hyper-parameters are chosen as the starting point. 
 * The "length" gives the length of the run: if it is positive, it gives the 
 * maximum number of line searches,if negative its absolute gives the max 
 * allowed number of function evaluations. The "red" parameter indicates the
 * reduction in function value to be expected in the first line-search. 
 *  
 * The function returns when either its length is up, or if no further progress
 * can be made (ie, we are at a (local) minimum, or so close that due to
 * numerical problems, we cannot get any closer). The function sets the final solution
 * as the updated log hyper-parameters for the covariance function.
 * 
 * The Polack-Ribiere flavour of conjugate gradients is used to compute search
 * directions, and a line search using quadratic and cubic polynomial
 * approximations and the Wolfe-Powell stopping criteria is used together with
 * the slope ratio method for guessing initial step sizes. Additionally a bunch
 * of checks are made to make sure that exploration is taking place and that
 * extrapolation will not be unboundedly large.
 * 
 */
__device__ void minimize(double *X, int nX, double *Y, int nY, int length, int red, double *lh0, int nh, int numthreads, int method, int n){
	if((method != 1) && (method != 2)){
		return;
	}
	char order = 'C';
	double f0, d3, x2, f2, d2, f3, x4 = 0, f4 = 0, d4 = 0, A, B;
	double *df0, *df3, *L0, *l, *l1;
	int i = 0, ls_failed = 0, j;
	l = (double *) malloc(sizeof(double) * nh);
	if (l == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	for(j = 0; j < nh; j++) l[j] = lh0[j];
	double int1 = 0.1, ext = 3.0 , ratio = 10, sig = 0.1;
	double rho = sig / 2;
	int mx = 20;
	if(red == -1) red = 1;
	df0 = (double *) malloc(sizeof(double) * nh);
	if (df0 == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	df3 = (double *) malloc(sizeof(double) * nh);
	if (df3 == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	L0 = (double *) malloc(sizeof(double) * nh);
	if (L0 == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	/* compute loglikelihood and derivatives */
	if(method == 1){
		computeMLLChol(&f0, df0, X, Y, nX, nY, lh0, nh, order);
	}
	/* end computation */
	//printf("Line search: Iteration = %d, value = %.6lf\n",i,f0); 
	double *s = (double *) malloc(sizeof(double) * nh);
	if (s == nullptr)
	{
		printf("could not allocate memory\n");
		return;
	}
	for (j = 0; j < nh; j++) s[j] = -1 * df0[j];
	if(fabsf(length) < 0) i = i + 1;
	//double d0 = -1 * cblas_ddot(nh,s,1,s,1);
	
	cublasHandle_t hdl;
	cublasStatus_t status = cublasCreate_v2(&hdl);
	
	if(status != CUBLAS_STATUS_SUCCESS)
		return;
	
	double ret;
	status = cublasDdot(hdl, nh, s, 1, s, 1, &ret);
	
	if (status != CUBLAS_STATUS_SUCCESS)
		return;
	
	if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
	{
		printf("cublasDestroy failed\n");
		return;
	}

	double d0 = -1 * ret;
	double x3 = red / (1 - d0);
	while(i < length){
		if(length > 0) i = i + 1;
		//cblas_dcopy(nh, l, 1, L0, 1);
		status = cublasCreate_v2(&hdl);

		if (status != CUBLAS_STATUS_SUCCESS)
			return;

		status = cublasDcopy(hdl, nh, l, 1, L0, 1);
		
		if (status != CUBLAS_STATUS_SUCCESS)
			return;
		
		if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
		{
			printf("cublasDestroy failed\n");
			return;
		}

		double F0 = f0;
		double *dF0 = (double *) malloc(sizeof(double) * nh);
		if (dF0 == nullptr)
		{
			printf("could not allocate memory\n");
			return;
		}
		for(j = 0; j < nh; j++) dF0[j] = df0[j];
		int m;
		if(length > 0) m = mx; else m = minu(mx, -1 * (length + i));
		l1 = (double *)malloc(sizeof(double) * nh);
		if (l1 == nullptr)
		{
			printf("could not allocate memory\n");
			return;
		}
		while(1){
			x2 = 0; f2 = f0; d2 = d0; f3 = f0;
			//df3 = (double *) malloc(sizeof(double) * nh);
			for(j = 0; j < nh; j++) df3[j] = df0[j];
			int success = 0;
			while ((!success) && (m > 0)){
				m -= 1;
				if(length < 0) i = i + 1;
				for(j = 0; j < nh; j++) l1[j] = l[j] + s[j] * x3;
				/* compute loglikelihood and derivatives */
				if(method == 1)  {  
					computeMLLChol(&f3, df3, X, Y, nX, nY, l1, nh, order);
				}
				int s1 = 0;
				if(isnan(f3) || isinf(f3)) s1 = 1;
				for(j = 0;j < nh; j++){
					if(isnan(df3[j]) || isinf(df3[j])) s1 = 1;
				}
				if(!s1) success = 1;
				else x3 = (x2 + x3)/2;
			}
			if(f3 < F0){
				for(j = 0; j < nh; j++) L0[j] = l[j] + s[j] * x3;	
				F0 = f3;
				for(j = 0; j < nh; j++) dF0[j] = df3[j];
			}
			//d3 = cblas_ddot(nh, df3, 1, s, 1); //recompute slope
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			status = cublasDdot(hdl, nh, df3, 1, s, 1, &ret);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			d3 = ret;
			
			if((d3 > sig * d0) || (f3 > f0 + x3 * rho * d0) || (m == 0)) break;

			double x1 = x2, f1 = f2, d1 = d2;
			x2 = x3; f2 = f3; d2 = d3;
			A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1);
			B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1);
			x3 = x1 - d1 * pow(x2 - x1, 2) / (B + sqrt(B * B - A * d1 * (x2 - x1)));
			if(isnan(x3) || isinf(x3) || x3 < 0)
				x3 = x2 * ext;                           
			else if(x3 > x2 * ext)
				x3 = x2 * ext;
			else if(x3 < x2 + int1 * (x2 - x1))
				x3 = x2 + int1 * (x2 - x1);
		}

		free(l1);

		while (((fabsf(d3) > -sig * d0) || (f3 > f0 + x3 * rho * d0)) && m > 0){
			if(d3 > 0 || f3 > f0+ x3 * rho * d0){                        
				x4 = x3; f4 = f3; d4 = d3;                      
			}else{
				x2 = x3; f2 = f3; d2 = d3;
			}
			if (f4 > f0){           
				x3 = x2 - (0.5 * d2 * pow(x4 - x2, 2) / (f4 - f2 - d2 * (x4 - x2))); 
			}else{
				A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2);                    
				B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2);
				x3 = x2 + (sqrt(B * B - A * d2 * pow(x4 - x2, 2)) - B) / A;        
			}
			if(isnan(x3) || isinf(x3)){
				x3 = (x2 + x4) / 2;               
			}
			x3 = maxu(minu(x3, x4 - int1 * (x4 - x2)), x2 + int1 * (x4 - x2)); 
			l1 = (double *) malloc(sizeof(double) * nh);
			if (l1 == nullptr)
			{
				printf("could not allocate memory\n");
				return;
			}
			for (j = 0; j < nh; j++) l1[j] = l[j] + s[j] * x3;
			/* compute loglikelihood and derivatives */
			if(method == 1){
				computeMLLChol(&f3, df3, X, Y, nX, nY, l1, nh, order);
			}

			/* end computation */

			if(f3 < F0){
				for(j = 0; j < nh; j++){
					L0[j] = l[j] + s[j] * x3; dF0[j] = df3[j];
				}
				F0 = f3; 
			}         
			m--;
			if(length < 0) i = i + 1;
			//d3 = cblas_ddot(nh, df3, 1, s, 1);
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			status = cublasDdot(hdl, nh, df3, 1, s, 1, &ret);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			d3 = ret;
		}
		if ((fabsf(d3) < -sig * d0) && (f3 < f0 + x3 * rho * d0)){
			for(j = 0; j < nh; j++) {l[j] += s[j] * x3;	}
			//printf("Line search: Iteration = %d, value = %.6lf\n",i,f0);
			f0 = f3;
			//double _int = (cblas_ddot(nh, df3, 1, df3, 1) - cblas_ddot(nh, df0, 1, df3, 1)) / (cblas_ddot(nh, df0, 1, df0, 1));
			double ret1, ret2, ret3;
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			status = cublasDdot(hdl, nh, df3, 1, df3, 1, &ret1);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			status = cublasDdot(hdl, nh, df0, 1, df3, 1, &ret2);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			status = cublasDdot(hdl, nh, df0, 1, df0, 1, &ret3);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			double _int = (ret1 - ret2) / ret3;
			for(j = 0; j < nh; j++) s[j] = s[j] * _int - df3[j];
			for(j = 0; j < nh; j++) df0[j] = df3[j];
			d3 = d0; 

			//d0 = cblas_ddot(nh,df0,1,s,1);
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			status = cublasDdot(hdl, nh, df0, 1, s, 1, &ret);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;
			
			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			d0 = ret;
			if(d0 > 0){            
				for(j = 0; j < nh; j++) s[j] = -1 * df0[j];
				//d0 = -1 * cblas_ddot(nh,s,1,s,1);
				status = cublasCreate_v2(&hdl);

				if (status != CUBLAS_STATUS_SUCCESS)
					return;
				
				status = cublasDdot(hdl, nh, s, 1, s, 1, &ret);
				
				if (status != CUBLAS_STATUS_SUCCESS)
					return;
				
				if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
				{
					printf("cublasDestroy failed\n");
					return;
				}

				d0 = -1 * ret;
			}
			x3 = x3 * minu(ratio, d3 / (d0 - FLT_MIN));          
			ls_failed = 0;                              
		}else{
			for(j = 0; j < nh; j++){
				l[j] = L0[j]; df0[j] = dF0[j]; dF0[j] = df3[j];
			}

			f0 = F0;
			if(ls_failed == 1|| i > fabsf(length)) break;      
			for(j = 0; j < nh; j++) s[j] = -1 * df0[j];
			//d0 = -1 * cblas_ddot(nh, s, 1, s, 1);
			status = cublasCreate_v2(&hdl);

			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			status = cublasDdot(hdl, nh, s, 1, s, 1, &ret);
			
			if (status != CUBLAS_STATUS_SUCCESS)
				return;

			if (cublasDestroy_v2(hdl) != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDestroy failed\n");
				return;
			}

			d0 = -1 * ret;
			x3 = 1/(1 - d0);                     
			ls_failed = 1;                     
		}
		free(dF0);
	}
	for(j = 0; j < nh; j++) lh0[j] = l[j];
	free(l);
	free(df0);
	free(df3);
	free(L0);
	free(s);
	free(df3);
}
