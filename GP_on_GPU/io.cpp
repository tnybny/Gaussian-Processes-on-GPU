#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MAXCHAR 24

void readmatrix(FILE *file, double *data, int nrows, int ncols, char order, char * delim){
  int rcnt = 0;
  char * L = (char *)malloc(sizeof(char) * ncols * MAXCHAR);
  if (L == nullptr)
  {
	  printf("memory allocation failed\n");
	  return;
  }
  while (fgets(L, sizeof(char) * ncols * MAXCHAR, file) && (rcnt < nrows)){
	int ccnt = 0;
	char *next_token1 = NULL;
    char *cL = strtok_s(L, delim, &next_token1);    
    while((ccnt < ncols) && (cL != NULL)){
      if(order == 'R'){
	data[rcnt * ncols + ccnt] = strtod(cL, NULL);
      }else{
	data[rcnt + ccnt * nrows] = strtod(cL, NULL);
      }
      cL = strtok_s(NULL, delim, &next_token1);
      ccnt++;
    }
    rcnt++;
  }
  free(L);
  fclose (file);  
}

void printmatrix(FILE *file, double *data, int nrows, int ncols, char order, char * delim){
  int i,j;
  for(i = 0; i < nrows; i++){
    for(j = 0; j < ncols - 1; j++){
      if(order == 'R')
	fprintf(file,"%.10lf%c",data[i*ncols + j],*delim);

      else
	fprintf(file,"%.10lf%c",data[i+j*nrows],*delim);
    }
    if(order == 'R')
      fprintf(file,"%.10lf\n",data[ncols*(i+1) - 1]);
    else
      fprintf(file,"%.10lf\n",data[i+(ncols-1)*nrows]);
  }
}
