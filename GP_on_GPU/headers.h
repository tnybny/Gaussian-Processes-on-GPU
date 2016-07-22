#include <stdio.h>

/* FILE IO */

/*
 * Reads data from a file and stores in a column oriented array
 * delim specifies the delimiter, can be space tab or comma
 */
void readmatrix(FILE *file, double *data, int nrows, int ncols, char order, char * delim);

/*
 * PRINTMATRIX - Prints a column oriented data array as a matrix to a file with specified delimiter
 */
void printmatrix(FILE *file, double *data, int nrows, int ncols, char order, char * delim);



void monitor(double *trY, double *teY, double *trx, double *tex, int nY, int ntrx, int ntex, double *lh, double *Z, double *teYhat, double *teVhat, double*teYcorr, double alpha, int corrM, double *teM);
void smooth(double *AL, double *Sl, double *Su, double *Z, int nr, int nc, double lambdau, double lambdal, double Mu, double Ml);

//void ztest(double *P, double *X, int nX, double *mean, int nMean, double *sigma, int nSigma, int tail);
