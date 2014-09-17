#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>

#define SIZE_DEFAULT	8
#define RANDLO_DEFAULT	0
#define RANDUP_DEFAULT	1
#define ITERS_DEFAULT	10
#define THREAD_DEFAULT	512
#define PRECISION_DEFAULT	10e-5

using namespace std;

typedef unsigned int U32;

extern U32 iters;

int getSize(int argc, char * argv[], U32 &n);
int getRandPara(int argc, char * argv[], int &randLowBound, int &randUpBound);
int getPrecision(int argc, char * argv[], float &precision);
int getPartial(int argc, char * argv[], U32 &procs, U32 &partial);
int getRange(int argc, char * argv[], U32 &ucnt, U32 &lcnt);
int getIters(int argc, char * argv[], U32 &iters);
int getThread(int argc, char * argv[], U32 &threads);
void boundary(float *p_a, float *p_b, U32 n, int argc, char * argv[], float &up, float &lo, U32 &n_up, U32 &n_lo);

void initZero(float * pA, U32 k, U32 n);
void randGenerate2D(float * pA, U32 m, U32 n, float randlo, float randup);
void randGenerate1D(float * pA, U32 k, U32 n, float randlo, float randup);
void print1D(float *data, U32 n);
void print1D(U32 *data, U32 n);
void print2D(float *data, U32 m, U32 n);
void print2D(U32 *data, U32 m, U32 n);
int load(char *filename, float *data, U32 n);
int check_orthogonal(float *p_eigMat, U32 n, float precision);
int check_decomp(float *p_eigMat, float *p_eigval, float *p_a, float *p_b, U32 n, float precision);
void bi2trisymmc(float *p_a, float *p_b, U32 n);
int check_equal(float *p1, float *p2, U32 n, float tao);

U32 eigenCount(float *p_a, float *p_b, U32 n, float upper);

#endif
