#ifndef __CU_HEAD_H__
#define __CU_HEAD_H__

#include "common.h"
#include <cuda_runtime.h>

#define MAX_THREADS		512
#define MAX_SMALL_MATRIX	512
#define MAX_THREADS_BLOCK	256
#define MAX_LARGE_MATRIX	1024
#define MAX_HUGE_THREAD		512
#define MAX_HUGE_BLOCK		1024
#define MAX_HUGE_MATRIX		(MAX_HUGE_THREAD * MAX_HUGE_BLOCK)
#define MAX_VEC_THREAD		512
#define MIN_INTERVAL		5.0e-37
#define cudaErrors(val)		_check((val), #val, __FILE__, __LINE__)

int findCudaDevice(int &devID);
void _check(cudaError_t cudaStatus, char const *const func, const char *const file, int const line);
void gPrint1D(float *d_data, U32 n);
void gPrint1D(unsigned int *d_data, U32 n);
void gPrint2D(float *d_data, U32 height, U32 length);
void par_eigenMat_v3(float *p_vec, float *d_a, float *d_b, U32 n, U32 n_lo, U32 n_up, float *p_val);

void par_eigenval(float *eigval, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao);
void small_eigval(float *p_val, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao);
void large_eigval(float *p_val, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao);
void huge_eigval(float *p_val, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao);


__global__ void bisectKernel(float *p_a, float *p_b, U32 n, float *g_left, U32 *g_lcnt, float lo, float up, U32 n_lo, U32 n_up, float tao);
__global__ void bisectKernelLarge(float *d_a, float *d_b, const unsigned int n, const float lg, const float ug, const unsigned int n_lg, const unsigned int n_ug, float epsilon,unsigned int *d_n_one, unsigned int *d_num_blocks_mul, float *d_l_one, float *d_u_one, unsigned int *d_p_one, float *d_l_mul, float *d_u_mul, unsigned int *d_lcnt_mul, unsigned int *d_ucnt_mul, unsigned int *d_blocks_mul, unsigned int *d_blocks_mul_sum);
__global__ void bisectKernelLarge_One(float *d_a, float *d_b, const unsigned int n, unsigned int num_intervals, float *d_lo, float *d_up, unsigned int *d_po, float tao);
__global__ void bisectKernelLarge_Mul(float *d_a, float *d_b, const unsigned int n, unsigned int *blocks_mul, unsigned int *blocks_mul_sum, float *d_lo, float *d_up, unsigned int *d_lcnt, unsigned int *d_ucnt, float *d_lambda, unsigned int *d_po, float tao);



#endif
