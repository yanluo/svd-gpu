#include "cu_head.h"
#include "cu_svd_util.h"

// Bisection to find eigenvals of a matrix
__global__ void bisectKernel(float *d_a, float *d_b, U32 n, float *d_lo, U32 *d_lcnt, float lg, float ug, U32 n_lg, U32 n_ug, float tao)
{
    __shared__  float  s_lo[MAX_THREADS];
    __shared__  float  s_up[MAX_THREADS];
    __shared__  unsigned int  s_lcnt[MAX_THREADS];
    __shared__  unsigned int  s_ucnt[MAX_THREADS];
    __shared__  unsigned int  s_cmpl[MAX_THREADS + 1];

    __shared__  unsigned int compact2;
    __shared__  unsigned int converge;
    __shared__  unsigned int nth_actv;
    __shared__  unsigned int nth_comp;

    unsigned int *s_cmpl_exc = s_cmpl + 1;
    float  lo = 0.0f, up = 0.0f, mid = 0.0f;
    unsigned int lcnt = 0, ucnt = 0, mcnt = 0;
    unsigned int  active2 = 0;

    s_cmpl[threadIdx.x] = 0;
    s_lo[threadIdx.x] = 0;
    s_up[threadIdx.x] = 0;
    s_lcnt[threadIdx.x] = 0;
    s_ucnt[threadIdx.x] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        s_lo[0] = lg;
        s_up[0] = ug;
        s_lcnt[0] = n_lg;
        s_ucnt[0] = n_ug;
        compact2 = 0;
        nth_actv = 1;
        nth_comp = 1;
    }

    while (true) {
        converge = 1;
        __syncthreads();

        active2 = 0;
        subdivideInterval(threadIdx.x, s_lo, s_up, s_lcnt, s_ucnt, nth_actv, lo, up, lcnt, ucnt, mid, converge);
        __syncthreads();

        if (converge == 1)    break;
        __syncthreads();
        mcnt = nEigvals(d_a, d_b, n, mid, threadIdx.x, nth_actv, s_lo, s_up, (lo == up));
        __syncthreads();

        if (threadIdx.x < nth_actv) {
            if (lo != up)
                storeIntervals(threadIdx.x, nth_actv, s_lo, s_up, s_lcnt, s_ucnt, lo, mid, up, lcnt, mcnt, ucnt, tao, compact2, s_cmpl_exc, active2);
            else
                storeConverged(s_lo, s_up, s_lcnt, s_ucnt, lo, mid, up, lcnt, mcnt, ucnt, s_cmpl_exc, compact2, nth_actv, active2);
        }
        __syncthreads();

        if (compact2 > 0) {
            createIndices(s_cmpl_exc, nth_comp);
            compactIntervals(s_lo, s_up, s_lcnt, s_ucnt, mid, up, mcnt, ucnt, s_cmpl, nth_actv, active2);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            nth_actv += s_cmpl[nth_actv];
            nth_comp = ceilPow2(nth_actv);
            compact2 = 0;
        }
        __syncthreads();

    }
    __syncthreads();

    if (threadIdx.x < n) {
        d_lo[threadIdx.x]  = s_lo[threadIdx.x];
        d_lcnt[threadIdx.x]  = s_lcnt[threadIdx.x];
    }
}


void small_eigval(float *p_val, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao)
{
    // Cuda Event for timing
    float time;
    cudaEvent_t start, stop;
    cudaErrors(cudaEventCreate(&start));
    cudaErrors(cudaEventCreate(&stop));

    // Allocate Memory on GPU
    float * d_lo;
    U32 * d_lcnt;
    cudaErrors(cudaMalloc((void **)&d_lo, sizeof(float)*n));
    cudaErrors(cudaMalloc((void **)&d_lcnt, sizeof(U32)*n));

    // Launch Kernel
    U32 threads = MAX_SMALL_MATRIX;
    cudaEventRecord(start, 0);
    for (U32 i=0; i<iters; i++){
        bisectKernel<<<1, threads>>>(p_a, p_b, n, d_lo, d_lcnt, lo, up, n_lo, n_up, tao);
    }
    cudaErrors(cudaGetLastError());
    cudaErrors(cudaDeviceSynchronize());
    cudaEventRecord(stop, 0);
    cudaErrors(cudaEventSynchronize(stop));
    cudaErrors(cudaEventElapsedTime(&time, start, stop));
    cout << "Parall Eigenval Time : " << fixed<<setprecision(3) << time/iters << " ms" << endl;

    // Copy Memory back to GPU
    float * p_lo = new float[n];
    U32 * p_lcnt = new U32[n];
    cudaErrors(cudaMemcpy(p_lo, d_lo, sizeof(float)*n, cudaMemcpyDeviceToHost));
    cudaErrors(cudaMemcpy(p_lcnt, d_lcnt, sizeof(U32)*n, cudaMemcpyDeviceToHost));
    for (U32 i=0; i<n; i++)  p_val[p_lcnt[i]] = p_lo[i];

    // Free Memory
    cudaErrors(cudaFree(d_lo));
    cudaErrors(cudaFree(d_lcnt));
    delete[] p_lo;
    delete[] p_lcnt;
    p_lo = NULL; p_lcnt = NULL;
}
