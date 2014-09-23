#include "cu_head.h"
#include "cu_svd_util.h"
#include <stdio.h>

__global__ void separateInterval(float *d_a, float*d_b, U32 n, U32 threads, float *d_up, U32 *d_ucnt, float lg, float ug, U32 n_lg, U32 n_ug, float tao)
{
    __shared__ float s_a[MAX_HUGE_THREAD];
    __shared__ float s_b[MAX_HUGE_THREAD];
    __shared__ unsigned int converge;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float t1 = fmaxf(fabsf(lg), fabsf(ug)) * tao;
    float lo = lg, up = ug, mid;
    unsigned int lcnt = n_lg, ucnt = n_ug, mcnt;
    unsigned int point = (tid + 1) * (ucnt - lcnt) / threads + lcnt;

    if (tid == 0){
        converge = 0;
        d_up[0] = lg;
        d_ucnt[0] = n_lg;
    }
    __syncthreads();

    while (!converge){
       mid = midPoint(lo, up); 
       mcnt = nEigvalsHuge(d_a, d_b, n, mid, tid, threads, s_a, s_b, 0);
       mcnt = fminf(fmaxf(mcnt, lcnt), ucnt);
       if (tid < threads) {
           if (mcnt >= point) {
               up = mid;
               ucnt = mcnt;
           } else {
               lo = mid;
               lcnt = mcnt;
           }
       }
       converge = 1;
       __syncthreads();
       if (tid < threads){
           if(ucnt != point || fabs(up-lo) >= fmaxf(t1, MIN_INTERVAL))
               converge = 0;
       }
       __syncthreads();
    }

    if (tid < threads) {
        d_up[tid+1] = up;
        d_ucnt[tid+1] = ucnt;
    }
}

// Bisection to find eigenvals of a matrix
__global__ void bisectKernelHuge(float *d_a, float *d_b, U32 n, float *d_eig, U32 *d_pos, float *d_lo, U32 *d_lcnt, float tao)
{
    __shared__  float  s_lo[2*MAX_HUGE_THREAD];
    __shared__  float  s_up[2*MAX_HUGE_THREAD];
    __shared__  unsigned int  s_lcnt[2*MAX_HUGE_THREAD];
    __shared__  unsigned int  s_ucnt[2*MAX_HUGE_THREAD];
    __shared__  unsigned int  s_cmpl[2*MAX_HUGE_THREAD + 2];

    __shared__  unsigned int compact2;
    __shared__  unsigned int converge;
    __shared__  unsigned int nth_actv;
    __shared__  unsigned int nth_comp;
    __shared__  unsigned int addr;
    __shared__  unsigned int cnt;

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
        s_lo[0] = d_lo[blockIdx.x];
        s_up[0] = d_lo[blockIdx.x+1];
        s_lcnt[0] = d_lcnt[blockIdx.x];
        s_ucnt[0] = d_lcnt[blockIdx.x+1];
        addr = d_lcnt[blockIdx.x] - d_lcnt[0];
        cnt = s_ucnt[0] - s_lcnt[0];
        compact2 = 0;
        nth_actv = 1;
        nth_comp = 1;
    }

    while (true) {
        if(threadIdx.x == 0)
            converge = 1;
        __syncthreads();

        active2 = 0;
        subdivideInterval(threadIdx.x, s_lo, s_up, s_lcnt, s_ucnt, nth_actv, lo, up, lcnt, ucnt, mid, converge);
        __syncthreads();

        if (converge == 1)    break;
        __syncthreads();
        mcnt = nEigvalsHuge(d_a, d_b, n, mid, threadIdx.x, nth_actv, s_lo, s_up, (lo == up));
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

    if (threadIdx.x < cnt) {
        d_eig[addr + threadIdx.x]  = s_lo[threadIdx.x];
        d_pos[addr + threadIdx.x]  = s_lcnt[threadIdx.x];
    }
}

// Separate the interval to find the best interval
void huge_eigval(float *p_val, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao)
{
    // Cuda Event for timing
#ifdef TIME
    float time;
    cudaEvent_t start, stop;
    cudaErrors(cudaEventCreate(&start));
    cudaErrors(cudaEventCreate(&stop));
#endif

    U32 size = n_up - n_lo;
    U32 interv = (size -1) / MAX_HUGE_THREAD + 1;

    // Allocate Memory on GPU
    float * p_lo = new float[interv+1];
    U32 * p_lcnt = new U32[interv+1];
    float * p_eig = new float[size];
    U32 * p_pos = new U32[size];
    float * d_lo, * d_eig;
    U32 * d_lcnt, * d_pos;
    cudaErrors(cudaMalloc((void **)&d_lo,   sizeof(float)*(interv+1)));
    cudaErrors(cudaMalloc((void **)&d_lcnt, sizeof(U32)*(interv+1)));
    cudaErrors(cudaMalloc((void **)&d_eig, sizeof(float)*size));
    cudaErrors(cudaMalloc((void **)&d_pos, sizeof(U32)*size));

//    cout << interv << " threads" << endl;
#ifdef TIME
    cudaEventRecord(start, 0);
#endif
    for (U32 i=0; i<iters; i++){
        separateInterval<<<1,MAX_HUGE_THREAD>>>(p_a, p_b, n, interv, d_lo, d_lcnt, lo, up, n_lo, n_up, tao);
        cudaErrors(cudaGetLastError());
        cudaErrors(cudaDeviceSynchronize());

        bisectKernelHuge<<<interv, MAX_HUGE_THREAD>>>(p_a, p_b, n, d_eig, d_pos, d_lo, d_lcnt, tao);
        cudaErrors(cudaGetLastError());
        cudaErrors(cudaDeviceSynchronize());
        cudaErrors(cudaMemcpy(p_eig, d_eig, sizeof(float)*size, cudaMemcpyDeviceToHost));
        cudaErrors(cudaMemcpy(p_pos, d_pos, sizeof(U32)*size, cudaMemcpyDeviceToHost));
        for (U32 j=0; j<size; j++){
            p_val[p_pos[j]] =p_eig[j];
        }
    }
#ifdef TIME
    cudaEventRecord(stop, 0);
    cudaErrors(cudaEventSynchronize(stop));
    cudaErrors(cudaEventElapsedTime(&time, start, stop));
    pthread_mutex_lock (&print);
    cout << "Parall Eigenval Time : " << fixed<<setprecision(3) << time/iters << " ms" << endl;
    pthread_mutex_unlock (&print);
#endif

    cudaErrors(cudaFree(d_lo));
    cudaErrors(cudaFree(d_lcnt));
    cudaErrors(cudaFree(d_eig));
    cudaErrors(cudaFree(d_pos));
    delete[] p_lo;
    delete[] p_lcnt;
    delete[] p_eig;
    delete[] p_pos;
}
