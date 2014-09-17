#ifndef __CU_EIGVAL_UTIL_H__
#define __CU_EIGVAL_UTIL_H__

#include "cu_head.h"

__device__ inline float fsgnf(const float &val)
{
    return (val < 0.0f) ? -1.0f : 1.0f;
}

__device__ inline int ceilPow2(int n)
{
    if (0 == (n & (n-1)))     return n;
    int exp;
    frexpf((float)n, &exp);
    return (1 << exp);
}

// Compute midpoint of interval
__device__ inline float midPoint(const float lo, const float up)
{
    float mid;
    if (fsgnf(lo) == fsgnf(up))
        mid = lo + (up - lo) * 0.5f;
    else
        mid = (lo + up) * 0.5f;
    return mid;
}


// Subdivide interval if active and not converged
template<class T> __device__ void
subdivideInterval(const unsigned int tid, float *s_lo, float *s_up, T *s_lcnt, T *s_ucnt, const unsigned int nth_actv, float &lo, float &up, unsigned int &lcnt, unsigned int &ucnt, float &mid, unsigned int &converge)
{
    if (tid < nth_actv) {
        lo = s_lo[tid];
        up = s_up[tid];
        lcnt = s_lcnt[tid];
        ucnt = s_ucnt[tid];

        if (lo != up) {
            mid = midPoint(lo, up);
            converge = 0;
        } else if ((ucnt - lcnt) > 1) {
            converge = 0;
        }
    }
}

// Compute number of eigenvals that are smaller than val
__device__ inline unsigned int
nEigvals(float *d_a, float *d_b, const unsigned int n, const float val, const unsigned int tid, const unsigned int nth_actv, float *s_a, float *s_b, unsigned int converge)
{
    float  delta = 1.0f, t = 0.0f;
    unsigned int count = 0;
    __syncthreads();

    if (threadIdx.x < n) {
        s_a[threadIdx.x] = *(d_a + threadIdx.x);
        s_b[threadIdx.x] = *(d_b + threadIdx.x);
    }
    __syncthreads();

    if ((tid < nth_actv) && (converge == 0)) {
        for (unsigned int k = 0; k < n; ++k) {
            t = t * (s_b[k]*s_b[k] / delta) - val*val;
            delta = s_a[k]*s_a[k] + t;
            count += (delta < 0) ? 1 : 0;
        }
    }
    return count;
}
__device__ inline unsigned int
nEigvalsLarge(float *d_a, float *d_b, const unsigned int n, const float val, const unsigned int tid, const unsigned int nth_actv, float *s_a, float *s_b, unsigned int converge)
{
    float  delta = 1.0f, t = 0.0f;
    unsigned int count = 0;
    unsigned int rem = n;

    for (unsigned int i = 0; i < n; i += blockDim.x) {
        __syncthreads();

        if ((i + threadIdx.x) < n) {
            s_a[threadIdx.x] = *(d_a + i + threadIdx.x);
            s_b[threadIdx.x] = *(d_b + i + threadIdx.x);
        }
        __syncthreads();

        if (tid < nth_actv) {
            for (unsigned int k = 0; k < fminf(rem,blockDim.x); ++k) {
                t = t * (s_b[k]*s_b[k] / delta) - val*val;
                delta = s_a[k]*s_a[k] + t;
                count += (delta < 0) ? 1 : 0;
            }
        }
        rem -= blockDim.x;
    }
    return count;
}
__device__ inline unsigned int
nEigvalsHuge(float *d_a, float *d_b, const unsigned int n, const float val, const unsigned int tid, const unsigned int nth_actv, float *s_a, float *s_b, unsigned int converge)
{
    float  delta = 1.0f, t = 0.0f;
    unsigned int count = 0;
    unsigned int rem = n;

    for (unsigned int i = 0; i < n; i += blockDim.x) {
        __syncthreads();

        if ((i + threadIdx.x) < n) {
            s_a[threadIdx.x] = *(d_a + i + threadIdx.x);
            s_b[threadIdx.x] = *(d_b + i + threadIdx.x);
        }
        __syncthreads();

        if (tid < nth_actv) {
            for (unsigned int k = 0; k < fminf(rem,blockDim.x); ++k) {
                t = t * (s_b[k]*s_b[k] / delta) - val*val;
                delta = s_a[k]*s_a[k] + t;
                count += (delta < 0) ? 1 : 0;
            }
        }
        rem -= blockDim.x;
    }
    return count;
}

// Check if interval converged and store appropriately
template<class S, class T> __device__ void
checkConverge(unsigned int tid, float *s_lo, float *s_up, T *s_lcnt, T *s_ucnt, float lo, float up, S lcnt, S ucnt, float tao)
{
    s_lcnt[tid] = lcnt;
    s_ucnt[tid] = ucnt;
    float t0 = fabsf(up - lo);
    float t1 = fmaxf(fabsf(lo), fabsf(up)) * tao;

    if (t0 <= fmaxf(MIN_INTERVAL, t1)) {
        float lambda = midPoint(lo, up);
        s_lo[tid] = lambda;
        s_up[tid] = lambda;
    } else {
        s_lo[tid] = lo;
        s_up[tid] = up;
    }
}

// Store all non-empty intervals from the subdivision of the interval
template<class S, class T> __device__ void
storeIntervals(unsigned int tid, const unsigned int nth_actv, float *s_lo, float *s_up, T *s_lcnt, T *s_ucnt, float lo, float mid, float up, const S lcnt, const S mcnt, const S ucnt, float tao, unsigned int &compact2, T *s_cmpl_exc, unsigned int &active2)
{
    if ((lcnt != mcnt) && (mcnt != ucnt)) {
        checkConverge(tid, s_lo, s_up, s_lcnt, s_ucnt, lo, mid, lcnt, mcnt, tao);
        active2 = 1;
        s_cmpl_exc[threadIdx.x] = 1;
        compact2 = 1;
    } else {
        active2 = 0;
        s_cmpl_exc[threadIdx.x] = 0;
        if (lcnt != mcnt)
            checkConverge(tid, s_lo, s_up, s_lcnt, s_ucnt, lo, mid, lcnt, mcnt, tao);
        else
            checkConverge(tid, s_lo, s_up, s_lcnt, s_ucnt, mid, up, mcnt, ucnt, tao);
    }
}

// Store intervals that have already converged
template<class T, class S> __device__ void
storeConverged(float *s_lo, float *s_up, T *s_lcnt, T *s_ucnt, float &lo, float &mid, float &up, S &lcnt, S &mcnt, S &ucnt, T *s_cmpl, unsigned int &compact2, const unsigned int nth_actv)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int muli = ucnt - lcnt;
    if (muli == 1) {
        s_lo[tid] = lo;
        s_up[tid] = up;
        s_lcnt[tid] = lcnt;
        s_ucnt[tid] = ucnt;
        s_ucnt[tid + nth_actv] = 0;
        s_cmpl[tid] = 0;
    } else {
        mcnt = lcnt + (muli >> 1);
        s_lo[tid] = lo;
        s_up[tid] = up;
        s_lcnt[tid] = lcnt;
        s_ucnt[tid] = mcnt;
        mid = lo;
        s_ucnt[tid + nth_actv] = ucnt;
        s_cmpl[tid] = 1;
        compact2 = 1;
    }
}
template<class T, class S> __device__ void
storeConverged(float *s_lo, float *s_up, T *s_lcnt, T *s_ucnt, float &lo, float &mid, float &up, S &lcnt, S &mcnt, S &ucnt, T *s_cmpl, unsigned int &compact2, const unsigned int nth_actv, unsigned int &active2)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int muli = ucnt - lcnt;
    if (muli == 1) {
        s_lo[tid] = lo;
        s_up[tid] = up;
        s_lcnt[tid] = lcnt;
        s_ucnt[tid] = ucnt;
        active2 = 0;
        s_cmpl[tid] = 0;
    } else {
        mcnt = lcnt + (muli >> 1);
        s_lo[tid] = lo;
        s_up[tid] = up;
        s_lcnt[tid] = lcnt;
        s_ucnt[tid] = mcnt;
        mid = lo;
        active2 = 1;
        s_cmpl[tid] = 1;
        compact2 = 1;
    }
}

// Create indices for compaction
template<class T> __device__ void
createIndices(T *s_cmpl_exc, unsigned int nth_comp)
{
    unsigned int offset = 1;
    const unsigned int tid = threadIdx.x;

    for (int d = (nth_comp >> 1); d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            unsigned int  ai = offset*(2*tid+1)-1;
            unsigned int  bi = offset*(2*tid+2)-1;
            s_cmpl_exc[bi] =  s_cmpl_exc[bi] + s_cmpl_exc[ai];
        }
        offset <<= 1;
    }

    for (int d = 2; d < nth_comp; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < (d-1)) {
            unsigned int  ai = offset*(tid+1) - 1;
            unsigned int  bi = ai + (offset >> 1);
            s_cmpl_exc[bi] =  s_cmpl_exc[bi] + s_cmpl_exc[ai];
        }
    }
    __syncthreads();
}

// Perform stream compaction for second child intervals
template<class T> __device__ void compactIntervals(float *s_lo, float *s_up, T *s_lcnt, T *s_ucnt, float mid, float up, unsigned int mcnt, unsigned int ucnt, T *s_cmpl, unsigned int nth_actv, unsigned int active2)
{
    const unsigned int tid = threadIdx.x;
    if ((tid < nth_actv) && (active2 == 1)) {
        unsigned int addr_w = nth_actv + s_cmpl[tid];
        s_lo[addr_w] = mid;
        s_up[addr_w] = up;
        s_lcnt[addr_w] = mcnt;
        s_ucnt[addr_w] = ucnt;
    }
}

// Perform initial scan for compaction of intervals containing one and muliple eigenvalues; also do initial scan to build blocks
__device__ inline void scanInitial(const unsigned int tid, const unsigned int tid_2, const unsigned int nth_actv, const unsigned int nth_comp, unsigned short *s_cl_one, unsigned short *s_cl_mul, unsigned short *s_cl_blocking, unsigned short *s_cl_helper)
{
    unsigned int offset = 1;
    for (int d = (nth_comp >> 1); d > 0; d >>= 1) {
        __syncthreads();

        if (tid < d) {
            unsigned int  ai = offset*(2*tid+1);
            unsigned int  bi = offset*(2*tid+2)-1;
            s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai - 1];
            s_cl_mul[bi] = s_cl_mul[bi] + s_cl_mul[ai - 1];

            if ((s_cl_helper[ai - 1] != 1) || (s_cl_helper[bi] != 1)) {
                if (s_cl_helper[ai - 1] == 1)
                    s_cl_helper[bi] = 1;
                else if (s_cl_helper[bi] == 1)
                    s_cl_helper[ai - 1] = 1;
                else {
                    unsigned int temp = s_cl_blocking[bi] + s_cl_blocking[ai - 1];
                    if (temp > MAX_THREADS_BLOCK) {
                        s_cl_helper[ai - 1] = 1;
                        s_cl_helper[bi] = 1;
                    } else {
                        s_cl_blocking[bi] = temp;
                        s_cl_blocking[ai - 1] = 0;
                    }
                }
            }
        }
        offset <<= 1;
    }

    for (int d = 2; d < nth_comp; d <<= 1) {
        offset >>= 1;
        __syncthreads();

        if (tid < (d-1)) {
            unsigned int  ai = offset*(tid+1) - 1;
            unsigned int  bi = ai + (offset >> 1);
            s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai];
            s_cl_mul[bi] = s_cl_mul[bi] + s_cl_mul[ai];
        }
    }
}

// Perform scan to obtain number of eigenvalues before a specific block
__device__ inline void scanSumBlocks(const unsigned int tid, const unsigned int tid_2, const unsigned int nth_actv, const unsigned int nth_comp, unsigned short *s_cl_blocking, unsigned short *s_cl_helper)
{
    unsigned int offset = 1;

    for (int d = nth_comp >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            unsigned int ai = offset*(2*tid+1)-1;
            unsigned int bi = offset*(2*tid+2)-1;
            s_cl_blocking[bi] += s_cl_blocking[ai];
        }
        offset *= 2;
    }

    for (int d = 2; d < (nth_comp - 1); d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < (d-1)) {
            unsigned int ai = offset*(tid+1) - 1;
            unsigned int bi = ai + (offset >> 1);
            s_cl_blocking[bi] += s_cl_blocking[ai];
        }
    }
    __syncthreads();

    if (0 == tid) {
        s_cl_helper[nth_actv - 1] = s_cl_helper[nth_comp - 1];
        s_cl_blocking[nth_actv - 1] = s_cl_blocking[nth_comp - 1];
    }
}

// Compute addresses to obtain compact list of block start addresses
__device__ inline void scanCompactBlocksStartAddress(const unsigned int tid, const unsigned int tid_2, const unsigned int nth_comp, unsigned short *s_cl_blocking, unsigned short *s_cl_helper)
{
    s_cl_blocking[tid] = s_cl_helper[tid];
    if (tid_2 < nth_comp)
        s_cl_blocking[tid_2] = s_cl_helper[tid_2];
    __syncthreads();

    unsigned int offset = 1;
    for (int d = (nth_comp >> 1); d > 0; d >>= 1) {
        __syncthreads();

        if (tid < d) {
            unsigned int  ai = offset*(2*tid+1)-1;
            unsigned int  bi = offset*(2*tid+2)-1;
            s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];
        }
        offset <<= 1;
    }

    for (int d = 2; d < nth_comp; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < (d-1)) {
            unsigned int  ai = offset*(tid+1) - 1;
            unsigned int  bi = ai + (offset >> 1);
            s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];
        }
    }
}


// Perform final stream compaction before writing data to global memory
__device__ inline void compactStreamsFinal(const unsigned int tid, const unsigned int tid_2, const unsigned int nth_actv, unsigned int &offset_mul_lambda, float *s_lo, float *s_up, unsigned short *s_lcnt, unsigned short *s_ucnt, unsigned short *s_cl_one, unsigned short *s_cl_mul, unsigned short *s_cl_blocking, unsigned short *s_cl_helper, unsigned int is_one_lambda, unsigned int is_one_lambda_2, float &lo, float &up, float &lo_2, float &up_2, unsigned int &lcnt, unsigned int &ucnt, unsigned int &lcnt_2, unsigned int &ucnt_2, unsigned int c_block_iend, unsigned int c_sum_block, unsigned int c_block_iend_2, unsigned int c_sum_block_2)
{
    lo = s_lo[tid];
    up = s_up[tid];
    if (tid_2 < nth_actv) {
        lo_2 = s_lo[tid_2];
        up_2 = s_up[tid_2];
    }
    __syncthreads();

    unsigned int ptr_w = 0;
    unsigned int ptr_w_2 = 0;
    unsigned int ptr_blocking_w = 0;
    unsigned int ptr_blocking_w_2 = 0;

    ptr_w = (1 == is_one_lambda) ? s_cl_one[tid] : s_cl_mul[tid] + offset_mul_lambda;

    if (0 != c_block_iend)
        ptr_blocking_w = s_cl_blocking[tid];
    if (tid_2 < nth_actv) {
        ptr_w_2 = (1 == is_one_lambda_2) ? s_cl_one[tid_2] : s_cl_mul[tid_2] + offset_mul_lambda;
        if (0 != c_block_iend_2)
            ptr_blocking_w_2 = s_cl_blocking[tid_2];
    }
    __syncthreads();

    s_lo[ptr_w] = lo;
    s_up[ptr_w] = up;
    s_lcnt[ptr_w] = lcnt;
    s_ucnt[ptr_w] = ucnt;

    if (0 != c_block_iend) {
        s_cl_blocking[ptr_blocking_w + 1] = c_block_iend - 1;
        s_cl_helper[ptr_blocking_w + 1] = c_sum_block;
    }
    if (tid_2 < nth_actv) {
        s_lo[ptr_w_2] = lo_2;
        s_up[ptr_w_2] = up_2;
        s_lcnt[ptr_w_2] = lcnt_2;
        s_ucnt[ptr_w_2] = ucnt_2;

        if (0 != c_block_iend_2) {
            s_cl_blocking[ptr_blocking_w_2 + 1] = c_block_iend_2 - 1;
            s_cl_helper[ptr_blocking_w_2 + 1] = c_sum_block_2;
        }
    }
}

// Write data to global memory
__device__ inline void writeToGmem(const unsigned int tid, const unsigned int tid_2, const unsigned int nth_actv, const unsigned int n_mul, float *d_l_one, float *g_u_one, unsigned int *g_p_one, float *g_l_mul, float *g_u_mul, unsigned int *g_lcnt_mul, unsigned int *g_ucnt_mul, float *s_left, float *s_right, unsigned short *s_lcnt, unsigned short *s_ucnt, unsigned int *g_blocks_mul, unsigned int *g_blocks_mul_sum, unsigned short *s_cmpl, unsigned short *s_cl_helper, unsigned int offset_mul_lambda )
{
    if (tid < offset_mul_lambda) {
        d_l_one[tid] = s_left[tid];
        g_u_one[tid] = s_right[tid];
        g_p_one[tid] = s_ucnt[tid];
    } else {
        g_l_mul[tid - offset_mul_lambda] = s_left[tid];
        g_u_mul[tid - offset_mul_lambda] = s_right[tid];
        g_lcnt_mul[tid - offset_mul_lambda] = s_lcnt[tid];
        g_ucnt_mul[tid - offset_mul_lambda] = s_ucnt[tid];
    }

    if (tid_2 < nth_actv) {
        if (tid_2 < offset_mul_lambda) {
            d_l_one[tid_2] = s_left[tid_2];
            g_u_one[tid_2] = s_right[tid_2];
            g_p_one[tid_2] = s_ucnt[tid_2];
        } else {
            g_l_mul[tid_2 - offset_mul_lambda] = s_left[tid_2];
            g_u_mul[tid_2 - offset_mul_lambda] = s_right[tid_2];
            g_lcnt_mul[tid_2 - offset_mul_lambda] = s_lcnt[tid_2];
            g_ucnt_mul[tid_2 - offset_mul_lambda] = s_ucnt[tid_2];
        }
    }

    if (tid <= n_mul) {
        g_blocks_mul[tid] = s_cmpl[tid];
        g_blocks_mul_sum[tid] = s_cl_helper[tid];
    }
    if (tid_2 <= n_mul) {
        g_blocks_mul[tid_2] = s_cmpl[tid_2];
        g_blocks_mul_sum[tid_2] = s_cl_helper[tid_2];
    }
}


#endif
