// This version global memory align
#include "cu_head.h"

__global__ void thread_eigvec_v3(float *d_vec, float *d_a, float *d_b, U32 n, U32 threads, float *d_val, float *d_l, float *d_u, float *d_dl, float *d_du)
{
   U32 k = blockIdx.x * blockDim.x + threadIdx.x;
   if (k > threads) return;
   float dl, du, sl, su, val;
   val = d_val[k];
   // ldl
   d_dl[0*threads+k] = dl = d_a[0]*d_a[0] - val*val;
   for (int i=0; i<n-1; i++) {
//      d_l[i*n+k] = sl = d_a[i]*d_b[i+1] / dl;
      d_l[i*threads+k] = sl = d_a[i]*d_b[i+1] / dl;
//      d_dl[(i+1)*n+k] = dl = d_a[i+1]*d_a[i+1] + d_b[i+1]*d_b[i+1] - val*val - sl * sl * dl;
      d_dl[(i+1)*threads+k] = dl = d_a[i+1]*d_a[i+1] + d_b[i+1]*d_b[i+1] - val*val - sl * sl * dl;
   }
   // udu
   d_du[(n-1)*threads+k] = du = d_a[n-1]*d_a[n-1] + d_b[n-1]*d_b[n-1] - val*val;
//   d_du[(n-1)*n+k] = du = d_a[n-1]*d_a[n-1] + d_b[n-1]*d_b[n-1] - val*val;
   for (int i=n-2; i>=0; i--) {
      d_u[i*threads+k] = su = d_a[i] * d_b[i+1] / du;
//      d_u[i*n+k] = su = d_a[i] * d_b[i+1] / du;
      d_du[i*threads+k] = du = d_a[i]*d_a[i] + d_b[i]*d_b[i] - val*val - su * su * du;
//      d_du[i*n+k] = du = d_a[i]*d_a[i] + d_b[i]*d_b[i] - val*val - su * su * du;
   }
   // The index of the smallest gamma
   du = fabsf(du);
   dl = fabsf(dl);
   U32 index = du < dl ? 0 : n-1;
   float gamma = fminf(du, dl);
   for (int i=1; i<n-1; i++){
//      float tmp = fabsf(d_du[i*n+k] - d_dl[(i-1)*n+k]*d_l[(i-1)*n+k]*d_l[(i-1)*n+k]);
      float tmp = fabsf(d_du[i*threads+k] - d_dl[(i-1)*threads+k]*d_l[(i-1)*threads+k]*d_l[(i-1)*threads+k]);
      if (gamma > tmp){
         gamma = tmp;
         index = i;
      }
   }
   // Obtain eigenvector & Square Sum
   float sqsum = 1.0f, tmp;
   d_vec[k*n+index] = 1;
   if(index != 0){
//      d_vec[k*n+index-1] = tmp = -d_l[(index-1)*n+k];
      d_vec[k*n+index-1] = tmp = -d_l[(index-1)*threads+k];
      sqsum += tmp * tmp;
   }
   if(index != n-1){
//      d_vec[k*n+index+1] = tmp = -d_u[index*n+k];
      d_vec[k*n+index+1] = tmp = -d_u[index*threads+k];
      sqsum += tmp * tmp;
   }
   if(index >= 1){
      tmp = d_vec[k*n+index-1];
      for(U32 i=index-1; i>0; i--){
//         d_vec[k*n+i-1] = tmp = -d_l[(i-1)*n+k]*tmp;
         d_vec[k*n+i-1] = tmp = -d_l[(i-1)*threads+k]*tmp;
         sqsum += tmp * tmp;
      }
   }
   if(index < n-2){
      tmp = d_vec[k*n+index+1];
      for(U32 i=index+2; i<n; i++){
//         d_vec[k*n+i] = tmp = -d_u[(i-1)*n+k]*tmp;
         d_vec[k*n+i] = tmp = -d_u[(i-1)*threads+k]*tmp;
         sqsum += tmp * tmp;
      }
   }
   sqsum = rsqrtf(sqsum);
   for(int i=0; i<n; i++)
      d_vec[k*n+i] = d_vec[k*n+i] * sqsum;
}

/*
 * The function calculate k eigenvectors based on k eigenvalues saved in p_val,
 * and save the eigenvectors in p_vec. Diagonal and Bidiagonal array are 
 * saved in d_a and d_b with size n, separately. d_b[0] = 0;
 */
void par_eigenMat_v3(float *p_vec, float *d_a, float *d_b, U32 n, U32 n_lo, U32 n_up, float *p_val)
{
   // Cuda Event for timing
   float time;
   cudaEvent_t start, stop;
   cudaErrors(cudaEventCreate(&start));
   cudaErrors(cudaEventCreate(&stop));

   float *d_l, *d_dl, *d_u, *d_du, *d_vec, *d_val;
   U32 k = n_up - n_lo;
   float size = k*n*sizeof(float);
   cudaErrors(cudaMalloc((void **)&d_val, k*sizeof(float)));
   cudaErrors(cudaMalloc((void **)&d_vec, size));
   cudaErrors(cudaMalloc((void **)&d_l,   size));
   cudaErrors(cudaMalloc((void **)&d_u,   size));
   cudaErrors(cudaMalloc((void **)&d_dl,  size));
   cudaErrors(cudaMalloc((void **)&d_du,  size));
   cudaErrors(cudaMemcpy(d_val, p_val+n_lo, k*sizeof(float), cudaMemcpyHostToDevice));
   
   U32 n_blocks = (k - 1) / MAX_VEC_THREAD + 1;
   cudaEventRecord(start, 0);
   for (U32 i=0; i<iters; i++){
      thread_eigvec_v3<<<n_blocks,MAX_VEC_THREAD>>>(d_vec, d_a, d_b, n, k, d_val, d_l, d_u, d_dl, d_du);
      cudaErrors(cudaGetLastError());
   }
   cudaEventRecord(stop, 0);
   cudaErrors(cudaEventSynchronize(stop));
   cudaErrors(cudaEventElapsedTime(&time, start, stop));
   cout << "Parall Eigenvec Time : " << fixed<<setprecision(3) << time/iters << " ms" << endl;

   cudaErrors(cudaMemcpy(p_vec, d_vec, size, cudaMemcpyDeviceToHost));
   cudaErrors(cudaFree(d_l));
   cudaErrors(cudaFree(d_u));
   cudaErrors(cudaFree(d_dl));
   cudaErrors(cudaFree(d_du));
   cudaErrors(cudaFree(d_vec));
   cudaErrors(cudaFree(d_val));
}

