#include "common.h"
#include "cu_head.h"
#include "eigen.h"
#include <stdio.h>
#include <pthread.h>
#include <cuda_profiler_api.h>

using namespace std;

U32 iters;
int main(int argc, char * argv[])
{

   // Initialize Input Data
   U32 n, procs, partial;
   int randlo, randup;
   float tao;
   getSize(argc, argv, n);
   getRandPara(argc, argv, randlo, randup);
   getPrecision(argc, argv, tao);
   getPartial(argc, argv, procs, partial);
   getIters(argc, argv, iters);

   float * p_a = new float[n];
   float * p_b = new float[n+1];
//   float * p_val = new float[n];
   randGenerate1D(p_a, 0, n, randlo, randup);
   randGenerate1D(p_b, 1, n-1, randlo, randup);
//   initZero(p_val, 0, n);
   p_b[0] = 0; p_b[n] = 0;

   // Obtain Boundary 
   float u, l;
   U32 n_u, n_l;
   boundary(p_a, p_b, n, argc, argv, u, l, n_u, n_l);
   cout << "Bound:\t" << u << "\t" << l << endl;
   cout << "Range:\t" << n_u << "\t" << n_l << endl;

   // Initialize Cuda Device
   int devID = 0;
   findCudaDevice(devID);
   cudaDeviceReset();

   float *d_a, *d_b;
   cudaErrors(cudaMalloc((void **)&d_a, sizeof(float)*n));
   cudaErrors(cudaMalloc((void **)&d_b, sizeof(float)*(n+1)));
   cudaErrors(cudaMemcpy(d_b, p_b, sizeof(float)*(n+1), cudaMemcpyHostToDevice));
   cudaErrors(cudaMemcpy(d_a, p_a, sizeof(float)*n, cudaMemcpyHostToDevice));

   // Obtain Eigenvalue Parallel
   U32 k = n_u - n_l;
   float * p_val = new float[n];
   par_eigenval(p_val, d_a, d_b, n, l, u, n_l, n_u, tao);
//   print1D(p_val, n);

   // Obtain Eigenvectors Serially
//   cudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
//   cudaProfilerStart();
   float * p_vec;
   p_vec = new float[k*n];

//   par_eigenMat_v3(p_vec, d_a, d_b, n, k, p_val);
   par_eigenMat_v3(p_vec, d_a, d_b, n, n_l, n_u, p_val);
//   print2D(p_vec, n, n);
//   cudaProfilerStop();

/*
   // check orthogonal and correct
   check_orthogonal(p_vec, n, tao)
   check_decomp(p_vec, p_val, p_a, p_b, n, tao)
*/

   // Free memory
   cudaErrors(cudaFree(d_a));
   cudaErrors(cudaFree(d_b));
   cudaDeviceReset();
   delete []p_a;
   delete []p_b;
   delete []p_val;
   delete []p_vec;
   p_a = p_b = p_val = p_vec = NULL;
  
   return 0;
}
