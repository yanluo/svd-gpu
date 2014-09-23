#include "common.h"
#include "cu_head.h"
#include "eigen.h"
#include <stdio.h>
#include <pthread.h>
#include <cuda_profiler_api.h>

using namespace std;

U32 iters;
pthread_mutex_t print;
void *thread_func(void* struc);
int main(int argc, char * argv[])
{

   // Initialize Input Data
   U32 n, procs, partial;
   int randlo, randup;
   float tao;
   U32 whole, part;
   getSize(argc, argv, n);
   getRandPara(argc, argv, randlo, randup);
   getPrecision(argc, argv, tao);
   getPartial(argc, argv, procs, partial);
   getDivide(argc, argv, whole, part);
   getIters(argc, argv, iters);

   float * p_a = new float[n];
   float * p_b = new float[n+1];
   float * p_val = new float[n];
   randGenerate1D(p_a, 0, n, randlo, randup);
   randGenerate1D(p_b, 1, n-1, randlo, randup);
   initZero(p_val, 0, n);
   p_b[0] = 0; p_b[n] = 0;

   // Initialize Cuda Device
   int devCount;
   cudaErrors(cudaGetDeviceCount(&devCount));
//   devCount = 6;
//   cout << devCount << " GPU Device(s)" << endl;

   // Obtain Boundary 
   float ug, lg;
   U32 n_ug, n_lg;
   boundary(p_a, p_b, n, argc, argv, ug, lg, n_ug, n_lg);
//   cout << "Bound:\t" << ug << "\t" << lg << endl;
//   cout << "Range:\t" << n_ug << "\t" << n_lg << endl;

   cuda_st cuda[devCount];
   pthread_t pthread[devCount];
   pthread_mutex_init(&print, NULL);

   part = 1; whole = 100;
for(part = 99; part >= whole/2; part--){
   cout << part << endl;
   for(int i=0; i<devCount; i++) {
      cuda[i].p_a = p_a;
      cuda[i].p_b = p_b;
      cuda[i].p_val = p_val;
      cuda[i].n = n;
      cuda[i].u = ug;
      cuda[i].l = lg;
      cuda[i].n_u = n_ug;
      cuda[i].n_l = n_lg; 
      cuda[i].err = tao;
      cuda[i].p = i;
      cuda[i].devs = devCount;
      cuda[i].part = part;
      cuda[i].whole = whole;
   }

   for(int i=0; i<devCount; i++) {
      pthread_create(&pthread[i], NULL, thread_func, (void*)&cuda[i]);
   }
   for(int i=0; i<devCount; i++) {
      pthread_join(pthread[i], NULL);
   }
}

   // Free memory
   pthread_mutex_destroy(&print);
   pthread_exit(NULL);
   delete []p_a;
   delete []p_b;
   delete []p_val;
   p_a = p_b = p_val = NULL;
   p_a = p_b = NULL;
  
   return 0;
}
   
void *thread_func(void* struc){
   cuda_st * data = (cuda_st*)struc;
   float * p_a = data->p_a;
   float * p_b = data->p_b;
   float * p_val = data->p_val;
   U32 n = data->n;
   U32 n_lg = data->n_l, n_ug = data->n_u;
   float lg = data->l, ug = data->u;
   float tao = data->err;
   int devID = data->p;
   U32 devCount = data->devs;
   U32 part = data->part, whole = data->whole;
   float u,l;
   U32 n_u, n_l;
   divide(p_a, p_b, n, ug, lg, n_ug, n_lg, u, l, n_u, n_l, devID+1, devCount,part, whole);
   findCudaDevice(devID);
   cudaDeviceReset();
   pthread_mutex_lock (&print);
//   cout << devID << "\t[" << l << "," << u << ")\t[" << n_l << "," << n_u << ")" <<endl;
   cout << devID << "\t" << l << " " << u << "\t" << n_l << " " << n_u <<endl;
   pthread_mutex_unlock (&print);

   float *d_a, *d_b;
   cudaErrors(cudaMalloc((void **)&d_a, sizeof(float)*n));
   cudaErrors(cudaMalloc((void **)&d_b, sizeof(float)*(n+1)));
   cudaErrors(cudaMemcpy(d_b, p_b, sizeof(float)*(n+1), cudaMemcpyHostToDevice));
   cudaErrors(cudaMemcpy(d_a, p_a, sizeof(float)*n, cudaMemcpyHostToDevice));

   float time;
   struct timeval tv_begin, tv_end;
   gettimeofday(&tv_begin, NULL);

   // Obtain Eigenvalue Parallel
   U32 k = n_u - n_l;
   par_eigenval(p_val, d_a, d_b, n, l, u, n_l, n_u, tao);
   // Obtain Eigenvectors Serially
   float * p_lvec = new float[k*n];
   float * p_rvec = new float[k*n];
   par_eigenMat_v3(p_lvec, p_rvec, d_a, d_b, n, n_l, n_u, p_val);

   gettimeofday(&tv_end, NULL);
   time = (tv_end.tv_sec-tv_begin.tv_sec)*1000 + (tv_end.tv_usec-tv_begin.tv_usec)/1000.0;
   pthread_mutex_lock (&print);
//   cout << "Thread " << devID << " total time : " << fixed<<setprecision(3) << time << " ms" << endl;
   cout << devID << " " << fixed<<setprecision(3) << time << endl;
   pthread_mutex_unlock (&print);

   cudaErrors(cudaFree(d_a));
   cudaErrors(cudaFree(d_b));
   cudaDeviceReset();
   delete []p_lvec;
   delete []p_rvec;

   pthread_exit((void*) 0);

}
