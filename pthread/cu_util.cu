#include "cu_head.h"
#include <iostream>

using namespace std;

int findCudaDevice(int &devID)
{
   cudaDeviceProp deviceProp;
   cudaErrors(cudaSetDevice(devID));
   cudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
   cout << "GPU Device " << devID << ": \"" << deviceProp.name << "\" with compute capability " << deviceProp.major << "." << deviceProp.minor << endl;
   return 0;
}

void _check(cudaError_t cudaStatus, char const *const func, const char *const file, int const line)
{
    if(cudaStatus){
        cout << "CUDA error at " << file << ":" << line << " code=" << cudaGetErrorString(cudaStatus) << "/" << func << endl;
        cudaDeviceReset();
        exit(-1);
    }
}

void gPrint2D(float *d_data, U32 height, U32 length)
{
   float * h_data;
   U32 size = height * length * sizeof(float);
   h_data = (float *) malloc(size);
   cudaErrors(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
   for (int i=0; i<height; i++){
      cout << "i = " << i << endl;
      for(int j=0; j<length; j++){
         cout << setprecision(6) << *(h_data + i*length + j) << " ";
      }
      cout << endl;
   }
   cout << endl;
   free(h_data);
}

void gPrint1D(float *d_data, U32 n)
{
   float * h_data;
   U32 size = n * sizeof(float);
   h_data = (float *) malloc(size);
   cudaErrors(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
   for (int i = 0; i < n; i++){
      cout << setprecision(6) << *(h_data + i) << " ";
   }
   cout << "\n" << endl;
   free(h_data);
}

void gPrint1D(unsigned int *d_data, U32 n)
{
   unsigned int * h_data;
   U32 size = n * sizeof(int);
   h_data = (unsigned int *) malloc(size);
   cudaErrors(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
   for (int i = 0; i < n; i++){
      cout << *(h_data + i) << " ";
   }
   cout << "\n" << endl;
   free(h_data);
}


