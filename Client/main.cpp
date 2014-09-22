#include "common.h"
#include "cu_head.h"
#include "eigen.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netinet/in.h>

#define DEFAULT_PORT 8000  
#define SERVER "129.63.205.82"

using namespace std;

U32 iters;
void *thread_func(void* struc);
int main(int argc, char * argv[])
{
   // Initialize Input Parameters
   cuda_st cuda;
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
   cout << whole << " " << part << endl;

   // Initialize Input Data
   float * p_a = new float[n];
   float * p_b = new float[n+1];
   float * p_val = new float[n];
   randGenerate1D(p_a, 0, n, randlo, randup);
   randGenerate1D(p_b, 1, n-1, randlo, randup);
   initZero(p_val, 0, n);
   p_b[0] = 0; p_b[n] = 0;

   // Total TIme
   float time;
   struct timeval tv_begin, tv_end;
   gettimeofday(&tv_begin, NULL);

   // Obtain Boundary 
   float ug, lg;
   U32 n_ug, n_lg;
   boundary(p_a, p_b, n, argc, argv, ug, lg, n_ug, n_lg);
   cout << "Bound:\t" << ug << "\t" << lg << endl;
   cout << "Range:\t" << n_ug << "\t" << n_lg << endl;

   cuda.n = n;
   cuda.u = ug;
   cuda.l = lg;
   cuda.n_u = n_ug;
   cuda.n_l = n_lg; 
   cuda.err = tao;
   cuda.part = part;
   cuda.whole = whole;
   cuda.iters = iters;

   // Initialize Socket
   int    socket_fd, rec_len;
   struct sockaddr_in     servaddr;
   if( (socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0 ){
      printf("create socket error: %s(errno: %d)\n",strerror(errno),errno);
      exit(0);
   }
   memset(&servaddr, 0, sizeof(servaddr));
   servaddr.sin_family = AF_INET;
   servaddr.sin_port = htons(DEFAULT_PORT);
   if( inet_pton(AF_INET, SERVER, &servaddr.sin_addr) <= 0){
      printf("inet_pton error for %s\n",argv[1]);
      exit(0);
   }
   if( connect(socket_fd, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0){
      printf("connect error: %s(errno: %d)\n",strerror(errno),errno);
      exit(0);
   }

   // Transfer Size to Server
   if(!fork()){
      if( send(socket_fd, &cuda, sizeof(cuda_st), 0) < 0)
      {
         printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
         exit(0);
      }
      U32 ret;
      if((rec_len = recv(socket_fd, &ret, sizeof(U32),0)) == -1) {
         perror("recv error");
         exit(1);
      }
      cout << cuda.n << "\t[" << cuda.l<<","<<cuda.u << ")\t[" << cuda.n_u<<","<<cuda.n_l << ")\t" << cuda.err << "\t" << cuda.whole << endl;
      // Transfer other data to Server
      if( send(socket_fd, p_a, sizeof(float)*n, 0) < 0 )
      {
         printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
         exit(0);
      }
      if( send(socket_fd, p_b, sizeof(float)*(n+1), 0 )< 0 )
      {
         printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
         exit(0);
      }
      exit(0);
   }

   float u,l;
   U32 n_u, n_l;
   divide(p_a, p_b, n, ug, lg, n_ug, n_lg, u, l, n_u, n_l, cuda.part, cuda.whole);
   int devID = 0;
   findCudaDevice(devID);
   cout << "[" << l << "," << u << ")\t[" << n_l << "," << n_u << ")" <<endl;

   float *d_a, *d_b;
   cudaErrors(cudaMalloc((void **)&d_a, sizeof(float)*n));
   cudaErrors(cudaMalloc((void **)&d_b, sizeof(float)*(n+1)));
   cudaErrors(cudaMemcpy(d_b, p_b, sizeof(float)*(n+1), cudaMemcpyHostToDevice));
   cudaErrors(cudaMemcpy(d_a, p_a, sizeof(float)*n, cudaMemcpyHostToDevice));

   // Obtain Eigenvalue Parallel
   U32 k = n_u - n_l;
//   float * p_val = new float[n];
   par_eigenval(p_val, d_a, d_b, n, l, u, n_l, n_u, tao);

   // Obtain Eigenvectors Serially
   float * p_vec;
   p_vec = new float[k*n];
   if(!fork()){
      if((recv(socket_fd, p_vec, sizeof(float)*k*n,MSG_WAITALL)) == -1) {
         perror("recv error");
         exit(1);
      }
      if((recv(socket_fd, p_vec, sizeof(float)*k*n,MSG_WAITALL)) == -1) {
         perror("recv error");
         exit(1);
      }
      gettimeofday(&tv_end, NULL);
      time = (tv_end.tv_sec-tv_begin.tv_sec)*1000 + (tv_end.tv_usec-tv_begin.tv_usec)/1000.0;
      cout << "Total Time : " << fixed<<setprecision(3) << time << " ms" << endl;
      exit(0);
   }

   par_eigenMat_v3(p_vec, d_a, d_b, n, n_l, n_u, p_val);
   par_eigenMat_v3(p_vec, d_a, d_b, n, n_l, n_u, p_val);

   gettimeofday(&tv_end, NULL);
   time = (tv_end.tv_sec-tv_begin.tv_sec)*1000 + (tv_end.tv_usec-tv_begin.tv_usec)/1000.0;
   cout << "Total Time : " << fixed<<setprecision(3) << time << " ms" << endl;

   // Free memory
   cudaErrors(cudaFree(d_a));
   cudaErrors(cudaFree(d_b));
   cudaDeviceReset();
   close(socket_fd);
   delete []p_a;
   delete []p_b;
   delete []p_val;
   delete []p_vec;
   p_a = p_b = p_val = NULL;
  
   return 0;
}
   
