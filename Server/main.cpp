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

using namespace std;

U32 iters;
int main(int argc, char * argv[])
{

   // Initialize Input Parameters
   cuda_st cuda;

   // Initialize Socket
   int    socket_fd, connect_fd;
   struct sockaddr_in servaddr;
   if( (socket_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1 ){
      printf("create socket error: %s(errno: %d)\n",strerror(errno),errno);
      exit(0);
   }
   memset(&servaddr, 0, sizeof(servaddr));
   servaddr.sin_family = AF_INET;
   servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
   servaddr.sin_port = htons(DEFAULT_PORT);
   int opt = 1;
   setsockopt(socket_fd,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
   if( bind(socket_fd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1){
      printf("bind socket error: %s(errno: %d)\n",strerror(errno),errno);
      exit(0);
   }
   if( listen(socket_fd, 10) == -1){
      printf("listen socket error: %s(errno: %d)\n",strerror(errno),errno);
      exit(0);
   }
   if( (connect_fd = accept(socket_fd, (struct sockaddr*)NULL, NULL)) == -1){
      printf("accept socket error: %s(errno: %d)",strerror(errno),errno);
   }
   int nb = recv(connect_fd, &cuda, sizeof(cuda), MSG_WAITALL);
   if(!fork()){
      if(send(connect_fd, &nb, sizeof(nb),0) == -1)
         perror("send error");
      close(connect_fd);
      exit(0);
   }
   cout << cuda.n << "\t[" << cuda.l<<","<<cuda.u << ")\t[" << cuda.n_u<<","<<cuda.n_l << ")\t" << cuda.err << "\t" << cuda.whole << endl;

   // Initialize Input Data
   U32 n = cuda.n;
   float * p_a = new float[n];
   float * p_b = new float[n+1];
   float * p_val = new float[n];
   initZero(p_val, 0, n);
   p_b[0] = 0; p_b[n] = 0;
   nb = recv(connect_fd, p_a, sizeof(float)*n, MSG_WAITALL);
   nb = recv(connect_fd, p_b, sizeof(float)*(n+1), MSG_WAITALL);

   float ug = cuda.u, lg = cuda.l;
   U32 n_ug = cuda.n_u, n_lg = cuda.n_l;
   float tao = cuda.err;
   iters = cuda.iters;

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
   par_eigenMat_v3(p_vec, d_a, d_b, n, n_l, n_u, p_val);
   par_eigenMat_v3(p_vec, d_a, d_b, n, n_l, n_u, p_val);

   if( send(connect_fd, p_vec, sizeof(float)*k*n, MSG_WAITALL) < 0)
   {
      printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
      exit(0);
   }
   if( send(connect_fd, p_vec, sizeof(float)*k*n, MSG_WAITALL) < 0)
   {
      printf("send msg error: %s(errno: %d)\n", strerror(errno), errno);
      exit(0);
   }

   // Free memory
   cudaErrors(cudaFree(d_a));
   cudaErrors(cudaFree(d_b));
   cudaDeviceReset();
   close(connect_fd);
   delete []p_a;
   delete []p_b;
   delete []p_val;
   delete []p_vec;
   p_a = p_b = p_val = NULL;
   close(socket_fd);
  
   return 0;
}


