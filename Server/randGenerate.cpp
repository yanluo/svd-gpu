#include "common.h"
#include <sys/time.h>

void randGenerate2D(float * pA, U32 m, U32 n, float randlo, float randup)
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   srand((unsigned)tv.tv_usec);
   float ratio = randup - randlo;
   for(int i=0; i<m; i++)
      for(int j=0; j<n; j++)
         *(pA + n*i + j) = (float)rand()/RAND_MAX * ratio + randlo;
}

void randGenerate1D(float * pA, U32 k, U32 n, float randlo, float randup)
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   srand((unsigned)tv.tv_usec);
   float ratio = randup - randlo;
   for(int i=0; i<n; i++)
      *(pA + i + k) = (float)rand()/RAND_MAX * ratio + randlo;
}

