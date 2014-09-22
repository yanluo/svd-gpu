#include "common.h"
#include <stdlib.h>
#include <string.h>

int getSize(int argc, char * argv[], U32 &n)
{
   for (unsigned int i=1; i<argc; i++){
       if (strcmp(argv[i],"-n")==0){
           n = atoi(argv[i+1]);
           return n;
       }
   }
   n = SIZE_DEFAULT;
   return 0;
}

int getRandPara(int argc, char * argv[], int &randLowBound, int &randUpBound)
{
   for (unsigned int i=1; i<argc; i++) {
      if (strcmp(argv[i],"-b")==0) {
         randLowBound = atoi(argv[i+1]);
         randUpBound  = atoi(argv[i+2]);
         return 1;
      }
   }
   randLowBound = RANDLO_DEFAULT;
   randUpBound  = RANDUP_DEFAULT;
   return 0;
}

int getPrecision(int argc, char * argv[], float &precision)
{
   for (unsigned int i=1; i<argc; i++){
       if (strcmp(argv[i],"-p")==0){
           precision = atof(argv[i+1]);
           return 1;
       }
   }
   precision = PRECISION_DEFAULT;
   return 0;
}

int getPartial(int argc, char * argv[], U32 &procs, U32 &partial)
{
   for (unsigned int i=1; i<argc; i++){
       if (strcmp(argv[i],"-r")==0){
           partial = atoi(argv[i+1]);
           procs = atoi(argv[i+2]);
           return 1;
       }
   }
   procs = 1;
   partial = 1;
   return 0;
}

int getRange(int argc, char * argv[], U32 &ucnt, U32 &lcnt)
{
   for (unsigned int i=1; i< argc; i++){
       if (strcmp(argv[i],"-c")==0){
           lcnt = atoi(argv[i+1]);
           ucnt = atoi(argv[i+2]);
           return 1;
       }
   }
   return 0;
}

int getIters(int argc, char * argv[], U32 &iters)
{
   for (unsigned int i=1; i<argc; i++){
       if (strcmp(argv[i],"-i")==0){
           iters = atoi(argv[i+1]);
           return iters;
       }
   }
   iters = ITERS_DEFAULT;
   return 0;
}

int getThread(int argc, char * argv[], U32 &threads)
{
   for (unsigned int i=1; i<argc; i++){
       if (strcmp(argv[i],"-t")==0){
           threads = atoi(argv[i+1]);
           return threads;
       }
   }
   threads = THREAD_DEFAULT;
   return 0;
}

