#include "common.h"

void print1D(float *data, U32 n)
{
   for(int i=0; i<n; i++){
      cout.setf(ios::fixed);
      cout << setprecision(6) << *(data+i) << " ";
      if (i % 10 == 9)
          cout << "\n" << endl;
   }
   cout << "\n" << endl;
}

void print1D(U32 *data, U32 n)
{
   for(int i=0; i<n; i++){
      cout << *(data+i) << " ";
      if (i % 10 == 9)
          cout << "\n" << endl;
   }
   cout << "\n" << endl;
}

void print2D(float *data, U32 m, U32 n)
{
   for (int i=0; i<m; i++){
      for(int j=0; j<n; j++){
         cout.setf(ios::fixed);
         cout << setprecision(6) << *(data+i*n+j) << " ";
      }
      cout << endl;
   }
   cout << " " << endl;
}

void print2D(U32 *data, U32 m, U32 n)
{
   for (int i=0; i<m; i++){
      for(int j=0; j<n; j++){
         cout.setf(ios::fixed);
         cout << setprecision(6) << *(data+i*n+j) << " ";
      }
      cout << endl;
   }
   cout << " " << endl;
}

