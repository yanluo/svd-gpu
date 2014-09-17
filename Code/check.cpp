#include "common.h"
#include <cmath>

int check_orthogonal(float *p_eigMat, U32 n, float precision)
{
   float err = 0.0f;
   float res;
   for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
         res = 0;
         for(int k=0; k<n; k++){
            res += p_eigMat[k*n+i] * p_eigMat[k*n+j];
         }
         if(i==j)   err += abs(res-1.0f);
         else       err += abs(res);
      }
   }
   if (err/n/n > precision){
      cout << "Error = " << err/n/n << endl;
      cout << "Eigenvectors are not Orthogonal" << endl;
      exit(-1);
   }
   return 0;
}

int check_decomp(float *p_eigMat, float *p_eigval, float *p_a, float *p_b, U32 n, float precision)
{
   float err = 0.0f;
   float res;
   for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
         res = 0;
         for(int k=0; k<n; k++){
            res += p_eigMat[k*n+i] * p_eigMat[k*n+j] * p_eigval[k];
         }
//         cout << i << " " << j << " " << res << endl;
         if(i==j)         err += abs(res-p_a[i]);
         else if(i-j==1)  err += abs(res-p_b[i]);
         else if(j-i==1)  err += abs(res-p_b[j]);
         else             err += abs(res);
      }
   }
   if (err/n/n > precision){
      cout << "Error = " << err/n/n << endl;
      cout << "Decomposition is not correct!" << endl;
      exit(-1);
   }
   return 0;
}

int check_equal(float *p1, float *p2, U32 n, float tao)
{
    int err = 0;
    for (int i=0; i<n; i++){
        if(abs(p1[i]-p2[i]) < tao * 10){
            continue;
        } else {
            cout << "The error is from position " << fixed<<setprecision(6) << i << " " << p1[i] << " " << p2[i] << endl;
            err = 1;
        }
    }
    if (err != 0)            exit(-1);
    return 0;
}
