#include "common.h"
#include <float.h>
#include <cmath>

#include <iostream>
#include <stdio.h>
using namespace std;

U32 eigenCount(float *p_a, float *p_b, U32 n, float upper)
{
  U32 cnt = 0;
  float d = 1, t = 0;
  for (U32 i=0; i<n; i++) {
    t = t * (p_b[i]*p_b[i]/d) - upper*upper;
    d = p_a[i]*p_a[i] + t;
    if (d < 0)   cnt++;
  }
  return cnt;
}

void divide(float *p_a, float *p_b, U32 n, float ug, float lg, U32 n_ug, U32 n_lg, float &up, float &lo, U32 &n_up, U32 &n_lo, U32 partial, U32 procs)
{
   float time;
   struct timeval tv_begin, tv_end;
   gettimeofday(&tv_begin, NULL);

   float l, u, m;
   U32 n_m, n_l, n_u;

   n_l = 0; n_u = n; n_m = -1;
   l = lg; u = ug; m = (lg+ug)*0.5f;
   while(n_m != n*partial*1.0/procs){
      n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
      if (n_m > n*partial*1.0/procs) {
         u = m;  n_u = n_m;
      } else {
         l = m;  n_l = n_m;
      }
      m = (u+l)*0.5f;
   }
   up = m; n_up = n_m;
   lo = lg; n_lo = n_lg;

/*
   if (partial == 1 && procs == 1) {
      n_lo = n_lg; lo = lg;
      n_up = n_ug; up = ug;
   } else if (partial == 1){
      n_l = 0; n_u = n; n_m = -1;
      l = lg; u = ug; m = (lg+ug)*0.5f;
      while(n_m != n_lg + (n_ug-n_lg) * partial / procs){
         n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
         if (n_m > n_lg + (n_ug-n_lg) * partial / procs) {
            u = m;  n_u = n_m;
         } else {
            l = m;  n_l = n_m;
         }
         m = (u+l)*0.5f;
      }
      up = m; n_up = n_m;
      lo = lg; n_lo = n_lg;
   } else if (partial == procs) {
      n_l = 0; n_u = n; n_m = -1;
      l = lg; u = ug; m = (lg+ug)*0.5f;
      while(n_m != n_lg + (n_ug-n_lg) * (partial-1) / procs){
         n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
         if (n_m > n_lg + (n_ug-n_lg) * (partial-1) / procs) {
            u = m;  n_u = n_m;
         } else {
            l = m;  n_l = n_m;
         }
         m = (u+l)*0.5f;
      }
      lo = m; n_lo = n_m;
      up = ug; n_up = n;
   } else {
      n_l = 0; n_u = n; n_m = -1;
      l = lg; u = ug; m = (lg+ug)*0.5f;
      while(n_m != n_lg + (n_ug-n_lg) * partial / procs){
         n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
         if (n_m > n * partial / procs) {
            u = m;  n_u = n_m;
         } else {
            l = m;  n_l = n_m;
         }
         m = (u+l)*0.5f;
      }
      up = m; n_up = n_m;

      n_l = 0; n_u = n; n_m = -1;
      l = lg; u = ug; m = (lg+ug)*0.5f;
      while(n_m != n_lg + (n_ug-n_lg) * (partial-1) / procs){
         n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
         if (n_m > n_lg + (n_ug-n_lg) * (partial-1) / procs) {
            u = m;  n_u = n_m;
         } else {
            l = m;  n_l = n_m;
         }
         m = (u+l)*0.5f;
      }
      lo = m; n_lo = n_m;
   }
*/
   gettimeofday(&tv_end, NULL);
   time = (tv_end.tv_sec-tv_begin.tv_sec)*1000 + (tv_end.tv_usec-tv_begin.tv_usec)/1000.0;
   cout << "Boundary Time : " << fixed<<setprecision(3) << time << " ms" << endl;
}

void ger_bound(float &ug, float &lg, float * p_a, float * p_b, U32 n)
{
   float up, lo;
   ug = p_a[0]*p_a[0] + p_b[0]*p_b[0] + abs(p_a[0]*p_b[1]);
   lg = p_a[0]*p_a[0] + p_b[0]*p_b[0] - abs(p_a[0]*p_b[1]);
   for (int i=1; i<n; i++)
   {
      up = p_a[i]*p_a[i] + p_b[i]*p_b[i] + abs(p_a[i-1]*p_b[i]) + abs(p_a[i]*p_b[i+1]);
      lo = p_a[i]*p_a[i] + p_b[i]*p_b[i] - abs(p_a[i-1]*p_b[i]) - abs(p_a[i]*p_b[i+1]);
      if (ug < up)   ug = up;
      if (lg > lo)   lg = lo;
   }
   float bnorm = max(abs(ug),abs(lg));
   float psi_0 = 11 * FLT_EPSILON * bnorm;
   float psi_n = 11 * FLT_EPSILON * bnorm;

   lg = lg - bnorm * 2 * n * FLT_EPSILON - psi_0;
   ug = ug + bnorm * 2 * n * FLT_EPSILON + psi_n;
   ug = max(lg, ug);
   if (lg < 0) lg = 0;
   else        lg = sqrt(lg);
   ug = sqrt(ug);
}

void boundary(float *p_a, float *p_b, U32 n, int argc, char * argv[], float &up, float &lo, U32 &n_up, U32 &n_lo)
{
   float time;
   struct timeval tv_begin, tv_end;
   gettimeofday(&tv_begin, NULL);

   float ug, lg;
   ger_bound(ug, lg, p_a, p_b, n);
   U32 procs, partial;
   float l, u, m;
   U32 n_m, n_l, n_u;
   if (getPartial(argc, argv, procs, partial)) {
      cout << n * partial / procs << endl;
      cout << n * (partial-1) / procs << endl;
      if (partial == 1 && procs == 1) {
         n_lo = 0; lo = lg;
         n_up = n; up = ug;
      } else if (partial == 1){
         n_l = 0; n_u = n; n_m = -1;
         l = lg; u = ug; m = (lg+ug)*0.5f;
         while(n_m != n * partial / procs){
            n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
            if (n_m > n * partial / procs) {
               u = m;  n_u = n_m;
            } else {
               l = m;  n_l = n_m;
            }
            m = (u+l)*0.5f;
         }
         up = m; n_up = n_m;
         lo = lg; lo = lg;
      } else if (partial == procs) {
         n_l = 0; n_u = n; n_m = -1;
         l = lg; u = ug; m = (lg+ug)*0.5f;
         while(n_m != n * (partial-1) / procs){
            n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
            if (n_m > n * (partial-1) / procs) {
               u = m;  n_u = n_m;
            } else {
               l = m;  n_l = n_m;
            }
            m = (u+l)*0.5f;
         }
         lo = m; n_lo = n_m;
         up = ug; n_up = n;
      } else {
         n_l = 0; n_u = n; n_m = -1;
         l = lg; u = ug; m = (lg+ug)*0.5f;
         while(n_m != n * partial / procs){
            n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
            if (n_m > n * partial / procs) {
               u = m;  n_u = n_m;
            } else {
               l = m;  n_l = n_m;
            }
            m = (u+l)*0.5f;
         }
         up = m; n_up = n_m;

         n_l = 0; n_u = n; n_m = -1;
         l = lg; u = ug; m = (lg+ug)*0.5f;
         while(n_m != n * (partial-1) / procs){
            n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
            if (n_m > n * (partial-1) / procs) {
               u = m;  n_u = n_m;
            } else {
               l = m;  n_l = n_m;
            }
            m = (u+l)*0.5f;
         }
         lo = m; n_lo = n_m;
      }
   } else if (getRange(argc, argv, n_up, n_lo)){
      n_l = 0; n_u = n; n_m = -1;
      l = lg; u = ug; m = (lg+ug)*0.5f;
      while(n_m != n_up){
         n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
         if (n_m > n_up) {
            u = m;  n_u = n_m;
         } else {
            l = m;  n_l = n_m;
         }
         m = (u+l)*0.5f;
      }
      up = m;
      n_l = 0; n_u = n; n_m = -1;
      l = lg; u = ug; m = (lg+ug)*0.5f;
      while(n_m != n_lo){
         n_m = min(max(eigenCount(p_a, p_b, n, m),n_l),n_u);
         if (n_m > n_lo) {
            u = m;  n_u = n_m;
         } else {
            l = m;  n_l = n_m;
         }
         m = (u+l)*0.5f;
      }
      lo = m;
   } else {
      n_lo = 0; lo = lg;
      n_up = n; up = ug;
   }

   gettimeofday(&tv_end, NULL);
   time = (tv_end.tv_sec-tv_begin.tv_sec)*1000 + (tv_end.tv_usec-tv_begin.tv_usec)/1000.0;
   cout << "Boundary Time : " << fixed<<setprecision(3) << time << " ms" << endl;
   return;
}
