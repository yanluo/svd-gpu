#ifndef __EIGEN_HPP__
#define __EIGEN_HPP__

#include "common.h"
#include <iostream>
#include <cmath>

class worktable
{
  public:
    float lo;
    float up;
    U32 n_lo;
    U32 n_up;

    void init(float l, float u, U32 nl, U32 nu){
        lo = l;
        up = u;
        n_lo = nl;
        n_up = nu;
    }
    void print(){
        std::cout << "left = " << lo << " right = " << up;
        std::cout << " NO = " << n_lo << " " << n_up << std::endl;
    }
    float getDiff()   { return abs(up - lo); }
    float getMid()    { return (up + lo) * 0.5f; }
    U32 getEigNum() { return n_up - n_lo; }
    
};


void ser_eigenval(float *eigval, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao);
U32 eigenCount(float *p_a, float *p_b, U32 n, float upper);
void ser_eigenMat(float *p_eigMat, float *p_a, float *p_b, U32 n, float *p_val);
void ldl(float *p_l, float *p_d, float *p_a, float *p_b, U32 n, float val);
void udu(float *p_u, float *p_d, float *p_a, float *p_b, U32 n, float val);
U32 best_index(float *p_du, float *p_dl, float *p_l, U32 n);
void ser_eigvec(float *p_vec, float *p_u, float *p_l, U32 n, U32 index);
void scaled(float *p_vec, U32 n);



#endif
