#include "cu_head.h"

void par_eigenval(float *p_val, float *p_a, float *p_b, U32 n, float lo, float up, U32 n_lo, U32 n_up, float tao)
{
   if (n <= MAX_SMALL_MATRIX)
        small_eigval(p_val, p_a, p_b, n, lo, up, n_lo, n_up, tao);
   else {
        huge_eigval(p_val, p_a, p_b, n, lo, up, n_lo, n_up, tao);
   }
}

