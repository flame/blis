/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "immintrin.h"
#include "blis.h"

/* One 512-bit AVX register holds 8 DP elements */
typedef union
{
    __m512d v;
    double  d[8] __attribute__((aligned(64)));
} v8df_t;

/**
 * daxpbyv kernel performs the axpbyv operation.
 * y := beta * y + alpha * conjx(x)
 * where,
 *      x & y are double precision vectors of length n.
 *      alpha & beta are scalars.
 */
void bli_daxpbyv_zen_int_avx512
     (
       conj_t           conjx,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

    // Redirecting to other L1 kernels based on alpha and beta values
    // If alpha is 0, we call DSCALV
    // This kernel would further reroute based on few other combinations
    // of alpha and beta. They are as follows :
    // When alpha = 0 :
    //   When beta = 0 --> DSETV
    //   When beta = 1 --> Early return
    //   When beta = !( 0 or 1 ) --> DSCALV
    if ( bli_deq0( *alpha ) )
    {
      bli_dscalv_zen_int10
      (
        BLIS_NO_CONJUGATE,
        n,
        beta,
        y, incy,
        cntx
      );

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
      return;
    }

    // If beta is 0, we call DSCAL2V
    // This kernel would further reroute based on few other combinations
    // of alpha and beta. They are as follows :
    // When beta = 0 :
    //   When alpha = 0 --> DSETV
    //   When alpha = 1 --> DCOPYV
    //   When alpha = !( 0 or 1 ) --> DSCAL2V
    else if ( bli_deq0( *beta ) )
    {
      bli_dscal2v_zen_int
      (
        conjx,
        n,
        alpha,
        x, incx,
        y, incy,
        cntx
      );

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
      return;
    }

    // If beta is 1, we have 2 scenarios for rerouting
    //   When alpha = 1 --> DADDV
    //   When alpha = !( 0 or 1 ) --> DAXPYV
    else if ( bli_deq1( *beta ) )
    {
      if( bli_deq1( *alpha ) )
      {
        bli_daddv_zen_int
        (
          conjx,
          n,
          x, incx,
          y, incy,
          cntx
        );
      }
      else
      {
        bli_daxpyv_zen_int
        (
          conjx,
          n,
          alpha,
          x, incx,
          y, incy,
          cntx
        );
      }

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
      return;
    }

    const dim_t n_elem_per_reg  = 8;    // number of elements per register

    dim_t i = 0;          // iterator

    // Local pointer aliases to the parameters
    double* restrict x0;
    double* restrict y0;

    // Registers to load/store the vectors
    v8df_t alphav;
    v8df_t betav;
    v8df_t yv[8];

    // Boolean to check for alpha being 1
    bool is_alpha_one = bli_seq1( *alpha );

    // Initialize local pointers
    x0 = x;
    y0 = y;

  if( incx == 1 && incy == 1 )
  {
    // Broadcasting beta onto a ZMM register
    betav.v = _mm512_set1_pd( *beta );

    if( is_alpha_one ) // Scale y with beta and add x to it
    {
      for( ; i + 63 < n; i += 64 )
      {
        // Loading Y vector onto 8 registers
        // Thus, we iterate in blocks of 64 elements
        yv[0].v = _mm512_loadu_pd( x0 + 0 * n_elem_per_reg );
        yv[1].v = _mm512_loadu_pd( x0 + 1 * n_elem_per_reg );
        yv[2].v = _mm512_loadu_pd( x0 + 2 * n_elem_per_reg );
        yv[3].v = _mm512_loadu_pd( x0 + 3 * n_elem_per_reg );
        yv[4].v = _mm512_loadu_pd( x0 + 4 * n_elem_per_reg );
        yv[5].v = _mm512_loadu_pd( x0 + 5 * n_elem_per_reg );
        yv[6].v = _mm512_loadu_pd( x0 + 6 * n_elem_per_reg );
        yv[7].v = _mm512_loadu_pd( x0 + 7 * n_elem_per_reg );

        // Loading Y vector and using it as part of beta scaling and adding to X
        yv[0].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 0 * n_elem_per_reg ), yv[0].v );
        yv[1].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 1 * n_elem_per_reg ), yv[1].v );
        yv[2].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 2 * n_elem_per_reg ), yv[2].v );
        yv[3].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 3 * n_elem_per_reg ), yv[3].v );
        yv[4].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 4 * n_elem_per_reg ), yv[4].v );
        yv[5].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 5 * n_elem_per_reg ), yv[5].v );
        yv[6].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 6 * n_elem_per_reg ), yv[6].v );
        yv[7].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 7 * n_elem_per_reg ), yv[7].v );

        // Storing the results onto Y vector
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );
        _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, yv[1].v );
        _mm512_storeu_pd( y0 + 2 * n_elem_per_reg, yv[2].v );
        _mm512_storeu_pd( y0 + 3 * n_elem_per_reg, yv[3].v );
        _mm512_storeu_pd( y0 + 4 * n_elem_per_reg, yv[4].v );
        _mm512_storeu_pd( y0 + 5 * n_elem_per_reg, yv[5].v );
        _mm512_storeu_pd( y0 + 6 * n_elem_per_reg, yv[6].v );
        _mm512_storeu_pd( y0 + 7 * n_elem_per_reg, yv[7].v );

        // Adjusting the pointers
        x0 += 8 * n_elem_per_reg;
        y0 += 8 * n_elem_per_reg;
      }

      for( ; i + 31 < n; i += 32 )
      {
        // Loading Y vector onto 4 registers
        // Thus, we iterate in blocks of 32 elements
        yv[0].v = _mm512_loadu_pd( x0 + 0 * n_elem_per_reg );
        yv[1].v = _mm512_loadu_pd( x0 + 1 * n_elem_per_reg );
        yv[2].v = _mm512_loadu_pd( x0 + 2 * n_elem_per_reg );
        yv[3].v = _mm512_loadu_pd( x0 + 3 * n_elem_per_reg );

        // Loading Y vector and using it as part of beta scaling and adding to X
        yv[0].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 0 * n_elem_per_reg ), yv[0].v );
        yv[1].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 1 * n_elem_per_reg ), yv[1].v );
        yv[2].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 2 * n_elem_per_reg ), yv[2].v );
        yv[3].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 3 * n_elem_per_reg ), yv[3].v );

        // Storing the results onto Y vector
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );
        _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, yv[1].v );
        _mm512_storeu_pd( y0 + 2 * n_elem_per_reg, yv[2].v );
        _mm512_storeu_pd( y0 + 3 * n_elem_per_reg, yv[3].v );

        // Adjusting the pointers
        x0 += 4 * n_elem_per_reg;
        y0 += 4 * n_elem_per_reg;
      }

      for( ; i + 15 < n; i += 16 )
      {
        // Loading Y vector onto 2 registers
        // Thus, we iterate in blocks of 16 elements
        yv[0].v = _mm512_loadu_pd( x0 + 0 * n_elem_per_reg );
        yv[1].v = _mm512_loadu_pd( x0 + 1 * n_elem_per_reg );

        // Loading Y vector and using it as part of beta scaling and adding to X
        yv[0].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 0 * n_elem_per_reg ), yv[0].v );
        yv[1].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 1 * n_elem_per_reg ), yv[1].v );

        // Storing the results onto Y vector
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );
        _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, yv[1].v );

        // Adjusting the pointers
        x0 += 2 * n_elem_per_reg;
        y0 += 2 * n_elem_per_reg;
      }

      for( ; i + 7 < n; i += 8 )
      {
        // Loading Y vector onto 1 register
        // Thus, we iterate in blocks of 8 elements
        yv[0].v = _mm512_loadu_pd( x0 + 0 * n_elem_per_reg );

        // Loading Y vector and using it as part of beta scaling and adding to X
        yv[0].v = _mm512_fmadd_pd( betav.v, _mm512_loadu_pd( y0 + 0 * n_elem_per_reg ), yv[0].v );

        // Storing the results onto Y vector
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );

        // Adjusting the pointers
        x0 += 1 * n_elem_per_reg;
        y0 += 1 * n_elem_per_reg;
      }

      // Handling the fringe cases
      if( i < n )
      {
        // Setting the mask for loading and storing the vectors
        __mmask8 n_mask = (1 << (n - i)) - 1;

        // Loading the X vector
        yv[0].v = _mm512_maskz_loadu_pd( n_mask, x0 + 0 * n_elem_per_reg );

        // Loading Y vector and using it as part of beta scaling and adding to X
        yv[0].v = _mm512_fmadd_pd( betav.v, _mm512_maskz_loadu_pd( n_mask, y0 + 0 * n_elem_per_reg ), yv[0].v );

        // Storing the results onto Y vector
        _mm512_mask_storeu_pd( y0 + 0 * n_elem_per_reg, n_mask, yv[0].v );

      }
    }
    else
    {
      // Broadcasting alpha onto a ZMM register
      alphav.v = _mm512_set1_pd( *alpha );
      for( ; i + 63 < n; i += 64 )
      {
        // Loading X vector onto 8 registers
        // Thus, we iterate in blocks of 64 elements
        yv[0].v = _mm512_loadu_pd( y0 + 0 * n_elem_per_reg );
        yv[1].v = _mm512_loadu_pd( y0 + 1 * n_elem_per_reg );
        yv[2].v = _mm512_loadu_pd( y0 + 2 * n_elem_per_reg );
        yv[3].v = _mm512_loadu_pd( y0 + 3 * n_elem_per_reg );
        yv[4].v = _mm512_loadu_pd( y0 + 4 * n_elem_per_reg );
        yv[5].v = _mm512_loadu_pd( y0 + 5 * n_elem_per_reg );
        yv[6].v = _mm512_loadu_pd( y0 + 6 * n_elem_per_reg );
        yv[7].v = _mm512_loadu_pd( y0 + 7 * n_elem_per_reg );

        // Beta scaling Y vector
        yv[0].v = _mm512_mul_pd( betav.v, yv[0].v );
        yv[1].v = _mm512_mul_pd( betav.v, yv[1].v );
        yv[2].v = _mm512_mul_pd( betav.v, yv[2].v );
        yv[3].v = _mm512_mul_pd( betav.v, yv[3].v );
        yv[4].v = _mm512_mul_pd( betav.v, yv[4].v );
        yv[5].v = _mm512_mul_pd( betav.v, yv[5].v );
        yv[6].v = _mm512_mul_pd( betav.v, yv[6].v );
        yv[7].v = _mm512_mul_pd( betav.v, yv[7].v );

        // Loading X vector and using it as part of alpha scaling and adding to Y
        yv[0].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 0 * n_elem_per_reg ), yv[0].v );
        yv[1].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 1 * n_elem_per_reg ), yv[1].v );
        yv[2].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 2 * n_elem_per_reg ), yv[2].v );
        yv[3].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 3 * n_elem_per_reg ), yv[3].v );
        yv[4].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 4 * n_elem_per_reg ), yv[4].v );
        yv[5].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 5 * n_elem_per_reg ), yv[5].v );
        yv[6].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 6 * n_elem_per_reg ), yv[6].v );
        yv[7].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 7 * n_elem_per_reg ), yv[7].v );

        // Storing the result onto Y
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );
        _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, yv[1].v );
        _mm512_storeu_pd( y0 + 2 * n_elem_per_reg, yv[2].v );
        _mm512_storeu_pd( y0 + 3 * n_elem_per_reg, yv[3].v );
        _mm512_storeu_pd( y0 + 4 * n_elem_per_reg, yv[4].v );
        _mm512_storeu_pd( y0 + 5 * n_elem_per_reg, yv[5].v );
        _mm512_storeu_pd( y0 + 6 * n_elem_per_reg, yv[6].v );
        _mm512_storeu_pd( y0 + 7 * n_elem_per_reg, yv[7].v );

        // Adjusting the pointers
        x0 += 8 * n_elem_per_reg;
        y0 += 8 * n_elem_per_reg;
      }

      for( ; i + 31 < n; i += 32 )
      {
        // Loading X vector onto 4 registers
        // Thus, we iterate in blocks of 32 elements
        yv[0].v = _mm512_loadu_pd( y0 + 0 * n_elem_per_reg );
        yv[1].v = _mm512_loadu_pd( y0 + 1 * n_elem_per_reg );
        yv[2].v = _mm512_loadu_pd( y0 + 2 * n_elem_per_reg );
        yv[3].v = _mm512_loadu_pd( y0 + 3 * n_elem_per_reg );

        // Beta scaling Y vector
        yv[0].v = _mm512_mul_pd( betav.v, yv[0].v );
        yv[1].v = _mm512_mul_pd( betav.v, yv[1].v );
        yv[2].v = _mm512_mul_pd( betav.v, yv[2].v );
        yv[3].v = _mm512_mul_pd( betav.v, yv[3].v );

        // Loading X vector and using it as part of alpha scaling and adding to Y
        yv[0].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 0 * n_elem_per_reg ), yv[0].v );
        yv[1].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 1 * n_elem_per_reg ), yv[1].v );
        yv[2].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 2 * n_elem_per_reg ), yv[2].v );
        yv[3].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 3 * n_elem_per_reg ), yv[3].v );

        // Storing the result onto Y
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );
        _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, yv[1].v );
        _mm512_storeu_pd( y0 + 2 * n_elem_per_reg, yv[2].v );
        _mm512_storeu_pd( y0 + 3 * n_elem_per_reg, yv[3].v );

        // Adjusting the pointers
        x0 += 4 * n_elem_per_reg;
        y0 += 4 * n_elem_per_reg;
      }

      for( ; i + 15 < n; i += 16 )
      {
        // Loading X vector onto 2 registers
        // Thus, we iterate in blocks of 16 elements
        yv[0].v = _mm512_loadu_pd( y0 + 0 * n_elem_per_reg );
        yv[1].v = _mm512_loadu_pd( y0 + 1 * n_elem_per_reg );

        // Beta scaling Y vector
        yv[0].v = _mm512_mul_pd( betav.v, yv[0].v );
        yv[1].v = _mm512_mul_pd( betav.v, yv[1].v );

        // Loading X vector and using it as part of alpha scaling and adding to Y
        yv[0].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 0 * n_elem_per_reg ), yv[0].v );
        yv[1].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 1 * n_elem_per_reg ), yv[1].v );

        // Storing the result onto Y
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );
        _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, yv[1].v );

        // Adjusting the pointers
        x0 += 2 * n_elem_per_reg;
        y0 += 2 * n_elem_per_reg;
      }

      for( ; i + 7 < n; i += 8 )
      {
        // Loading X vector onto 1 register
        // Thus, we iterate in blocks of 8 elements
        yv[0].v = _mm512_loadu_pd( y0 + 0 * n_elem_per_reg );

        // Beta scaling Y vector
        yv[0].v = _mm512_mul_pd( betav.v, yv[0].v );

        // Loading X vector and using it as part of alpha scaling and adding to Y
        yv[0].v = _mm512_fmadd_pd( alphav.v, _mm512_loadu_pd( x0 + 0 * n_elem_per_reg ), yv[0].v );

        // Storing the result onto Y
        _mm512_storeu_pd( y0 + 0 * n_elem_per_reg, yv[0].v );

        // Adjusting the pointers
        x0 += 1 * n_elem_per_reg;
        y0 += 1 * n_elem_per_reg;
      }

      // Handling the fringe cases
      if( i < n )
      {
        // Setting the mask to load/store the remaining elements
        __mmask8 n_mask = (1 << (n - i)) - 1;

        // Loading Y vector
        yv[0].v = _mm512_maskz_loadu_pd( n_mask, y0 + 0 * n_elem_per_reg );

        // Beta scaling Y vector
        yv[0].v = _mm512_mul_pd( betav.v, yv[0].v );

        // Loading X vector and using it as part of alpha scaling and adding to Y
        yv[0].v = _mm512_fmadd_pd( alphav.v, _mm512_maskz_loadu_pd( n_mask, x0 + 0 * n_elem_per_reg ), yv[0].v );

        // Storing the result onto Y
        _mm512_mask_storeu_pd( y0 + 0 * n_elem_per_reg, n_mask, yv[0].v );

      }
    }
  }
  else
  {
    if( is_alpha_one )
    {
      for ( ; i < n; ++i )
      {
        *y0 = (*beta) * (*y0) + (*x0);

        x0 += incx;
        y0 += incy;
      }
    }
    else
    {
      for ( ; i < n; ++i )
      {
        *y0 = (*beta) * (*y0) + (*alpha) * (*x0);

        x0 += incx;
        y0 += incy;
      }
    }
  }

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}
