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

/* Union data structure to access AVX-512 registers
*  One 512-bit AVX register holds 8 DP elements. */
typedef union
{
  __m512d v;
  double  d[8] __attribute__((aligned(64)));
} v8df_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
  __m256d v;
  double  d[4] __attribute__((aligned(64)));
} v4df_t;

/* Union data structure to access AVX registers
*  One 128-bit AVX register holds 2 DP elements. */
typedef union
{
  __m128d v;
  double  d[2] __attribute__((aligned(64)));
} v2df_t;

// -----------------------------------------------------------------------------

void bli_zdotxv_zen_int_avx512
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       dcomplex* restrict beta,
       dcomplex* restrict rho,
       cntx_t* restrict cntx
     )
{
  dim_t i = 0;

  dcomplex* restrict x0;
  dcomplex* restrict y0;
  dcomplex rho0;

  // Performing XOR of conjx and conjy.
  // conj_op is set if either X or Y has conjugate(not both)
  conj_t conj_op = conjx ^ conjy;

  // If beta is zero, initialize rho to zero instead of scaling
  // rho by beta (in case rho contains NaN or Inf).
  if ( PASTEMAC(z,eq0)( *beta ) )
  {
    PASTEMAC(z,set0s)( *rho );
  }
  else
  {
    PASTEMAC(z,scals)( *beta, *rho );
  }

  // If the vector dimension is zero, output rho and return early.
  if ( bli_zero_dim1( n ) || PASTEMAC(z,eq0)( *alpha ) ) return;

  // Initialize local pointers.
  x0 = x;
  y0 = y;

  // Computation to handle unit-stride cases
  if ( incx == 1 && incy == 1 )
  {
    dim_t n_elem_per_reg = 4;

    // Declaring 8 registers, to store partial sums over multiple loads
    // Further declaring 4 registers for loading X and 8 for loading
    // and permuting Y for complex datatype arithmetic.
    v8df_t rhov[8], xv[4], yv[8];

    // Initialize the unrolled iterations' rho vectors to zero.
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();

    rhov[4].v = _mm512_setzero_pd();
    rhov[5].v = _mm512_setzero_pd();
    rhov[6].v = _mm512_setzero_pd();
    rhov[7].v = _mm512_setzero_pd();

    // Setting 2 vectors to 0 and 1 for the compute.
    v8df_t zero_reg, scale_one;
    zero_reg.v = _mm512_setzero_pd();
    scale_one.v = _mm512_set1_pd(1.0);

    // Checking to see if we should take the unmasked vector code
    if( n >= 4 )
    {
      for (; ( i + 15 ) < n; i += 16 )
      {
        // Load elements from X and Y
        xv[0].v = _mm512_loadu_pd((double *) (x0 + 0*n_elem_per_reg) );
        yv[0].v = _mm512_loadu_pd((double *) (y0 + 0*n_elem_per_reg) );

        xv[1].v = _mm512_loadu_pd((double *) (x0 + 1*n_elem_per_reg) );
        yv[1].v = _mm512_loadu_pd((double *) (y0 + 1*n_elem_per_reg) );

        xv[2].v = _mm512_loadu_pd((double *) (x0 + 2*n_elem_per_reg) );
        yv[2].v = _mm512_loadu_pd((double *) (y0 + 2*n_elem_per_reg) );

        xv[3].v = _mm512_loadu_pd((double *) (x0 + 3*n_elem_per_reg) );
        yv[3].v = _mm512_loadu_pd((double *) (y0 + 3*n_elem_per_reg) );

        // Permute to duplicate the imag part for every element
        // yv[4].v = I0 I0 I1 I1 ...
        yv[4].v = _mm512_permute_pd( yv[0].v, 0xFF );
        yv[5].v = _mm512_permute_pd( yv[1].v, 0xFF );
        yv[6].v = _mm512_permute_pd( yv[2].v, 0xFF );
        yv[7].v = _mm512_permute_pd( yv[3].v, 0xFF );

        // Permute to duplicate the real part for every element
        // yv[0].v = R0 R0 R1 R1 ...
        yv[0].v = _mm512_permute_pd( yv[0].v, 0x00 );
        yv[1].v = _mm512_permute_pd( yv[1].v, 0x00 );
        yv[2].v = _mm512_permute_pd( yv[2].v, 0x00 );
        yv[3].v = _mm512_permute_pd( yv[3].v, 0x00 );

        // Compute the element-wise product of the X and Y vectors,
        // storing in the corresponding rho vectors.
        rhov[0].v = _mm512_fmadd_pd( xv[0].v, yv[0].v, rhov[0].v );
        rhov[1].v = _mm512_fmadd_pd( xv[1].v, yv[1].v, rhov[1].v );
        rhov[2].v = _mm512_fmadd_pd( xv[2].v, yv[2].v, rhov[2].v );
        rhov[3].v = _mm512_fmadd_pd( xv[3].v, yv[3].v, rhov[3].v );

        rhov[4].v = _mm512_fmadd_pd( xv[0].v, yv[4].v, rhov[4].v );
        rhov[5].v = _mm512_fmadd_pd( xv[1].v, yv[5].v, rhov[5].v );
        rhov[6].v = _mm512_fmadd_pd( xv[2].v, yv[6].v, rhov[6].v );
        rhov[7].v = _mm512_fmadd_pd( xv[3].v, yv[7].v, rhov[7].v );

        // Adjust the pointers accordingly
        x0 += ( n_elem_per_reg * 4 );
        y0 += ( n_elem_per_reg * 4 );
      }
      for (; ( i + 7 ) < n; i += 8 )
      {
        // Load elements from X and Y
        xv[0].v = _mm512_loadu_pd((double *) (x0 + 0*n_elem_per_reg) );
        yv[0].v = _mm512_loadu_pd((double *) (y0 + 0*n_elem_per_reg) );

        xv[1].v = _mm512_loadu_pd((double *) (x0 + 1*n_elem_per_reg) );
        yv[1].v = _mm512_loadu_pd((double *) (y0 + 1*n_elem_per_reg) );

        // Permute to duplicate the imag part for every element
        // yv[4].v = I0 I0 I1 I1 ...
        yv[4].v = _mm512_permute_pd( yv[0].v, 0xFF );
        yv[5].v = _mm512_permute_pd( yv[1].v, 0xFF );

        // Permute to duplicate the real part for every element
        // yv[0].v = R0 R0 R1 R1 ...
        yv[0].v = _mm512_permute_pd( yv[0].v, 0x00 );
        yv[1].v = _mm512_permute_pd( yv[1].v, 0x00 );

        // Compute the element-wise product of the X and Y vectors,
        // storing in the corresponding rho vectors.
        rhov[0].v = _mm512_fmadd_pd( xv[0].v, yv[0].v, rhov[0].v );
        rhov[1].v = _mm512_fmadd_pd( xv[1].v, yv[1].v, rhov[1].v );

        rhov[4].v = _mm512_fmadd_pd( xv[0].v, yv[4].v, rhov[4].v );
        rhov[5].v = _mm512_fmadd_pd( xv[1].v, yv[5].v, rhov[5].v );

        // Adjust the pointers accordingly
        x0 += ( n_elem_per_reg * 2 );
        y0 += ( n_elem_per_reg * 2 );
      }
      for (; ( i + 3 ) < n; i += 4 )
      {
        // Load elements from X and Y
        xv[0].v = _mm512_loadu_pd((double *) (x0 + 0*n_elem_per_reg) );
        yv[0].v = _mm512_loadu_pd((double *) (y0 + 0*n_elem_per_reg) );

        // Permute to duplicate the imag part for every element
        // yv[4].v = I0 I0 I1 I1 ...
        yv[4].v = _mm512_permute_pd( yv[0].v, 0xFF );

        // Permute to duplicate the real part for every element
        // yv[0].v = R0 R0 R1 R1 ...
        yv[0].v = _mm512_permute_pd( yv[0].v, 0x00 );

        // Compute the element-wise product of the X and Y vectors,
        // storing in the corresponding rho vectors.
        rhov[0].v = _mm512_fmadd_pd( xv[0].v, yv[0].v, rhov[0].v );

        rhov[4].v = _mm512_fmadd_pd( xv[0].v, yv[4].v, rhov[4].v );

        x0 += ( n_elem_per_reg * 1 );
        y0 += ( n_elem_per_reg * 1 );
      }
    }
    if ( i < n )
    {
      // Setting the mask bit based on remaining elements
      // Since each dcomplex elements corresponds to 2 doubles
      // we need to load and store 2*(n-i) elements.
      __mmask8 n_mask = (1 << 2*(n - i)) - 1;

      // Load elements from X and Y
      xv[0].v = _mm512_maskz_loadu_pd(n_mask, (double *)x0 );
      yv[0].v = _mm512_maskz_loadu_pd(n_mask, (double *)y0 );

      // Permute to duplicate the imag part for every element
      // yv[4].v = I0 I0 I1 I1 ...
      yv[4].v = _mm512_permute_pd( yv[0].v, 0xFF );

      // Permute to duplicate the real part for every element
      // yv[0].v = R0 R0 R1 R1 ...
      yv[0].v = _mm512_permute_pd( yv[0].v, 0x00 );

      // Compute the element-wise product of the X and Y vectors,
      // storing in the corresponding rho vectors.
      rhov[0].v = _mm512_fmadd_pd( xv[0].v, yv[0].v, rhov[0].v );

      rhov[4].v = _mm512_fmadd_pd( xv[0].v, yv[4].v, rhov[4].v );
    }

    // Permuting for final accumulation of real and imag parts
    rhov[4].v = _mm512_permute_pd(rhov[4].v, 0x55);
    rhov[5].v = _mm512_permute_pd(rhov[5].v, 0x55);
    rhov[6].v = _mm512_permute_pd(rhov[6].v, 0x55);
    rhov[7].v = _mm512_permute_pd(rhov[7].v, 0x55);

    // Accumulate the unrolled rho vectors into a single vector
    // rhov[0] contains element by element real-part scaling
    // rhov[4] contains element by element imag-part scaling
    rhov[0].v = _mm512_add_pd(rhov[1].v, rhov[0].v);
    rhov[2].v = _mm512_add_pd(rhov[3].v, rhov[2].v);
    rhov[0].v = _mm512_add_pd(rhov[2].v, rhov[0].v);

    rhov[4].v = _mm512_add_pd(rhov[5].v, rhov[4].v);
    rhov[6].v = _mm512_add_pd(rhov[7].v, rhov[6].v);
    rhov[4].v = _mm512_add_pd(rhov[6].v, rhov[4].v);

    /*
      conj_op maps to the compute as follows :
      A = (a + ib), X = (x + iy)
      -----------------------------------------------------------
      |      A       |      X       |  Real part  |  Imag Part  |
      -----------------------------------------------------------
      | No-Conjugate | No-Conjugate |   ax - by	  |   bx + ay   |
      | No-Conjugate |   Conjugate  |   ax + by   |   bx - ay   |
      |   Conjugate  | No-Conjugate |   ax + by   | -(bx - ay)  |
      |   Conjugate  |   Conjugate  |   ax - by   | -(bx + ay)  |
      -----------------------------------------------------------

      If only X or A has conjugate, fmsubadd is performed.
      Else, fmaddsub is performed.

      In the final reduction step, the imaginary part of every
      partial sum is negated if conjat is true
    */

    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm512_fmaddsub_pd(scale_one.v, rhov[0].v, rhov[4].v);
    }
    else
    {
      rhov[0].v = _mm512_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[4].v);
    }

    // Negate the imaginary part if conjy is congutgate
    if ( bli_is_conj( conjx ) )
    {
      rhov[0].v = _mm512_fmsubadd_pd(zero_reg.v, zero_reg.v, rhov[0].v);
    }

    // Intermediate registers for final reduction
    v4df_t inter[2];

    inter[0].v = _mm512_extractf64x4_pd(rhov[0].v, 0x00);
    inter[1].v = _mm512_extractf64x4_pd(rhov[0].v, 0x01);

    inter[0].v = _mm256_add_pd(inter[1].v, inter[0].v);

    // Accumulate the final rho vector into a single scalar result.
    rho0.real = inter[0].d[0] + inter[0].d[2];
    rho0.imag = inter[0].d[1] + inter[0].d[3];

  }
  else
  {
    v2df_t rhov[2], xv, yv[2];

    rhov[0].v = _mm_setzero_pd();
    rhov[1].v = _mm_setzero_pd();

    for(; i < n; i += 1)
    {
      // Load elements from X and Y
      xv.v = _mm_loadu_pd((double *)x0 );
      yv[0].v = _mm_loadu_pd((double *)y0 );

      // Permute to duplicate the imag part for every element
      // yv[1].v = I0 I0
      yv[1].v = _mm_permute_pd( yv[0].v, 0b11 );

      // Permute to duplicate the real part for every element
      // yv[0].v = R0 R0
      yv[0].v = _mm_permute_pd( yv[0].v, 0b00 );

      // Compute the element-wise product of the X and Y vectors,
      // storing in the corresponding rho vectors.
      rhov[0].v = _mm_fmadd_pd( xv.v, yv[0].v, rhov[0].v );

      rhov[1].v = _mm_fmadd_pd( xv.v, yv[1].v, rhov[1].v );

      x0 += incx;
      y0 += incy;
    }

    // Permute for final reduction
    rhov[1].v = _mm_permute_pd(rhov[1].v, 0x01);

    v2df_t zero_reg, scale_one;

    zero_reg.v = _mm_setzero_pd();
    scale_one.v = _mm_set1_pd(1.0);

    if ( bli_is_noconj( conj_op ) )
    {
      rhov[0].v = _mm_addsub_pd(rhov[0].v, rhov[1].v);
    }
    else
    {
      rhov[0].v = _mm_fmsubadd_pd(scale_one.v, rhov[0].v, rhov[1].v);
    }
    if( bli_is_conj( conjx ) )
    {
      rhov[0].v = _mm_fmsubadd_pd(zero_reg.v, rhov[0].v, rhov[0].v);
    }

    rho0.real = rhov[0].d[0];
    rho0.imag = rhov[0].d[1];
  }

  // Accumulate the final result into the output variable.
  PASTEMAC(z,axpys)( *alpha, rho0, *rho );
}
