/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Linaro Limited

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

#include "blis.h"
#include <arm_neon.h>

#if defined(__clang__)
#define PRAGMA_NOUNROLL _Pragma("nounroll")
#define PRAGMA_UNROLL_2 _Pragma("unroll 2")
#elif defined(__GNUC__)
#define PRAGMA_NOUNROLL _Pragma("GCC unroll 1")
#define PRAGMA_UNROLL_2 _Pragma("GCC unroll 2")
#else
#define PRAGMA_NOUNROLL
#define PRAGMA_UNROLL_2
#endif

void bli_dpackm_armv8a_int_8xk
     (
       conj_t              conja,
       pack_t              schema,
       dim_t               cdim0,
       dim_t               k0,
       dim_t               k0_max,
       double*    restrict kappa,
       double*    restrict a, inc_t inca0, inc_t lda0,
       double*    restrict p,              inc_t ldp0,
       cntx_t*    restrict cntx
     )
{
  // This is the panel dimension assumed by the packm kernel.
  const dim_t    mnr    = 8;

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t       k_iter = k0 / 2;
  uint64_t       k_left = k0 % 2;
  double*        a_loc  = a;
  double*        p_loc  = p;

  // NOTE: For the purposes of the comments in this packm kernel, we
  // interpret inca and lda as rs_a and cs_a, respectively, and similarly
  // interpret ldp as cs_p (with rs_p implicitly unit). Thus, when reading
  // this packm kernel, you should think of the operation as packing an
  // m x n micropanel, where m and n are tiny and large, respectively, and
  // where elements of each column of the packed matrix P are contiguous.
  // (This packm kernel can still be used to pack micropanels of matrix B
  // in a gemm operation.)
  const uint64_t inca   = inca0;
  const uint64_t lda    = lda0;
  const uint64_t ldp    = ldp0;

  const bool     gs     = ( inca0 != 1 && lda0 != 1 );

  // NOTE: If/when this kernel ever supports scaling by kappa within the
  // assembly region, this constraint should be lifted.
  const bool     unitk  = bli_deq1( *kappa );


  // -------------------------------------------------------------------------

  if ( cdim0 == mnr && !gs )
  {
    if ( unitk )
    {
      if ( inca == 1 )
      {
        // No need to use k-loops here.
        // Simply let compiler to expand loops.
        PRAGMA_UNROLL_2
        for ( dim_t ik = k_iter * 2 + k_left; ik > 0; --ik )
        {
          float64x2_t v0 = vld1q_f64( a_loc + 0 );
          float64x2_t v1 = vld1q_f64( a_loc + 2 );
          float64x2_t v2 = vld1q_f64( a_loc + 4 );
          float64x2_t v3 = vld1q_f64( a_loc + 6 );

          vst1q_f64( p_loc + 0, v0 );
          vst1q_f64( p_loc + 2, v1 );
          vst1q_f64( p_loc + 4, v2 );
          vst1q_f64( p_loc + 6, v3 );

          a_loc += lda;
          p_loc += ldp;
        }
      }
      else // if ( lda == 1 )
      {
        float64x2_t v0 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v1 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v2 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v3 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v4 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v5 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v6 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v7 = (float64x2_t)vdupq_n_u64( 0 );

        PRAGMA_NOUNROLL
        for ( ; k_iter > 0; --k_iter )
        {
          v0 = vld1q_f64( a_loc + inca * 0 );
          v1 = vld1q_f64( a_loc + inca * 1 );
          v2 = vld1q_f64( a_loc + inca * 2 );
          v3 = vld1q_f64( a_loc + inca * 3 );
          v4 = vld1q_f64( a_loc + inca * 4 );
          v5 = vld1q_f64( a_loc + inca * 5 );
          v6 = vld1q_f64( a_loc + inca * 6 );
          v7 = vld1q_f64( a_loc + inca * 7 );

          // In-register transpose.
          float64x2_t vd0_1 = vtrn1q_f64( v0, v1 );
          float64x2_t vd1_1 = vtrn1q_f64( v2, v3 );
          float64x2_t vd2_1 = vtrn1q_f64( v4, v5 );
          float64x2_t vd3_1 = vtrn1q_f64( v6, v7 );
          float64x2_t vd0_2 = vtrn2q_f64( v0, v1 );
          float64x2_t vd1_2 = vtrn2q_f64( v2, v3 );
          float64x2_t vd2_2 = vtrn2q_f64( v4, v5 );
          float64x2_t vd3_2 = vtrn2q_f64( v6, v7 );

          vst1q_f64( p_loc + 0, vd0_1 );
          vst1q_f64( p_loc + 2, vd1_1 );
          vst1q_f64( p_loc + 4, vd2_1 );
          vst1q_f64( p_loc + 6, vd3_1 );
          p_loc += ldp;

          vst1q_f64( p_loc + 0, vd0_2 );
          vst1q_f64( p_loc + 2, vd1_2 );
          vst1q_f64( p_loc + 4, vd2_2 );
          vst1q_f64( p_loc + 6, vd3_2 );
          p_loc += ldp;
          a_loc += 2 * lda; // 2;
        }
        for ( ; k_left > 0; --k_left )
        {
          v0 = vld1q_lane_f64( a_loc + inca * 0, v0, 0 );
          v0 = vld1q_lane_f64( a_loc + inca * 1, v0, 1 );
          v1 = vld1q_lane_f64( a_loc + inca * 2, v1, 0 );
          v1 = vld1q_lane_f64( a_loc + inca * 3, v1, 1 );
          v2 = vld1q_lane_f64( a_loc + inca * 4, v2, 0 );
          v2 = vld1q_lane_f64( a_loc + inca * 5, v2, 1 );
          v3 = vld1q_lane_f64( a_loc + inca * 6, v3, 0 );
          v3 = vld1q_lane_f64( a_loc + inca * 7, v3, 1 );

          vst1q_f64( p_loc + 0, v0 );
          vst1q_f64( p_loc + 2, v1 );
          vst1q_f64( p_loc + 4, v2 );
          vst1q_f64( p_loc + 6, v3 );
          p_loc += ldp;
          a_loc += lda; // 1;
        }
      }
    }
    else // if ( !unitk )
    {
      float64x2_t vkappa = vld1q_dup_f64( kappa );

      if ( inca == 1 )
      {
        // No need to use k-loops here.
        // Simply let compiler to expand loops.
        PRAGMA_UNROLL_2
        for ( dim_t ik = k_iter * 2 + k_left; ik > 0; --ik )
        {
          float64x2_t v0 = vld1q_f64( a_loc + 0 );
          float64x2_t v1 = vld1q_f64( a_loc + 2 );
          float64x2_t v2 = vld1q_f64( a_loc + 4 );
          float64x2_t v3 = vld1q_f64( a_loc + 6 );

          // Scale by kappa.
          v0 = vmulq_f64( v0, vkappa );
          v1 = vmulq_f64( v1, vkappa );
          v2 = vmulq_f64( v2, vkappa );
          v3 = vmulq_f64( v3, vkappa );

          vst1q_f64( p_loc + 0, v0 );
          vst1q_f64( p_loc + 2, v1 );
          vst1q_f64( p_loc + 4, v2 );
          vst1q_f64( p_loc + 6, v3 );

          a_loc += lda;
          p_loc += ldp;
        }
      }
      else // if ( lda == 1 )
      {
        float64x2_t v0 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v1 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v2 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v3 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v4 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v5 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v6 = (float64x2_t)vdupq_n_u64( 0 );
        float64x2_t v7 = (float64x2_t)vdupq_n_u64( 0 );

        PRAGMA_NOUNROLL
        for ( ; k_iter > 0; --k_iter )
        {
          v0 = vld1q_f64( a_loc + inca * 0 );
          v1 = vld1q_f64( a_loc + inca * 1 );
          v2 = vld1q_f64( a_loc + inca * 2 );
          v3 = vld1q_f64( a_loc + inca * 3 );
          v4 = vld1q_f64( a_loc + inca * 4 );
          v5 = vld1q_f64( a_loc + inca * 5 );
          v6 = vld1q_f64( a_loc + inca * 6 );
          v7 = vld1q_f64( a_loc + inca * 7 );

          // Scale by kappa.
          v0 = vmulq_f64( v0, vkappa );
          v1 = vmulq_f64( v1, vkappa );
          v2 = vmulq_f64( v2, vkappa );
          v3 = vmulq_f64( v3, vkappa );
          v4 = vmulq_f64( v4, vkappa );
          v5 = vmulq_f64( v5, vkappa );
          v6 = vmulq_f64( v6, vkappa );
          v7 = vmulq_f64( v7, vkappa );

          // In-register transpose.
          float64x2_t vd0_1 = vtrn1q_f64( v0, v1 );
          float64x2_t vd1_1 = vtrn1q_f64( v2, v3 );
          float64x2_t vd2_1 = vtrn1q_f64( v4, v5 );
          float64x2_t vd3_1 = vtrn1q_f64( v6, v7 );
          float64x2_t vd0_2 = vtrn2q_f64( v0, v1 );
          float64x2_t vd1_2 = vtrn2q_f64( v2, v3 );
          float64x2_t vd2_2 = vtrn2q_f64( v4, v5 );
          float64x2_t vd3_2 = vtrn2q_f64( v6, v7 );

          vst1q_f64( p_loc + 0, vd0_1 );
          vst1q_f64( p_loc + 2, vd1_1 );
          vst1q_f64( p_loc + 4, vd2_1 );
          vst1q_f64( p_loc + 6, vd3_1 );
          p_loc += ldp;

          vst1q_f64( p_loc + 0, vd0_2 );
          vst1q_f64( p_loc + 2, vd1_2 );
          vst1q_f64( p_loc + 4, vd2_2 );
          vst1q_f64( p_loc + 6, vd3_2 );
          p_loc += ldp;
          a_loc += 2 * lda; // 2;
        }
        for ( ; k_left > 0; --k_left )
        {
          v0 = vld1q_lane_f64( a_loc + inca * 0, v0, 0 );
          v0 = vld1q_lane_f64( a_loc + inca * 1, v0, 1 );
          v1 = vld1q_lane_f64( a_loc + inca * 2, v1, 0 );
          v1 = vld1q_lane_f64( a_loc + inca * 3, v1, 1 );
          v2 = vld1q_lane_f64( a_loc + inca * 4, v2, 0 );
          v2 = vld1q_lane_f64( a_loc + inca * 5, v2, 1 );
          v3 = vld1q_lane_f64( a_loc + inca * 6, v3, 0 );
          v3 = vld1q_lane_f64( a_loc + inca * 7, v3, 1 );

          // Scale by kappa.
          v0 = vmulq_f64( v0, vkappa );
          v1 = vmulq_f64( v1, vkappa );
          v2 = vmulq_f64( v2, vkappa );
          v3 = vmulq_f64( v3, vkappa );

          vst1q_f64( p_loc + 0, v0 );
          vst1q_f64( p_loc + 2, v1 );
          vst1q_f64( p_loc + 4, v2 );
          vst1q_f64( p_loc + 6, v3 );
          p_loc += ldp;
          a_loc += lda; // 1;
        }
      }
    }
  }
  else // if ( cdim0 < mnr || gs )
  {
    PASTEMAC(dscal2m,BLIS_TAPI_EX_SUF)
    (
      0,
      BLIS_NONUNIT_DIAG,
      BLIS_DENSE,
      ( trans_t )conja,
      cdim0,
      k0,
      kappa,
      a, inca0, lda0,
      p,     1, ldp0,
      cntx,
      NULL
    );

    if ( cdim0 < mnr )
    {
      // Handle zero-filling along the "long" edge of the micropanel.

      const dim_t      i      = cdim0;
      const dim_t      m_edge = mnr - cdim0;
      const dim_t      n_edge = k0_max;
      double* restrict p_edge = p + (i  )*1;

      bli_dset0s_mxn
      (
        m_edge,
        n_edge,
        p_edge, 1, ldp 
      );
    }
  }

//bli_dfprintm( stdout, "packm 8xk ker: a_packed", cdim0, k0_max, p, 1, ldp0, "%5.2f", "" );

  if ( k0 < k0_max )
  {
    // Handle zero-filling along the "short" (far) edge of the micropanel.

    const dim_t      j      = k0;
    const dim_t      m_edge = mnr;
    const dim_t      n_edge = k0_max - k0;
    double* restrict p_edge = p + (j  )*ldp;

    bli_dset0s_mxn
    (
      m_edge,
      n_edge,
      p_edge, 1, ldp 
    );
  }
}

