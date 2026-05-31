
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
#define PRAGMA_UNROLL_4 _Pragma("unroll 4")
#elif defined(__GUNC__)
#define PRAGMA_NOUNROLL _Pragma("GCC unroll 1")
#define PRAGMA_UNROLL_2 _Pragma("GCC unroll 2")
#define PRAGMA_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define PRAGMA_NOUNROLL
#define PRAGMA_UNROLL_2
#define PRAGMA_UNROLL_4
#endif


void bli_bpackm_armv8a_int_12x16
     (
            conj_t  conja,
            pack_t  schema,
            dim_t   cdim0,
            dim_t   cdim_max,
            dim_t   cdim_bcast,
            dim_t   k0,
            dim_t   k0_max,
      const void*   kappa,
      const void*   a, inc_t inca0, inc_t lda0,
            void*   p,              inc_t ldp0,
      const void*   params,
      const cntx_t* cntx
     )
{
    // This is the panel dimension assumed by the packm kernel.
    const dim_t    mr     = 12;
    const dim_t    nr     = 16;
    
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t       k_iter = k0 / 8;
    uint64_t       k_left = k0 % 8;
    
    const bfloat16_t*   a_loc  = a;
          bfloat16_t*   p_loc  = p;
    
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
    const bool     unitk  = bli_deq1( *( ( bfloat16_t* ) kappa ) );
    
    
    // -------------------------------------------------------------------------
    if ( cdim0 == mr && cdim_bcast == 1 && !gs )  // packing A
    {
        if ( unitk )
        {
            if ( inca == 1 ) // A is column major
            {
                // No need to use k-loops here.
                // Simple let complier to expand loops.
                PRAGMA_UNROLL_4
                for( dim_t ik = k_iter * 8 + k_left; ik > 0; --ik )
                {
                    bfloat16x8_t v0 = vld1q_bf16( a_loc + 0 );
                    bfloat16x4_t v1 = vld1_bf16( a_loc + 8 );

                    vst1q_bf16( p_loc + 0, v0 );
                    vst1_bf16( p_loc + 8, v1 );

                    a_loc += lda;
                    p_loc += ldp;
                }
            }
            else // if ( lda == 1 )
            {
                printf(" inca = %d, lda = %d\n", k0, 1) ;
                bfloat16x8_t v0 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v1 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v2 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v3 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v4 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v5 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v6 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v7 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v8 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v9 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v10 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t v11 = (bfloat16x8_t)vdupq_n_bf16( 0 );
                bfloat16x8_t vt0;
                bfloat16x8_t vt1;
                bfloat16x8_t vt2;
                bfloat16x8_t vt3;
                bfloat16x8_t vt4;
                bfloat16x8_t vt5;
                bfloat16x8_t vt6;
                bfloat16x8_t vt7;
                bfloat16x8_t vt8;
                bfloat16x8_t vt9;
                bfloat16x8_t vt10;
                bfloat16x8_t vt11;

                bfloat16x4_t hv0;
                bfloat16x4_t hv1;
                bfloat16x4_t hv2;
                bfloat16x4_t hv3;
                bfloat16x4_t hv4;
                bfloat16x4_t hv5;
                bfloat16x4_t hv6;
                bfloat16x4_t hv7;

                PRAGMA_NOUNROLL
                for( ; k_iter > 0; --k_iter)
                {
                    v0  = vld1q_bf16( a_loc + inca * 0 );
                    v1  = vld1q_bf16( a_loc + inca * 1 );
                    v2  = vld1q_bf16( a_loc + inca * 2 );
                    v3  = vld1q_bf16( a_loc + inca * 3 );
                    v4  = vld1q_bf16( a_loc + inca * 4 );
                    v5  = vld1q_bf16( a_loc + inca * 5 );
                    v6  = vld1q_bf16( a_loc + inca * 6 );
                    v7  = vld1q_bf16( a_loc + inca * 7 );
                    v8  = vld1q_bf16( a_loc + inca * 8 );
                    v9  = vld1q_bf16( a_loc + inca * 9 );
                    v10 = vld1q_bf16( a_loc + inca * 10 );
                    v11 = vld1q_bf16( a_loc + inca * 11 );

                    // In-register transpose
                    // 
                    // Column 0-7
                    vt0 = vtrn1q_u16( v0, v1 );
                    vt1 = vtrn2q_u16( v0, v1 );
                    vt2 = vtrn1q_u16( v2, v3 );
                    vt3 = vtrn2q_u16( v2, v3 );
                    vt4 = vtrn1q_u16( v4, v5 );
                    vt5 = vtrn2q_u16( v4, v5 );
                    vt6 = vtrn1q_u16( v6, v7 );
                    vt7 = vtrn2q_u16( v6, v7 );
                    v0  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt0, (float32x4_t)vt2 );
                    v1  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt1, (float32x4_t)vt3 );
                    v2  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt0, (float32x4_t)vt2 );
                    v3  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt1, (float32x4_t)vt3 );
                    v4  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt4, (float32x4_t)vt6 );
                    v5  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt5, (float32x4_t)vt7 );
                    v6  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt4, (float32x4_t)vt6 );
                    v7  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt5, (float32x4_t)vt7 );
                    vt0 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v0, (float64x2_t)v4 );
                    vt1 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v1, (float64x2_t)v5 );
                    vt2 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v2, (float64x2_t)v6 );
                    vt3 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v3, (float64x2_t)v7 );
                    vt4 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v0, (float64x2_t)v4 );
                    vt5 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v1, (float64x2_t)v5 );
                    vt6 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v2, (float64x2_t)v6 );
                    vt7 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v3, (float64x2_t)v7 );
                    // Column 8-12
                    vt8  = vtrn1q_u16( v8, v9 );
                    vt9  = vtrn2q_u16( v8, v9 );
                    vt10 = vtrn1q_u16( v10, v11 );   
                    vt11 = vtrn2q_u16( v10, v11 );
                    v8   = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt8, (float32x4_t)vt10 );
                    v9   = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt9, (float32x4_t)vt11 );
                    v10  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt8, (float32x4_t)vt10 );
                    v11  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt9, (float32x4_t)vt11 );
                    hv0  = vget_low_bf16(v8);
                    hv1  = vget_low_bf16(v9);
                    hv2  = vget_low_bf16(v10);
                    hv3  = vget_low_bf16(v11);
                    hv4  = vget_high_bf16(v8);
                    hv5  = vget_high_bf16(v9);
                    hv6  = vget_high_bf16(v10);
                    hv7  = vget_high_bf16(v11);

                    vst1q_bf16( p_loc + 0, vt0 );
                    vst1_bf16 ( p_loc + 8, hv0 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt1 );
                    vst1_bf16 ( p_loc + 8, hv1 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt2 );
                    vst1_bf16 ( p_loc + 8, hv2 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt3 );
                    vst1_bf16 ( p_loc + 8, hv3 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt4 );
                    vst1_bf16 ( p_loc + 8, hv4 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt5 );
                    vst1_bf16 ( p_loc + 8, hv5 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt6 );
                    vst1_bf16 ( p_loc + 8, hv6 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt7 );
                    vst1_bf16 ( p_loc + 8, hv7 );
                    p_loc += ldp;
                    a_loc += 8 * lda;
                } //  end of k_iter for loop
                for( ; k_left > 0; --k_left )
                {
                    v0 = vld1q_lane_bf16( a_loc + inca * 0, v0, 0);
                    v0 = vld1q_lane_bf16( a_loc + inca * 1, v0, 1);
                    v0 = vld1q_lane_bf16( a_loc + inca * 2, v0, 2);
                    v0 = vld1q_lane_bf16( a_loc + inca * 3, v0, 3);
                    v0 = vld1q_lane_bf16( a_loc + inca * 4, v0, 4);
                    v0 = vld1q_lane_bf16( a_loc + inca * 5, v0, 5);
                    v0 = vld1q_lane_bf16( a_loc + inca * 6, v0, 6);
                    v0 = vld1q_lane_bf16( a_loc + inca * 7, v0, 7);
                    hv0 = vld1_lane_bf16( a_loc + inca * 8, hv0, 0);
                    hv0 = vld1_lane_bf16( a_loc + inca * 9, hv0, 1);
                    hv0 = vld1_lane_bf16( a_loc + inca * 10, hv0, 2);
                    hv0 = vld1_lane_bf16( a_loc + inca * 11, hv0, 3);

                    vst1q_bf16( p_loc + 0,  v0 );
                    vst1_bf16 ( p_loc + 8,  hv0 );
                    p_loc += ldp;
                    a_loc += lda; // 1;
                }
            } // end of  else if ( lda == 1)
        } //  end of unitk
        else // if ( !unitk )
        {
            float32x4_t vkappa = vld1q_dup_f32( (float*)kappa );
            //bfloat16x8_t vkappa = vld1q_dup_bf16( kappa );
            //bfloat16x4_t hvkappa = vld1_dup_bf16( kappa );

            if ( inca == 1 )
            {
                // No need to use k-loops here;
                // Simply let complier to expand loops
                PRAGMA_UNROLL_4
                for( dim_t ik = k_iter * 8 + k_left; ik > 0; --ik)
                {
                    bfloat16x8_t v0 = vld1q_bf16( a_loc + 0 );
                    bfloat16x4_t hv0 = vld1_bf16( a_loc + 8 );

                    // Scale by kappa
                    float32x4_t v0_low  = (float32x4_t)vcvtq_low_f32_bf16(v0);
                    float32x4_t v0_high = (float32x4_t)vcvtq_high_f32_bf16(v0);
                    float32x4_t hv0_f32 = (float32x4_t) vcvt_f32_bf16(hv0);

                    v0_low  = vmulq_f32( v0_low,  vkappa );
                    v0_high = vmulq_f32( v0_high, vkappa );
                    hv0_f32 = vmulq_f32( hv0_f32, vkappa );

                    v0 = vcvtq_low_bf16_f32(v0_low);
                    v0 = vcvtq_high_bf16_f32(v0, v0_high);
                    hv0 = vcvt_bf16_f32(hv0_f32);
                    
                    vst1q_bf16( p_loc + 0, v0  );
                    vst1_bf16 ( p_loc + 8, hv0 );

                    a_loc += lda;
                    p_loc += ldp;
                }
            }
            else // if ( lda == 1 )
            {
                bfloat16x8_t v0 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v1 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v2 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v3 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v4 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v5 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v6 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v7 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v8 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v9 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v10 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v11 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t vt0;
                bfloat16x8_t vt1;
                bfloat16x8_t vt2;
                bfloat16x8_t vt3;
                bfloat16x8_t vt4;
                bfloat16x8_t vt5;
                bfloat16x8_t vt6;
                bfloat16x8_t vt7;
                bfloat16x8_t vt8;
                bfloat16x8_t vt9;
                bfloat16x8_t vt10;
                bfloat16x8_t vt11;

                bfloat16x4_t hv0;
                bfloat16x4_t hv1;
                bfloat16x4_t hv2;
                bfloat16x4_t hv3;
                bfloat16x4_t hv4;
                bfloat16x4_t hv5;
                bfloat16x4_t hv6;
                bfloat16x4_t hv7;

                PRAGMA_NOUNROLL
                for( ; k_iter > 0; --k_iter)
                {
                    v0 = vld1q_bf16( a_loc + inca * 0 );
                    v1 = vld1q_bf16( a_loc + inca * 1 );
                    v2 = vld1q_bf16( a_loc + inca * 2 );
                    v3 = vld1q_bf16( a_loc + inca * 3 );
                    v4 = vld1q_bf16( a_loc + inca * 4 );
                    v5 = vld1q_bf16( a_loc + inca * 5 );
                    v6 = vld1q_bf16( a_loc + inca * 6 );
                    v7 = vld1q_bf16( a_loc + inca * 7 );
                    v8 = vld1q_bf16( a_loc + inca * 8 );
                    v9 = vld1q_bf16( a_loc + inca * 9 );
                    v10 = vld1q_bf16( a_loc + inca * 10 );
                    v11 = vld1q_bf16( a_loc + inca * 11 );

                    // Scale by kappa
                    float32x4_t v0_low  = (float32x4_t)vcvtq_low_f32_bf16( v0);
                    float32x4_t v0_high = (float32x4_t)vcvtq_high_f32_bf16(v0);
                    float32x4_t v1_low  = (float32x4_t)vcvtq_low_f32_bf16( v1);
                    float32x4_t v1_high = (float32x4_t)vcvtq_high_f32_bf16(v1);
                    float32x4_t v2_low  = (float32x4_t)vcvtq_low_f32_bf16( v2);
                    float32x4_t v2_high = (float32x4_t)vcvtq_high_f32_bf16(v2);
                    float32x4_t v3_low  = (float32x4_t)vcvtq_low_f32_bf16( v3);
                    float32x4_t v3_high = (float32x4_t)vcvtq_high_f32_bf16(v3);
                    float32x4_t v4_low  = (float32x4_t)vcvtq_low_f32_bf16( v4);
                    float32x4_t v4_high = (float32x4_t)vcvtq_high_f32_bf16(v4);
                    float32x4_t v5_low  = (float32x4_t)vcvtq_low_f32_bf16( v5);
                    float32x4_t v5_high = (float32x4_t)vcvtq_high_f32_bf16(v5);
                    float32x4_t v6_low  = (float32x4_t)vcvtq_low_f32_bf16( v6);
                    float32x4_t v6_high = (float32x4_t)vcvtq_high_f32_bf16(v6);
                    float32x4_t v7_low  = (float32x4_t)vcvtq_low_f32_bf16( v7);
                    float32x4_t v7_high = (float32x4_t)vcvtq_high_f32_bf16(v7);
                    float32x4_t v8_low  = (float32x4_t)vcvtq_low_f32_bf16( v8);
                    float32x4_t v8_high = (float32x4_t)vcvtq_high_f32_bf16(v8);
                    float32x4_t v9_low  = (float32x4_t)vcvtq_low_f32_bf16( v9);
                    float32x4_t v9_high = (float32x4_t)vcvtq_high_f32_bf16(v9);
                    float32x4_t v10_low  = (float32x4_t)vcvtq_low_f32_bf16( v10);
                    float32x4_t v10_high = (float32x4_t)vcvtq_high_f32_bf16(v10);
                    float32x4_t v11_low  = (float32x4_t)vcvtq_low_f32_bf16( v11);
                    float32x4_t v11_high = (float32x4_t)vcvtq_high_f32_bf16(v11);
                    v0_low  = vmulq_f32( v0_low,  vkappa );
                    v0_high = vmulq_f32( v0_high, vkappa );
                    v1_low  = vmulq_f32( v1_low,  vkappa );
                    v1_high = vmulq_f32( v1_high, vkappa );
                    v2_low  = vmulq_f32( v2_low,  vkappa );
                    v2_high = vmulq_f32( v2_high, vkappa );
                    v3_low  = vmulq_f32( v3_low,  vkappa );
                    v3_high = vmulq_f32( v3_high, vkappa );
                    v4_low  = vmulq_f32( v4_low,  vkappa );
                    v4_high = vmulq_f32( v4_high, vkappa );
                    v5_low  = vmulq_f32( v5_low,  vkappa );
                    v5_high = vmulq_f32( v5_high, vkappa );
                    v6_low  = vmulq_f32( v6_low,  vkappa );
                    v6_high = vmulq_f32( v6_high, vkappa );
                    v7_low  = vmulq_f32( v7_low,  vkappa );
                    v7_high = vmulq_f32( v7_high, vkappa );
                    v8_low  = vmulq_f32( v8_low,  vkappa );
                    v8_high = vmulq_f32( v8_high, vkappa );
                    v9_low  = vmulq_f32( v9_low,  vkappa );
                    v9_high = vmulq_f32( v9_high, vkappa );
                    v10_low  = vmulq_f32( v10_low,  vkappa );
                    v10_high = vmulq_f32( v10_high, vkappa );
                    v11_low  = vmulq_f32( v11_low,  vkappa );
                    v11_high = vmulq_f32( v11_high, vkappa );
                    v0 = vcvtq_low_bf16_f32(v0_low);
                    v0 = vcvtq_high_bf16_f32(v0, v0_high);
                    v1 = vcvtq_low_bf16_f32(v1_low);
                    v1 = vcvtq_high_bf16_f32(v1, v1_high);
                    v2 = vcvtq_low_bf16_f32(v2_low);
                    v2 = vcvtq_high_bf16_f32(v2, v2_high);
                    v3 = vcvtq_low_bf16_f32(v3_low);
                    v3 = vcvtq_high_bf16_f32(v3, v3_high);
                    v4 = vcvtq_low_bf16_f32(v4_low);
                    v4 = vcvtq_high_bf16_f32(v4, v4_high);
                    v5 = vcvtq_low_bf16_f32(v5_low);
                    v5 = vcvtq_high_bf16_f32(v5, v5_high);
                    v6 = vcvtq_low_bf16_f32(v6_low);
                    v6 = vcvtq_high_bf16_f32(v6, v6_high);
                    v7 = vcvtq_low_bf16_f32(v7_low);
                    v7 = vcvtq_high_bf16_f32(v7, v7_high);
                    v8 = vcvtq_low_bf16_f32(v8_low);
                    v8 = vcvtq_high_bf16_f32(v8, v8_high);
                    v9 = vcvtq_low_bf16_f32(v9_low);
                    v9 = vcvtq_high_bf16_f32(v9, v9_high);
                    v10 = vcvtq_low_bf16_f32(v10_low);
                    v10 = vcvtq_high_bf16_f32(v10, v10_high);
                    v11 = vcvtq_low_bf16_f32(v11_low);
                    v11 = vcvtq_high_bf16_f32(v11, v11_high);

                    //v0 = vmulq_bf16( v0, vkappa );
                    //v1 = vmulq_bf16( v1, vkappa );
                    //v2 = vmulq_bf16( v2, vkappa );
                    //v3 = vmulq_bf16( v3, vkappa );
                    //v4 = vmulq_bf16( v4, vkappa );
                    //v5 = vmulq_bf16( v5, vkappa );
                    //v6 = vmulq_bf16( v6, vkappa );
                    //v7 = vmulq_bf16( v7, vkappa );
                    //v8 = vmulq_bf16( v8, vkappa );
                    //v9 = vmulq_bf16( v9, vkappa );
                    //v10 = vmulq_bf16( v10, vkappa );
                    //v11 = vmulq_bf16( v11, vkappa );

                    // In-register transpose
                    // 
                    // Column 0-7
                    vt0 = vtrn1q_u16( v0, v1 );
                    vt1 = vtrn2q_u16( v0, v1 );
                    vt2 = vtrn1q_u16( v2, v3 );
                    vt3 = vtrn2q_u16( v2, v3 );
                    vt4 = vtrn1q_u16( v4, v5 );
                    vt5 = vtrn2q_u16( v4, v5 );
                    vt6 = vtrn1q_u16( v6, v7 );
                    vt7 = vtrn2q_u16( v6, v7 );
                    v0 = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt0, (float32x4_t)vt2 );
                    v1 = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt1, (float32x4_t)vt3 );
                    v2 = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt0, (float32x4_t)vt2 );
                    v3 = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt1, (float32x4_t)vt3 );
                    v4 = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt4, (float32x4_t)vt6 );
                    v5 = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt5, (float32x4_t)vt7 );
                    v6 = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt4, (float32x4_t)vt6 );
                    v7 = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt5, (float32x4_t)vt7 );
                    vt0 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v0, (float64x2_t)v4 );
                    vt1 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v1, (float64x2_t)v5 );
                    vt2 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v2, (float64x2_t)v6 );
                    vt3 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v3, (float64x2_t)v7 );
                    vt4 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v0, (float64x2_t)v4 );
                    vt5 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v1, (float64x2_t)v5 );
                    vt6 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v2, (float64x2_t)v6 );
                    vt7 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v3, (float64x2_t)v7 );
                    // Column 8-12
                    vt8 = vtrn1q_u16( v8, v9 );
                    vt9 = vtrn2q_u16( v8, v9 );
                    vt10 = vtrn1q_u16( v10, v11 );   
                    vt11 = vtrn2q_u16( v10, v11 );
                    v8 = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt8, (float32x4_t)vt10 );
                    v9 = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt9, (float32x4_t)vt11 );
                    v10 = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt8, (float32x4_t)vt10 );
                    v11 = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt9, (float32x4_t)vt11 );
                    hv0  = vget_low_bf16(v8);
                    hv1  = vget_low_bf16(v9);
                    hv2  = vget_low_bf16(v10);
                    hv3  = vget_low_bf16(v11);
                    hv4  = vget_high_bf16(v8);
                    hv5  = vget_high_bf16(v9);
                    hv6  = vget_high_bf16(v10);
                    hv7  = vget_high_bf16(v11);


                    vst1q_bf16( p_loc + 0, vt0 );
                    vst1_bf16 ( p_loc + 8, hv0 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt1 );
                    vst1_bf16 ( p_loc + 8, hv1 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt2 );
                    vst1_bf16 ( p_loc + 8, hv2 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt3 );
                    vst1_bf16 ( p_loc + 8, hv3 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt4 );
                    vst1_bf16 ( p_loc + 8, hv4 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt5 );
                    vst1_bf16 ( p_loc + 8, hv5 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt6 );
                    vst1_bf16 ( p_loc + 8, hv6 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt7 );
                    vst1_bf16 ( p_loc + 8, hv7 );
                    p_loc += ldp;
                    a_loc += 8 * lda;
                } //  end of k_iter for loop
                for( ; k_left > 0; --k_left )
                {
                    v0 = vld1q_lane_bf16( a_loc + inca * 0, v0, 0);
                    v0 = vld1q_lane_bf16( a_loc + inca * 1, v0, 1);
                    v0 = vld1q_lane_bf16( a_loc + inca * 2, v0, 2);
                    v0 = vld1q_lane_bf16( a_loc + inca * 3, v0, 3);
                    v0 = vld1q_lane_bf16( a_loc + inca * 4, v0, 4);
                    v0 = vld1q_lane_bf16( a_loc + inca * 5, v0, 5);
                    v0 = vld1q_lane_bf16( a_loc + inca * 6, v0, 6);
                    v0 = vld1q_lane_bf16( a_loc + inca * 7, v0, 7);
                    hv0 = vld1_lane_bf16( a_loc + inca * 8, hv0, 0);
                    hv0 = vld1_lane_bf16( a_loc + inca * 9, hv0, 1);
                    hv0 = vld1_lane_bf16( a_loc + inca * 10, hv0, 2);
                    hv0 = vld1_lane_bf16( a_loc + inca * 11, hv0, 3);

                    // Scale by kappa
                    v0  = vmulq_bf16( v0,  vkappa );
                    hv0 = vmul_bf16( hv0, hvkappa );

                    vst1q_bf16( p_loc + 0,  v0 );
                    vst1_bf16 ( p_loc + 8, hv0 );
                    p_loc += ldp;
                    a_loc += lda; // 1;
                }
            }
        }
    } //  end of if ( cdim0 == mr && cdim_bcast == 1 && !gs )
    else if ( cdim0 == nr && cdim_bcast == 1 && !gs )
    {
        if ( unitk )
        {
            if ( inca == 1 )
            {
                // No need to use k-loops here.
                // Simply let compiler to expand loops
                PRAGMA_UNROLL_4
                for( dim_t ik = k_iter * 8 + k_left; ik > 0; --ik )
                {
                    bfloat16x8_t v0 = vld1q_bf16( a_loc + 0 );
                    bfloat16x8_t v1 = vld1q_bf16( a_loc + 8 );

                    vst1q_bf16( p_loc + 0, v0 );
                    vst1q_bf16( p_loc + 8, v1 );

                    a_loc += lda;
                    p_loc += ldp;
                }
            }
            else // if ( lda == 1 )
            {
                bfloat16x8_t v0 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v1 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v2 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v3 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v4 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v5 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v6 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v7 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v8 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v9 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v10 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v11 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v12 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v13 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v14 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v15 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t vt0;
                bfloat16x8_t vt1;
                bfloat16x8_t vt2;
                bfloat16x8_t vt3;
                bfloat16x8_t vt4;
                bfloat16x8_t vt5;
                bfloat16x8_t vt6;
                bfloat16x8_t vt7;
                bfloat16x8_t vt8;
                bfloat16x8_t vt9;
                bfloat16x8_t vt10;
                bfloat16x8_t vt11;
                bfloat16x8_t vt12;
                bfloat16x8_t vt13;
                bfloat16x8_t vt14;
                bfloat16x8_t vt15;

                PRAGMA_NOUNROLL
                for( ; k_iter > 0; --k_iter )
                {
                    v0  = vld1q_bf16( a_loc + inca * 0  );
                    v1  = vld1q_bf16( a_loc + inca * 1  );
                    v2  = vld1q_bf16( a_loc + inca * 2  );
                    v3  = vld1q_bf16( a_loc + inca * 3  );
                    v4  = vld1q_bf16( a_loc + inca * 4  );
                    v5  = vld1q_bf16( a_loc + inca * 5  );
                    v6  = vld1q_bf16( a_loc + inca * 6  );
                    v7  = vld1q_bf16( a_loc + inca * 7  );
                    v8  = vld1q_bf16( a_loc + inca * 8  );
                    v9  = vld1q_bf16( a_loc + inca * 9  );
                    v10 = vld1q_bf16( a_loc + inca * 10 );
                    v11 = vld1q_bf16( a_loc + inca * 11 );
                    v12 = vld1q_bf16( a_loc + inca * 12 );
                    v13 = vld1q_bf16( a_loc + inca * 13 );
                    v14 = vld1q_bf16( a_loc + inca * 14 );
                    v15 = vld1q_bf16( a_loc + inca * 15 );

                    // In-register transpose.
                    //
                    // Column 0-7
                    vt0 = vtrn1q_u16( v0, v1 );
                    vt1 = vtrn2q_u16( v0, v1 );
                    vt2 = vtrn1q_u16( v2, v3 ); 
                    vt3 = vtrn2q_u16( v2, v3 ); 
                    vt4 = vtrn1q_u16( v4, v5 );
                    vt5 = vtrn2q_u16( v4, v5 );
                    vt6 = vtrn1q_u16( v6, v7 ); 
                    vt7 = vtrn2q_u16( v6, v7 ); 
                    v0  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt0, (float32x4_t)vt2 ); 
                    v1  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt1, (float32x4_t)vt3 ); 
                    v2  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt0, (float32x4_t)vt2 ); 
                    v3  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt1, (float32x4_t)vt3 ); 
                    v4  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt4, (float32x4_t)vt6 ); 
                    v5  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt5, (float32x4_t)vt7 ); 
                    v6  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt4, (float32x4_t)vt6 ); 
                    v7  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt5, (float32x4_t)vt7 ); 
                    vt0 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v0,  (float64x2_t)v4 );
                    vt1 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v1,  (float64x2_t)v5 );
                    vt2 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v2,  (float64x2_t)v6 );
                    vt3 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v3,  (float64x2_t)v7 );
                    vt4 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v0,  (float64x2_t)v4 );
                    vt5 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v1,  (float64x2_t)v5 );
                    vt6 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v2,  (float64x2_t)v6 );
                    vt7 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v3,  (float64x2_t)v7 );

                    // Column 8-15
                    vt8  = vtrn1q_u16(  v8,  v9 );
                    vt9  = vtrn2q_u16(  v8,  v9 );
                    vt10 = vtrn1q_u16( v10, v11 ); 
                    vt11 = vtrn2q_u16( v10, v11 ); 
                    vt12 = vtrn1q_u16( v12, v13 );
                    vt13 = vtrn2q_u16( v12, v13 );
                    vt14 = vtrn1q_u16( v14, v15 ); 
                    vt15 = vtrn2q_u16( v14, v15 ); 
                    v8   = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt8,  (float32x4_t)vt10 ); 
                    v9   = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt9,  (float32x4_t)vt11 ); 
                    v10  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt8,  (float32x4_t)vt10 ); 
                    v11  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt9,  (float32x4_t)vt11 ); 
                    v12  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt12, (float32x4_t)vt14 ); 
                    v13  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt13, (float32x4_t)vt15 ); 
                    v14  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt12, (float32x4_t)vt14 ); 
                    v15  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt13, (float32x4_t)vt15 );
                    vt8  = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v8,   (float64x2_t)v12 );
                    vt9  = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v9,   (float64x2_t)v13 );
                    vt10 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v10,  (float64x2_t)v14 );
                    vt11 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v11,  (float64x2_t)v15 );
                    vt12 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v8,   (float64x2_t)v12 );
                    vt13 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v9,   (float64x2_t)v13 );
                    vt14 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v10,  (float64x2_t)v14 );
                    vt15 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v11,  (float64x2_t)v15 );

                    vst1q_bf16( p_loc + 0, vt0 );
                    vst1q_bf16( p_loc + 8, vt8 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt1 );
                    vst1q_bf16( p_loc + 8, vt9 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt2 );
                    vst1q_bf16( p_loc + 8, vt10 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt3 );
                    vst1q_bf16( p_loc + 8, vt11 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt4 );
                    vst1q_bf16( p_loc + 8, vt12 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt5 );
                    vst1q_bf16( p_loc + 8, vt13 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt6 );
                    vst1q_bf16( p_loc + 8, vt14 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt7 );
                    vst1q_bf16( p_loc + 8, vt15 );
                    p_loc += ldp;
                    a_loc += 8 * lda;
                }
                for( ; k_left > 0; --k_left )
                {
                    v0 = vld1q_lane_bf16( a_loc + inca * 0, v0, 0 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 1, v0, 1 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 2, v0, 2 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 3, v0, 3 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 4, v0, 4 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 5, v0, 5 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 6, v0, 6 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 7, v0, 7 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 8, v1, 0 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 9, v1, 1 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 10, v1, 2 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 11, v1, 3 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 12, v1, 4 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 13, v1, 5 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 14, v1, 6 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 15, v1, 7 );

                    vst1q_bf16( p_loc + 0,  v0 );
                    vst1q_bf16( p_loc + 8,  v1 );
                    p_loc += ldp;
                    a_loc += lda; // 1;
                }
            } // end of lda == 1
        } // end of unitk
        else // if ( !unitk )
        {
            bfloat16x8_t vkappa = vld1q_dup_bf16( kappa );

            if ( inca == 1 )           
            {
                // No need to use k-loops here.
                // Simply let compiler to expand loops
                PRAGMA_UNROLL_4
                for( dim_t ik = k_iter * 8 + k_left; ik > 0; --ik )
                {
                    bfloat16x8_t v0 = vld1q_bf16( a_loc + 0 );
                    bfloat16x8_t v1 = vld1q_bf16( a_loc + 8 );

                    // Scale by kappa
                    v0 = vmulq_bf16( v0, vkappa );
                    v1 = vmulq_bf16( v1, vkappa );

                    vst1q_bf16( p_loc + 0, v0 );
                    vst1q_bf16( p_loc + 8, v1 );

                    a_loc += lda;
                    p_loc += ldp;
                }
            } // end if ( inca == 1 )
            else // if ( lda == 1 )
            {
                bfloat16x8_t v0 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v1 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v2 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v3 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v4 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v5 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v6 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v7 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v8 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v9 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v10 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v11 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v12 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v13 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v14 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t v15 = (bfloat16x8_t)vdupq_n_u16( 0 );
                bfloat16x8_t vt0;
                bfloat16x8_t vt1;
                bfloat16x8_t vt2;
                bfloat16x8_t vt3;
                bfloat16x8_t vt4;
                bfloat16x8_t vt5;
                bfloat16x8_t vt6;
                bfloat16x8_t vt7;
                bfloat16x8_t vt8;
                bfloat16x8_t vt9;
                bfloat16x8_t vt10;
                bfloat16x8_t vt11;
                bfloat16x8_t vt12;
                bfloat16x8_t vt13;
                bfloat16x8_t vt14;
                bfloat16x8_t vt15;

                PRAGMA_NOUNROLL
                for( ; k_iter > 0; --k_iter )
                {
                    v0  = vld1q_bf16( a_loc + inca * 0  );
                    v1  = vld1q_bf16( a_loc + inca * 1  );
                    v2  = vld1q_bf16( a_loc + inca * 2  );
                    v3  = vld1q_bf16( a_loc + inca * 3  );
                    v4  = vld1q_bf16( a_loc + inca * 4  );
                    v5  = vld1q_bf16( a_loc + inca * 5  );
                    v6  = vld1q_bf16( a_loc + inca * 6  );
                    v7  = vld1q_bf16( a_loc + inca * 7  );
                    v8  = vld1q_bf16( a_loc + inca * 8  );
                    v9  = vld1q_bf16( a_loc + inca * 9  );
                    v10 = vld1q_bf16( a_loc + inca * 10 );
                    v11 = vld1q_bf16( a_loc + inca * 11 );
                    v12 = vld1q_bf16( a_loc + inca * 12 );
                    v13 = vld1q_bf16( a_loc + inca * 13 );
                    v14 = vld1q_bf16( a_loc + inca * 14 );
                    v15 = vld1q_bf16( a_loc + inca * 15 );

                    // Scale by kappa
                    v0 = vmulq_bf16( v0, vkappa );
                    v1 = vmulq_bf16( v1, vkappa );
                    v2 = vmulq_bf16( v2, vkappa );
                    v3 = vmulq_bf16( v3, vkappa );
                    v4 = vmulq_bf16( v4, vkappa );
                    v5 = vmulq_bf16( v5, vkappa );
                    v6 = vmulq_bf16( v6, vkappa );
                    v7 = vmulq_bf16( v7, vkappa );
                    v8 = vmulq_bf16( v8, vkappa );
                    v9 = vmulq_bf16( v9, vkappa );
                    v10 = vmulq_bf16( v10, vkappa );
                    v11 = vmulq_bf16( v11, vkappa );
                    v12 = vmulq_bf16( v12, vkappa );
                    v13 = vmulq_bf16( v13, vkappa );
                    v14 = vmulq_bf16( v14, vkappa );
                    v15 = vmulq_bf16( v15, vkappa );

                    // In-register transpose.
                    //
                    // Column 0-7
                    vt0 = vtrn1q_u16( v0, v1 );
                    vt1 = vtrn2q_u16( v0, v1 );
                    vt2 = vtrn1q_u16( v2, v3 ); 
                    vt3 = vtrn2q_u16( v2, v3 ); 
                    vt4 = vtrn1q_u16( v4, v5 );
                    vt5 = vtrn2q_u16( v4, v5 );
                    vt6 = vtrn1q_u16( v6, v7 ); 
                    vt7 = vtrn2q_u16( v6, v7 ); 
                    v0  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt0, (float32x4_t)vt2 ); 
                    v1  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt1, (float32x4_t)vt3 ); 
                    v2  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt0, (float32x4_t)vt2 ); 
                    v3  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt1, (float32x4_t)vt3 ); 
                    v4  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt4, (float32x4_t)vt6 ); 
                    v5  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt5, (float32x4_t)vt7 ); 
                    v6  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt4, (float32x4_t)vt6 ); 
                    v7  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt5, (float32x4_t)vt7 ); 
                    vt0 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v0,  (float64x2_t)v4 );
                    vt1 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v1,  (float64x2_t)v5 );
                    vt2 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v2,  (float64x2_t)v6 );
                    vt3 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v3,  (float64x2_t)v7 );
                    vt4 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v0,  (float64x2_t)v4 );
                    vt5 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v1,  (float64x2_t)v5 );
                    vt6 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v2,  (float64x2_t)v6 );
                    vt7 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v3,  (float64x2_t)v7 );

                    // Column 8-15
                    vt8  = vtrn1q_u16(  v8,  v9 );
                    vt9  = vtrn2q_u16(  v8,  v9 );
                    vt10 = vtrn1q_u16( v10, v11 ); 
                    vt11 = vtrn2q_u16( v10, v11 ); 
                    vt12 = vtrn1q_u16( v12, v13 );
                    vt13 = vtrn2q_u16( v12, v13 );
                    vt14 = vtrn1q_u16( v14, v15 ); 
                    vt15 = vtrn2q_u16( v14, v15 ); 
                    v8   = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt8,  (float32x4_t)vt10 ); 
                    v9   = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt9,  (float32x4_t)vt11 ); 
                    v10  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt8,  (float32x4_t)vt10 ); 
                    v11  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt9,  (float32x4_t)vt11 ); 
                    v12  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt12, (float32x4_t)vt14 ); 
                    v13  = (bfloat16x8_t)vtrn1q_u32( (float32x4_t)vt13, (float32x4_t)vt15 ); 
                    v14  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt12, (float32x4_t)vt14 ); 
                    v15  = (bfloat16x8_t)vtrn2q_u32( (float32x4_t)vt13, (float32x4_t)vt15 );
                    vt8  = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v8,   (float64x2_t)v12  );
                    vt9  = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v9,   (float64x2_t)v13  );
                    vt10 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v10,  (float64x2_t)v14  );
                    vt11 = (bfloat16x8_t)vtrn1q_u64( (float64x2_t)v11,  (float64x2_t)v15  );
                    vt12 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v8,   (float64x2_t)v12  );
                    vt13 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v9,   (float64x2_t)v13  );
                    vt14 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v10,  (float64x2_t)v14  );
                    vt15 = (bfloat16x8_t)vtrn2q_u64( (float64x2_t)v11,  (float64x2_t)v15  );

                    vst1q_bf16( p_loc + 0, vt0 );
                    vst1q_bf16( p_loc + 8, vt8 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt1 );
                    vst1q_bf16( p_loc + 8, vt9 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt2 );
                    vst1q_bf16( p_loc + 8, vt10 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt3 );
                    vst1q_bf16( p_loc + 8, vt11 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt4 );
                    vst1q_bf16( p_loc + 8, vt12 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt5 );
                    vst1q_bf16( p_loc + 8, vt13 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt6 );
                    vst1q_bf16( p_loc + 8, vt14 );
                    p_loc += ldp;

                    vst1q_bf16( p_loc + 0, vt7 );
                    vst1q_bf16( p_loc + 8, vt15 );
                    p_loc += ldp;
                    a_loc += 8 * lda;
                }
                for( ; k_left > 0; --k_left )
                {
                    v0 = vld1q_lane_bf16( a_loc + inca * 0, v0, 0 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 1, v0, 1 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 2, v0, 2 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 3, v0, 3 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 4, v0, 4 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 5, v0, 5 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 6, v0, 6 );
                    v0 = vld1q_lane_bf16( a_loc + inca * 7, v0, 7 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 8, v1, 0 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 9, v1, 1 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 10, v1, 2 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 11, v1, 3 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 12, v1, 4 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 13, v1, 5 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 14, v1, 6 );
                    v1 = vld1q_lane_bf16( a_loc + inca * 15, v1, 7 );

                    // Scale by kappa
                    v0 = vmulq_bf16( v0, vkappa );
                    v1 = vmulq_bf16( v1, vkappa );

                    vst1q_bf16( p_loc + 0,  v0 );
                    vst1q_bf16( p_loc + 8,  v1 );
                    p_loc += ldp;
                    a_loc += lda; // 1;
                }
            } // end of lda == 1
        }
    }
    else
    {
		bli_sscal2bbs_mxn
		(
		    conja,
		    cdim0,
		    k0,
		    kappa,
		    a,       inca, lda,
		    p, cdim_bcast, ldp
		);
    }

	bli_sset0s_edge
	(
	    cdim0*cdim_bcast, cdim_max*cdim_bcast,
	    k0, k0_max,
	    p, ldp
	);


}
