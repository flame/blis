/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_types.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_utils.h"
#include "lpgemm_pack_f32.h"

// Kernel function prototypes
typedef void (*lpgemm_rowvar_f32)
     (
       const dim_t,
       const dim_t,
       const dim_t,
       const float*,
       const dim_t,
       const dim_t,
       const dim_t,
       const float*,
       const dim_t,
       const dim_t,
       float*,
       const dim_t,
       const dim_t,
       const float,
       const float,
       lpgemm_post_op*,
       lpgemm_post_op_attr
     );

#ifdef BLIS_KERNELS_ZEN4
LPGEMV_TINY(float, float, float, f32f32f32of32)
{
    const float* a_use = ( float* )a;
    inc_t rs_a_use = rs_a;
    inc_t cs_a_use = cs_a;

    float* b_use = ( float* )b;
    inc_t rs_b_use = rs_b;
    inc_t cs_b_use = cs_b;

    lpgemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type = c_downscale;
    if (c_downscale < F32) post_ops_attr.buf_downscale = c;
    else  post_ops_attr.buf_downscale = NULL;

    if(n == 1)
    {
        float* pack_a_buffer_f32f32f32of32 = NULL;
        float* pack_b_buffer_f32f32f32of32 = NULL;
        err_t err = BLIS_SUCCESS;

        //TODO: AVX2 support need to be added
        // Increased MR from 6 to 16 to make use of 32 ZMM registers
        dim_t MR = 16;

        // Pack B matrix if rs_b > 1
        if( ( mtag_b == PACK ) && ( rs_b != 1 ) )
        {
            siz_t mem_b_size_req = sizeof( float ) * k;
            pack_b_buffer_f32f32f32of32 =
                ( float* )bli_malloc_user(mem_b_size_req, &err);

            for( dim_t k0 = 0; k0 < k; k0++ )
            {
                pack_b_buffer_f32f32f32of32[k0] = b[ k0*rs_b ];
            }

            b_use = pack_b_buffer_f32f32f32of32;
            rs_b_use = 1;
            cs_b_use = 1;
        }

        if( ( mtag_a == PACK ) && ( cs_a != 1 ) )
        {
            siz_t mem_a_size_req = sizeof(float) * m * k;
            pack_a_buffer_f32f32f32of32 =
                ( float* )bli_malloc_user(mem_a_size_req, &err);

            packa_mr16_f32f32f32of32_col_major
            (
              pack_a_buffer_f32f32f32of32,
              a_use, rs_a, cs_a,
              m, k,
              &rs_a_use, &cs_a_use
            );
            a_use = pack_a_buffer_f32f32f32of32;
        }

        post_ops_attr.post_op_c_i = 0;
        post_ops_attr.post_op_c_j = 0;
        lpgemv_n_one_f32f32f32of32
        (
          m, k,
          a_use, rs_a_use, cs_a_use, mtag_a,
          b_use, rs_b_use, cs_b_use, mtag_b,
          c, rs_c, cs_c,
          alpha, beta,
          MR, k,
          post_op_list,
          &post_ops_attr
        );

        if ( pack_a_buffer_f32f32f32of32 != NULL )
        {
            bli_free_user( pack_a_buffer_f32f32f32of32 );
        }
        if ( pack_b_buffer_f32f32f32of32 != NULL )
        {
            bli_free_user( pack_b_buffer_f32f32f32of32 );
        }
    }
}
#endif

LPGEMM_TINY(float,float,float,f32f32f32of32)
{
#ifdef BLIS_KERNELS_ZEN4
  // Handle using LPGEMV when m or/and n equal to 1
  // The avx512 check will be removed when avx2 kernels added in future
  if ( ( n == 1 ) &&
       ( bli_cpuid_is_avx512_supported() == TRUE ) &&
       ( lpgemm_get_enabled_arch() != BLIS_ARCH_ZEN3 ) )
  {
    lpgemv_rowvar_tiny_f32f32f32of32(m, n, k,
                                a, rs_a, cs_a, mtag_a,
                                b, rs_b, cs_b, mtag_b,
                                c, rs_c, cs_c,
                                alpha,
                                beta,
                                lcntx,
                                post_op_list,
                                c_downscale);
    return;
  }
#endif

    const dim_t NR = lcntx->blksz.NR;
    const dim_t MR = lcntx->blksz.MR;

    // Strides are updated based on matrix packing/reordering.
    const float* a_use = NULL;
    dim_t rs_a_use = rs_a;
    dim_t cs_a_use = cs_a;

    const float* b_use = NULL;
    dim_t rs_b_use = rs_b;
    dim_t cs_b_use = cs_b;

    // Pack buffer for B.
    float* pack_b_buffer_f32f32f32of32 = NULL;
    siz_t mem_b_size_req = 0;

    dim_t rs_c_downscale = rs_c;

    inc_t ps_a_use;
    inc_t ps_b_use;

    const dim_t cs_c_use = 1;

    lpgemm_post_op_attr post_ops_attr;
    post_ops_attr.c_stor_type = c_downscale;

    if ( c_downscale < F32 )
    {
        post_ops_attr.buf_downscale = c;
    }
    else
    {
        post_ops_attr.buf_downscale = NULL;
    }

    bool is_first_k = TRUE;
    post_ops_attr.is_first_k = is_first_k;
    bool is_last_k = TRUE;
    post_ops_attr.is_last_k = is_last_k;

    // Even if the mtag_b is set to PACK, for tiny sizes its better to
    // pack only if it affects output accuracy (like column major B),
    // else ignore it.
    if ( ( mtag_b == PACK ) && ( rs_b == 1 ) )
    {
        dim_t nc0_updated = make_multiple_of_n( n, NR );
        mem_b_size_req = sizeof( float ) * nc0_updated * k;

        err_t err = BLIS_SUCCESS;
        pack_b_buffer_f32f32f32of32 =
            ( float* )bli_malloc_user(mem_b_size_req, &err);

        ( ( lpgemm_pack_f32 )lcntx->packb_fun_ptr )
        (
          pack_b_buffer_f32f32f32of32,
          b,
          rs_b, cs_b, n, k, &rs_b_use, &cs_b_use
        );

        rs_b_use = NR;
        cs_b_use = 1;
        ps_b_use = k;

        b_use = pack_b_buffer_f32f32f32of32;
    }
    else if ( mtag_b == REORDERED )
    {
        b_use = b;
        rs_b_use = NR;
        cs_b_use = 1;
        ps_b_use = k;
    }
    else
    {
        b_use = b;
        ps_b_use = 1;
    }

    if ( mtag_a == REORDERED )
    {
        a_use = a;
        rs_a_use = 1;
        cs_a_use = MR;
        ps_a_use = MR * k;
    }
    else
    {
        a_use = a;
        ps_a_use = MR * rs_a;
    }

    for ( dim_t jr = 0; jr < n; jr += NR )
    {
        dim_t nr0 = bli_min( ( n - jr ), NR );

        // Post ops meta attributes.
        post_ops_attr.post_op_c_i = 0;
        post_ops_attr.post_op_c_j = jr;
        post_ops_attr.rs_c_downscale = rs_c_downscale;

        ( ( lpgemm_rowvar_f32 )lcntx->kern_fun_ptr )
        (
          m, nr0, k,
          ( float* )a_use, rs_a_use, cs_a_use, ps_a_use,
          ( float* )( b_use + ( jr * ps_b_use ) ), rs_b_use, cs_b_use,
          ( c + jr ), rs_c, cs_c_use,
          alpha , beta,
          post_op_list, post_ops_attr
        );
    }

    if ( pack_b_buffer_f32f32f32of32 != NULL )
    {
        bli_free_user( pack_b_buffer_f32f32f32of32 );
    }
}
