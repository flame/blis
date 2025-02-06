/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#define D_MR_ 24
#define D_NR_ 8
#define Z_MR_ 12
#define Z_NR_ 4

/*
   declaration of trsm small kernels function pointer
*/
typedef err_t (*trsmsmall_ker_ft)
    (
      obj_t*   AlphaObj,
      obj_t*   a,
      obj_t*   b,
      cntx_t*  cntx,
      cntl_t*  cntl
    );


/*
    Order of kernels in table for each datatype[s/c/d/z]
    {   index,       kernel
        0 : 0b000   LLN[N\U],
        1 : 0b001   LLT[N\U],
        2 : 0b010   LUN[N\U],
        3 : 0b011   LUT[N\U],
        4 : 0b100   RLN[N\U],
        5 : 0b101   RLT[N\U],
        6 : 0b110   RUN[N\U],
        7 : 0b111   RUT[N\U],
    }
*/
#define DATATYPES 4
#define VARIANTS  8
trsmsmall_ker_ft ker_fps_zen5[DATATYPES][VARIANTS] =
  {
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
    {bli_dtrsm_small_AutXB_AlXB_ZEN5,
     bli_dtrsm_small_AltXB_AuXB_ZEN5,
     bli_dtrsm_small_AltXB_AuXB_ZEN5,
     bli_dtrsm_small_AutXB_AlXB_ZEN5,
     bli_dtrsm_small_XAutB_XAlB_ZEN5,
     bli_dtrsm_small_XAltB_XAuB_ZEN5,
     bli_dtrsm_small_XAltB_XAuB_ZEN5,
     bli_dtrsm_small_XAutB_XAlB_ZEN5},
    {bli_ztrsm_small_AutXB_AlXB_ZEN5,
     bli_ztrsm_small_AltXB_AuXB_ZEN5,
     bli_ztrsm_small_AltXB_AuXB_ZEN5,
     bli_ztrsm_small_AutXB_AlXB_ZEN5,
     bli_ztrsm_small_XAutB_XAlB_ZEN5,
     bli_ztrsm_small_XAltB_XAuB_ZEN5,
     bli_ztrsm_small_XAltB_XAuB_ZEN5,
     bli_ztrsm_small_XAutB_XAlB_ZEN5},
};

err_t bli_trsm_small_ZEN5
     (
       side_t   side,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl,
       bool     is_parallel
     )
{
    err_t err;
    bool uplo   = bli_obj_is_upper(a);
    bool transa = bli_obj_has_trans(a);
    num_t dt    = bli_obj_dt(a);

    if (dt == BLIS_SCOMPLEX || dt == BLIS_FLOAT)
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    /* If alpha is zero, B matrix is set to zero */
    if (bli_obj_equals(alpha, &BLIS_ZERO))
    {
        return BLIS_NOT_YET_IMPLEMENTED; // scale B by alpha
    }

    // only col major is supported
    if ((bli_obj_row_stride(a) != 1) ||
        (bli_obj_row_stride(b) != 1))
    {
        return BLIS_INVALID_ROW_STRIDE;
    }

    // A is expected to be triangular in trsm
    if (!bli_obj_is_upper_or_lower(a))
    {
        return BLIS_EXPECTED_TRIANGULAR_OBJECT;
    }

    /*
     *  Compose kernel index based on inputs
     *  3 least significant bits of keridx are used to find index of kernel.
     *  Set least significant bit if transa is true,
     *  Set 2nd least significant bit of keridx if uplo == 'U'
     *  Set 3rd least significant bit of Side =='R'
     *    0b[side == 'R' ? 1: 0][uplo == 'U' ? 1 : 0][transa == 'T' ? 1 : 0]
     *  Example: for RUT[N/U]      0b111
     *           for LLT[N/U]      0b001
     */
    dim_t keridx = (((side & 0x1) << 2) |
                    ((uplo & 0x1) << 1) |
                    (transa & 0x1));

    trsmsmall_ker_ft ker_fp = ker_fps_zen5[dt][keridx];
    /*Call the kernel*/
    err = ker_fp
          (
            alpha,
            a,
            b,
            cntx,
            cntl
           );
    return err;
}

#ifdef BLIS_ENABLE_OPENMP
/*
 * Parallelized dtrsm_small across m-dimension or n-dimension based on side(Left/Right)
 */
err_t bli_trsm_small_mt_ZEN5
     (
       side_t   side,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl,
       bool     is_parallel
     )
{
    gint_t m = bli_obj_length(b); // number of rows of matrix b
    gint_t n = bli_obj_width(b);  // number of columns of Matrix b
    dim_t d_mr, d_nr;

    num_t dt = bli_obj_dt(a);
    switch (dt)
    {
        case BLIS_DOUBLE:
        {
            d_mr = D_MR_, d_nr = D_NR_;
            break;
        }
        case BLIS_DCOMPLEX:
        {
            d_mr = Z_MR_, d_nr = Z_NR_;
            break;
        }
        default:
        {
            return BLIS_NOT_YET_IMPLEMENTED;
            break;
        }
    }
    rntm_t rntm;
    bli_rntm_init_from_global(&rntm);
#ifdef AOCL_DYNAMIC
    // If dynamic-threading is enabled, calculate optimum number
    //  of threads.
    //  rntm will be updated with optimum number of threads.
    if (bli_obj_is_double(b) || bli_obj_is_dcomplex(b) )
    {
        bli_nthreads_optimum(a, b, b, BLIS_TRSM, &rntm);
    }
#endif
    // Query the total number of threads from the rntm_t object.
    dim_t n_threads = bli_rntm_num_threads(&rntm);
    if (n_threads < 0)
        n_threads = 1;
    err_t status = BLIS_SUCCESS;
    _Pragma("omp parallel num_threads(n_threads)")
    {
        // Query the thread's id from OpenMP.
        const dim_t tid = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();

        // if num threads requested and num thread available
        // is not same then use single thread small
        if(nt_real != n_threads)
        {
            if(tid == 0)
            {
                bli_trsm_small_ZEN5
                (
                side,
                alpha,
                a,
                b,
                cntx,
                cntl,
                is_parallel
                );
            }
        }
        else
        {
            obj_t b_t;
            dim_t start; // Each thread start Index
            dim_t end;   // Each thread end Index
            thrinfo_t thread;

            thread.n_way = n_threads;
            thread.work_id = tid;
            thread.ocomm_id = tid;

            // Compute start and end indexes of matrix partitioning for each thread
            if (bli_is_right(side))
            {
                bli_thread_range_sub
                (
                  &thread,
                  m,
                  d_mr, // Need to decide based on type
                  FALSE,
                  &start,
                  &end
                );
                // For each thread acquire matrix block on which they operate
                // Data-based parallelism

                bli_acquire_mpart_mdim(BLIS_FWD, BLIS_SUBPART1, start, end - start, b, &b_t);
            }
            else
            {
                bli_thread_range_sub
                (
                  &thread,
                  n,
                  d_nr,// Need to decide based on type
                  FALSE,
                  &start,
                  &end
                );
                // For each thread acquire matrix block on which they operate
                // Data-based parallelism

                bli_acquire_mpart_ndim(BLIS_FWD, BLIS_SUBPART1, start, end - start, b, &b_t);
            }

            // Parallelism is only across m-dimension/n-dimension - therefore matrix a is common to
            // all threads
            err_t status_l = BLIS_SUCCESS;

            status_l = bli_trsm_small_ZEN5
              (
                side,
                alpha,
                a,
                &b_t,
                NULL,
                NULL,
                is_parallel
              );
            // To capture the error populated from any of the threads
            if ( status_l != BLIS_SUCCESS )
            {
                _Pragma("omp critical")
                status = (status != BLIS_NOT_YET_IMPLEMENTED) ? status_l : status;
            }
        }
    }
    return status;
}
#endif

