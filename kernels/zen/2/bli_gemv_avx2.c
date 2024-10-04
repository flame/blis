/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * bli_dgemv_n_avx2(...) handles cases where op(A) = NO_TRANSPOSE for Zen/2/3
 * architectures and is based on the previous approach of using the fused
 * kernels, namely AXPYF, to perform the GEMV operation.
 */
void bli_dgemv_n_avx2
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       double* alpha,
       double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx,
       double* beta,
       double* y, inc_t incy,
       cntx_t* cntx
     )
{
    AOCL_DTL_TRACE_ENTRY( AOCL_DTL_LEVEL_TRACE_4 );
    double*  A1;
    double*  x1;
    dim_t   i;
    dim_t   f, b_fuse;
    dim_t   m0, n0;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    // Memory pool declarations for packing vector Y.
    mem_t   mem_bufY;
    rntm_t  rntm;
    double* y_temp    = y;
    inc_t   temp_incy = incy;

    // Boolean to check if y vector is packed and memory needs to be freed.
    bool is_y_temp_buf_created = FALSE;

    // Update dimensions and strides based on op(A).
    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &m0, &n0, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    // Function pointer declaration for the functions that will be used.
    daxpyf_ker_ft  axpyf_kr_ptr;        // DAXPYF
    dscal2v_ker_ft scal2v_kr_ptr;       // DSCAL2V
    dscalv_ker_ft  scalv_kr_ptr;        // DSCALV
    dcopyv_ker_ft  copyv_kr_ptr;        // DCOPYV

    // Setting the fuse factor based on bli_daxpyf_zen_int_8 kernel.
    b_fuse        = 8;
    axpyf_kr_ptr  = bli_daxpyf_zen_int_8;       // DAXPYF
    scal2v_kr_ptr = bli_dscal2v_zen_int;        // DSCAL2V
    scalv_kr_ptr  = bli_dscalv_zen_int10;       // DSCALV
    copyv_kr_ptr  = bli_dcopyv_zen_int;         // DCOPYV

    /*
      If alpha is equal to zero, y is only scaled by beta and returned.
      In this case, packing and unpacking y will be costly and it is
      avoided.
    */
    if ( (incy != 1) && (!bli_deq0( *alpha )))
    {
        /*
          Initialize mem pool buffer to NULL and size to 0
          "buf" and "size" fields are assigned once memory
          is allocated from the pool in bli_pba_acquire_m().
          This will ensure bli_mem_is_alloc() will be passed on
          an allocated memory if created or a NULL .
        */
        mem_bufY.pblk.buf = NULL;   mem_bufY.pblk.block_size = 0;
        mem_bufY.buf_type = 0;      mem_bufY.size = 0;
        mem_bufY.pool = NULL;

        /* In order to get the buffer from pool via rntm access to memory broker
        is needed. Following are initializations for rntm */
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        //calculate the size required for m0 double elements in vector Y.
        size_t buffer_size = m0 * sizeof(double);

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dgemv_n_avx2(): get mem pool block\n" );
        #endif

        /* Acquire a Buffer(m0*size(double)) from the memory broker
        and save the associated mem_t entry to mem_bufY. */
        bli_pba_acquire_m
        (
          &rntm,
          buffer_size,
          BLIS_BUFFER_FOR_B_PANEL,
          &mem_bufY
        );

        /* Continue packing Y if buffer memory is allocated. */
        if ( bli_mem_is_alloc( &mem_bufY ) )
        {
            y_temp = bli_mem_buffer(&mem_bufY);

            // Stride of vector y_temp
            temp_incy = 1;

            // Query the context if it is NULL. This will be necessary for Zen architectures
            if(cntx == NULL) cntx = bli_gks_query_cntx();

            scal2v_kr_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_SCAL2V_KER, cntx);

            // Invoke the SCAL2V function using the function pointer
            scal2v_kr_ptr
            (
              BLIS_NO_CONJUGATE,
              m0,
              beta,
              y, incy,
              y_temp, temp_incy,
              cntx
            );

            /*
              Set y is packed as the memory allocation was successful
              and contents have been scaled and copied to a temp buffer
            */
            is_y_temp_buf_created = TRUE;
        }
    }
    else
    {
        // Invoke the DSCALV function using the function pointer
        scalv_kr_ptr
        (
          BLIS_NO_CONJUGATE,
          m0,
          beta,
          y_temp, temp_incy,
          cntx
        );
    }

    if( bli_deq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
        return;
    }

    for ( i = 0; i < n0; i += f )
    {
        f = bli_determine_blocksize_dim_f(i, n0, b_fuse);

        A1 = a + (i * cs_at);
        x1 = x + (i * incx);

        axpyf_kr_ptr
        (
          conja,
          conjx,
          m0,
          f,
          alpha,
          A1, rs_at, cs_at,
          x1, incx,
          y_temp, temp_incy,
          cntx
        );
    }

    // If y was packed into y_temp, copy the contents back to y and free memory.
    if ( is_y_temp_buf_created )
    {
        // Store the result from unit strided y_buf to non-unit strided Y.
        // Invoke the COPYV function using the function pointer.
        copyv_kr_ptr
        (
          BLIS_NO_CONJUGATE,
          m0,
          y_temp, temp_incy,
          y, incy,
          cntx
        );

#ifdef BLIS_ENABLE_MEM_TRACING
        printf( "bli_dgemv_n_avx2(): releasing mem pool block\n" );
#endif
        // Return the buffer to pool
        bli_pba_release( &rntm , &mem_bufY );
    }

    AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_4 );
}
