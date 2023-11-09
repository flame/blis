/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#define BLIS_DGEMV_VAR2_FUSE 4

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
\
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3); \
\
    bli_init_once(); \
\
    if(cntx == NULL) cntx = bli_gks_query_cntx(); \
\
    const num_t dt = PASTEMAC(ch,type); \
\
    ctype*  A1; \
    ctype*  x1; \
    ctype*  y1; \
    dim_t   i; \
    dim_t   b_fuse, f; \
    dim_t   n_elem, n_iter; \
    inc_t   rs_at, cs_at; \
    conj_t  conja; \
\
    bli_set_dims_incs_with_trans( transa, \
                                  m, n, rs_a, cs_a, \
                                  &n_elem, &n_iter, &rs_at, &cs_at ); \
\
    conja = bli_extract_conj( transa ); \
\
    /*
        Memory pool declarations for packing vector Y.
    */\
    mem_t mem_bufY;\
    rntm_t rntm;\
    ctype* y_buf = y;\
    inc_t buf_incy = incy;\
\
    /*
        Boolean to check if the y has been packed
        and memory needs to be freed in the end
    */\
    bool is_y_temp_buf_created = FALSE;\
\
    /*
        If alpha is equal to zero, y = beta * y + alpha * A * x
        becomes y = beat * y in that case packing will be costly.
        y is only scaled with SCALV and returned.
    */\
    if (incy > 1 && (!PASTEMAC(ch,eq0)( *alpha )))\
    {\
        /*
            Initialize mem pool buffer to NULL and size to 0
            "buf" and "size" fields are assigned once memory
            is allocated from the pool in bli_pba_acquire_m().
            This will ensure bli_mem_is_alloc() will be passed on
            an allocated memory if created or a NULL .
        */\
        mem_bufY.pblk.buf = NULL;\
        mem_bufY.pblk.block_size = 0;\
        mem_bufY.buf_type = 0;\
        mem_bufY.size = 0;\
        mem_bufY.pool = NULL;\
\
        /*
        In order to get the buffer from pool via rntm access to memory broker
        is needed.Following are initializations for rntm
        */\
\
        bli_rntm_init_from_global(&rntm);\
        bli_rntm_set_num_threads_only(1, &rntm);\
        bli_pba_rntm_set_pba(&rntm);\
\
        /*
            Calculate the size required for n_elem double elements in vector Y.
        */\
        size_t buffer_size = n_elem * sizeof(ctype);\
\
        /*
            Acquire a Buffer(n_elem*size(double)) from the memory broker
            and save the associated mem_t entry to mem_bufY.
        */\
        bli_pba_acquire_m(&rntm,\
                            buffer_size,\
                            BLIS_BUFFER_FOR_B_PANEL,\
                            &mem_bufY);\
\
        /*
            Continue packing Y if buffer memory is allocated
        */\
        if ((bli_mem_is_alloc(&mem_bufY)))\
        {\
            y_buf = bli_mem_buffer(&mem_bufY);\
            buf_incy = 1;\
            PASTECH(ch,scal2v_ker_ft) scal2v_kr_ptr;\
            scal2v_kr_ptr = bli_cntx_get_l1v_ker_dt( dt, BLIS_SCAL2V_KER, cntx );\
\
            /*
                Invoke the SCAL2V function using the function pointer
            */\
            scal2v_kr_ptr\
            (\
                BLIS_NO_CONJUGATE,\
                n_elem,\
                beta,\
                y, incy,\
                y_buf, buf_incy,\
                cntx\
            );\
\
            /*
                Set y is packed as the memory allocation was
                successful and contents have been copied
            */\
            is_y_temp_buf_created = TRUE;\
        }\
    }\
    else\
    {\
        /*
            Invoke the SCALV function using the function pointer
        */\
        PASTECH(ch,scalv_ker_ft) scalv_kr_ptr;\
        scalv_kr_ptr = bli_cntx_get_l1v_ker_dt(dt, BLIS_SCALV_KER, cntx);\
\
        scalv_kr_ptr\
        (\
            BLIS_NO_CONJUGATE,\
            n_elem,\
            beta,\
            y_buf, buf_incy,\
            cntx\
        );\
    }\
\
    /*
        If alpha is zero(0), we only need to scalv y and return
    */\
    if (PASTEMAC(ch,eq0)( *alpha ))\
    {\
        /*
            Return early for alpha is zero(0)
        */\
        return;\
    }\
\
    PASTECH(ch,axpyf_ker_ft) kfp_af; \
\
    /* Query the context for the kernel function pointer and fusing factor. */ \
    kfp_af = bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx ); \
    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_AF, cntx ); \
\
    for ( i = 0; i < n_iter; i += f ) \
    { \
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse ); \
\
        A1 = a + (0  )*rs_at + (i  )*cs_at; \
        x1 = x + (i  )*incx; \
        y1 = y_buf + (0  )* buf_incy; \
\
        /* y = y + alpha * A1 * x1; */ \
        kfp_af \
        ( \
          conja, \
          conjx, \
          n_elem, \
          f, \
          alpha, \
          A1, rs_at, cs_at, \
          x1, incx, \
          y1, buf_incy, \
          cntx  \
        ); \
    } \
\
    /*
        Check if temp y buffer was used for compute
    */\
    if (is_y_temp_buf_created)\
    {\
        /*
            Store the result from unit strided y_buf to non-unit strided Y
            Invoke the COPYV function using the function pointer
        */\
        PASTECH(ch,copyv_ker_ft) copyv_kr_ptr;\
        copyv_kr_ptr = bli_cntx_get_l1v_ker_dt(dt, BLIS_COPYV_KER, cntx);\
\
        copyv_kr_ptr\
        (\
            BLIS_NO_CONJUGATE,\
            n_elem,\
            y_buf, buf_incy,\
            y, incy,\
            cntx\
        );\
\
        /* Return the buffer to pool */\
        bli_pba_release(&rntm, &mem_bufY);\
  }\
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3); \
}

INSERT_GENTFUNC_BASIC0( gemv_unf_var2 )
