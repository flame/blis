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
    /* Memory pool declarations for packing vector X. */\
    mem_t  mem_bufX;\
    rntm_t rntm;\
    ctype* x_buf = x;\
    inc_t  buf_incx = incx;\
    /*
     Boolean to check if the y has been packed
     and memory needs to be freed in the end
    */\
    bool is_x_temp_buf_created = FALSE;\
\
    bli_set_dims_incs_with_trans( transa, \
                                  m, n, rs_a, cs_a, \
                                  &n_iter, &n_elem, &rs_at, &cs_at ); \
\
    conja = bli_extract_conj( transa ); \
\
    PASTECH(ch,dotxf_ker_ft) kfp_df; \
\
    if( incx > 1 )\
    {\
        /*
        Initialize mem pool buffer to NULL and size to 0
        "buf" and "size" fields are assigned once memory
        is allocated from the pool in bli_pba_acquire_m().
        This will ensure bli_mem_is_alloc() will be passed on
        an allocated memory if created or a NULL .
        */\
        mem_bufX.pblk.buf = NULL;\
        mem_bufX.pblk.block_size = 0;\
        mem_bufX.buf_type = 0;\
        mem_bufX.size = 0;\
        mem_bufX.pool = NULL;\
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
          and save the associated mem_t entry to mem_bufX.
        */\
        bli_pba_acquire_m(&rntm, buffer_size, BLIS_BUFFER_FOR_B_PANEL, &mem_bufX);\
\
        /*
         Continue packing X if buffer memory is allocated
        */\
        if ((bli_mem_is_alloc(&mem_bufX)))\
        {\
            x_buf = bli_mem_buffer(&mem_bufX);\
            buf_incx = 1;\
            ctype* alpha_passed = PASTEMAC(ch,1);\
\
            PASTECH(ch,scal2v_ker_ft) scal2v_kr_ptr;\
\
            scal2v_kr_ptr = bli_cntx_get_l1v_ker_dt(dt, BLIS_SCAL2V_KER, cntx);\
\
            /*
              Invoke the ZSCAL2V function using the function pointer
            */\
            scal2v_kr_ptr\
            (\
              BLIS_NO_CONJUGATE,\
              n_elem,\
              alpha_passed,\
              x, incx,\
              x_buf, buf_incx,\
              cntx\
            );\
\
            /*
              Set x is packed as the memory allocation was
              successful and contents have been copied
            */\
            is_x_temp_buf_created = TRUE;\
        }\
    }\
\
    /* Query the context for the kernel function pointer and fusing factor. */ \
    kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx ); \
    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx ); \
\
    for ( i = 0; i < n_iter; i += f ) \
    { \
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse ); \
\
        A1 = a + (i  )*rs_at + (0  )*cs_at; \
        x1 = x_buf + (0  )*buf_incx; \
        y1 = y + (i  )*incy; \
\
        /* y1 = beta * y1 + alpha * A1 * x; */ \
        kfp_df \
        ( \
          conja, \
          conjx, \
          n_elem, \
          f, \
          alpha, \
          A1,   cs_at, rs_at, \
          x1,   buf_incx, \
          beta, \
          y1,   incy, \
          cntx  \
        ); \
\
    } \
    /*
      Check if temp y buffer was used for compute
    */\
    if (is_x_temp_buf_created)\
    {\
        /*
          Return the buffer to pool
        */\
        bli_pba_release(&rntm, &mem_bufX);\
    }\
}

INSERT_GENTFUNC_BASIC0( gemv_unf_var1 )

