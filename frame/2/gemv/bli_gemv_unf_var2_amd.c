/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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
    ctype*  zero       = PASTEMAC(ch,0); \
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
      If alpha is equal to zero, y is only scaled by beta and returned.
      In this case, packing and unpacking y will be costly and it is
      avoided.
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

void bli_dgemv_unf_var2
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       double*  alpha,
       double*  a, inc_t rs_a, inc_t cs_a,
       double*  x, inc_t incx,
       double*  beta,
       double*  y, inc_t incy,
       cntx_t* cntx
     )
{

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
    double*  A1;
    double*  x1;
    dim_t   i;
    dim_t   f, b_fuse;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    // Memory pool declarations for packing vector Y.
    mem_t   mem_bufY;
    rntm_t  rntm;
    double* y_temp = y;
    inc_t   temp_incy = incy;

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_elem, &n_iter, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    /*
      Fatbinary config amdzen when run on non-AMD X86 will query for
      the support of AVX512 or AVX2, if AVX512 - arch_id will be zen4
      or for AVX2 it will be zen3.
    */
    arch_t id = bli_arch_query_id();

    /*
      Function pointer declaration for the functions
      that will be used by this API
    */
    daxpyf_ker_ft   axpyf_kr_ptr; // DAXPYF
    dscal2v_ker_ft  scal2v_kr_ptr; // DSCAL2V
    dscalv_ker_ft   scalv_kr_ptr; // DSCALV
    dcopyv_ker_ft   copyv_kr_ptr; // DCOPYV

    /*
      Boolean to check if the y has been packed
      and memory needs to be freed in the end
    */
    bool is_y_temp_buf_created = FALSE;

    switch (id)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
        /*
          Assign the AVX512 based kernel function pointers for
          AXPYF, SCALV, COPYV and corresponding fusing
          factor of DAXPYF kernel
        */

        axpyf_kr_ptr = bli_daxpyf_zen_int_avx512;
        b_fuse = 32;

        scalv_kr_ptr = bli_dscalv_zen_int_avx512;

        copyv_kr_ptr = bli_dcopyv_zen_int;

        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

        /*
          Assign the AVX2 based kernel function pointers for
          AXPYF, SCALV, COPYV and corresponding fusing
          factor of DAXPYF kernel
        */

        axpyf_kr_ptr = bli_daxpyf_zen_int_8;
        b_fuse = 8;

        scalv_kr_ptr = bli_dscalv_zen_int10;

        copyv_kr_ptr = bli_dcopyv_zen_int;

        break;
      default:
        // For non-Zen architectures, query the context if it is NULL
        if(cntx == NULL) cntx = bli_gks_query_cntx();

        /*
          Query the context for the kernel function pointers for
          AXPYF, SCALV, COPYV and corresponding fusing
          factor of AXPYF kernel
        */
        axpyf_kr_ptr = bli_cntx_get_l1f_ker_dt(BLIS_DOUBLE, BLIS_AXPYF_KER, cntx);
        b_fuse = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_AF, cntx);

        scalv_kr_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_SCALV_KER, cntx);

        copyv_kr_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_COPYV_KER, cntx);
    }

    /*
      If alpha is equal to zero, y is only scaled by beta and returned.
      In this case, packing and unpacking y will be costly and it is
      avoided.
    */
    if ( (incy > 1) && (!bli_deq0( *alpha )))
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
        is needed.Following are initializations for rntm */

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        //calculate the size required for n_elem double elements in vector Y.
        size_t buffer_size = n_elem * sizeof(double);

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dgemv_unf_var2(): get mem pool block\n" );
        #endif

        /*acquire a Buffer(n_elem*size(double)) from the memory broker
        and save the associated mem_t entry to mem_bufY.*/
        bli_pba_acquire_m(&rntm,
                                buffer_size,
                                BLIS_BUFFER_FOR_B_PANEL,
                                &mem_bufY);

        /*Continue packing Y if buffer memory is allocated*/
        if ((bli_mem_is_alloc( &mem_bufY )))
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
              n_elem,
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
          n_elem,
          beta,
          y_temp, temp_incy,
          cntx
        );
    }

    if( bli_deq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    for (i = 0; i < n_iter; i += f)
    {
      f = bli_determine_blocksize_dim_f(i, n_iter, b_fuse);

      A1 = a + (i * cs_at);
      x1 = x + (i * incx);

      axpyf_kr_ptr
      (
        conja,
        conjx,
        n_elem,
        f,
        alpha,
        A1, rs_at, cs_at,
        x1, incx,
        y_temp, temp_incy,
        cntx
      );
    }

    if (is_y_temp_buf_created)
    {
        // Store the result from unit strided y_buf to non-unit strided Y
        // Invoke the COPYV function using the function pointer
        copyv_kr_ptr
        (
          BLIS_NO_CONJUGATE,
          n_elem,
          y_temp, temp_incy,
          y, incy,
          cntx
        );

#ifdef BLIS_ENABLE_MEM_TRACING
        printf( "bli_dgemv_unf_var2(): releasing mem pool block\n" );
#endif
        // Return the buffer to pool
        bli_pba_release(&rntm , &mem_bufY);
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}

void bli_sgemv_unf_var2
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       float*  alpha,
       float*  a, inc_t rs_a, inc_t cs_a,
       float*  x, inc_t incx,
       float*  beta,
       float*  y, inc_t incy,
       cntx_t* cntx
     )
{

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
    float*  A1;
    float*  x1;
    float*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    // For AMD these APIS are invoked skipping intermediate framework layers
    // Hence we need to ensure that cntx is set here.
    bli_init_once();
    if(cntx == NULL) cntx = bli_gks_query_cntx();

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_elem, &n_iter, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == FALSE)
    {
        const num_t dt = PASTEMAC(s,type);
        /* If beta is zero, use setv. Otherwise, scale by beta. */
        if ( PASTEMAC(s,eq0)( *beta ) )
        {
            float*  zero = PASTEMAC(s,0);
            /* y = 0; */
            PASTEMAC2(s,setv,BLIS_TAPI_EX_SUF)
            (
              BLIS_NO_CONJUGATE,
              n_elem,
              zero,
              y, incy,
              cntx,
              NULL
            );
        }
        else
        {
            /* y = beta * y; */
            PASTEMAC2(s,scalv,BLIS_TAPI_EX_SUF)
            (
              BLIS_NO_CONJUGATE,
              n_elem,
              beta,
              y, incy,
              cntx,
              NULL
            );
        }

        PASTECH(s,axpyf_ker_ft) kfp_af;

        /* Query the context for the kernel function pointer and fusing factor. */
        kfp_af = bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx );
        b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_AF, cntx );

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

            A1 = a + (0  )*rs_at + (i  )*cs_at;
            x1 = x + (i  )*incx;
            y1 = y + (0  )*incy;

            /* y = y + alpha * A1 * x1; */
            kfp_af
            (
              conja,
              conjx,
              n_elem,
              f,
              alpha,
              A1, rs_at, cs_at,
              x1, incx,
              y1, incy,
              cntx
            );
        }
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
        return;
    }

    /* If beta is zero, use setv. Otherwise, scale by beta. */
        /* y = beta * y; */
    /* beta=0 case is handled by scalv internally */
    bli_sscalv_zen_int10
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      beta,
      y, incy,
      cntx
    );

    if( bli_seq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    /* Query the context for the kernel function pointer and fusing factor. */
    b_fuse = 5;

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

        A1 = a + (0  )*rs_at + (i  )*cs_at;
        x1 = x + (i  )*incx;
        y1 = y + (0  )*incy;

        /* y = y + alpha * A1 * x1; */
        bli_saxpyf_zen_int_5
        (
          conja,
          conjx,
          n_elem,
          f,
          alpha,
          A1, rs_at, cs_at,
          x1, incx,
          y1, incy,
          cntx
        );
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}


void bli_zgemv_unf_var2
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       dcomplex*  alpha,
       dcomplex*  a, inc_t rs_a, inc_t cs_a,
       dcomplex*  x, inc_t incx,
       dcomplex*  beta,
       dcomplex*  y, inc_t incy,
       cntx_t* cntx
     )
{

  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);

  dcomplex *A1;
  dcomplex *x1;
  dcomplex *y1;

  dim_t i, b_fuse, f;
  dim_t n_elem, n_iter;
  inc_t rs_at, cs_at;
  conj_t conja;

  // Memory pool declarations for packing vector Y.
  mem_t mem_bufY;
  rntm_t rntm;
  dcomplex *y_buf = y;
  inc_t buf_incy = incy;

  bli_set_dims_incs_with_trans(transa,
                                m, n, rs_a, cs_a,
                                &n_elem, &n_iter, &rs_at, &cs_at);

  conja = bli_extract_conj(transa);

  // Query the architecture ID
  arch_t id = bli_arch_query_id();

  /*
    Function pointer declaration for the functions
    that will be used by this API
  */
  zaxpyf_ker_ft   axpyf_kr_ptr;  // ZAXPYF
  zscal2v_ker_ft  scal2v_kr_ptr; // ZSCAL2V
  zscalv_ker_ft   scalv_kr_ptr;  // ZSCALV
  zcopyv_ker_ft   copyv_kr_ptr;  // ZCOPYV
  zsetv_ker_ft    setv_kr_ptr;   // ZSETV

  /*
    Boolean to check if the y has been packed
    and memory needs to be freed in the end
  */
  bool is_y_temp_buf_created = FALSE;

  switch (id)
  {
    case BLIS_ARCH_ZEN5:
    case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
      axpyf_kr_ptr = bli_zaxpyf_zen_int_8_avx512;
      b_fuse = 8;

      scal2v_kr_ptr = bli_zscal2v_zen_int;

      scalv_kr_ptr = bli_zscalv_zen_int;

      copyv_kr_ptr = bli_zcopyv_zen_int;

      setv_kr_ptr = bli_zsetv_zen_int_avx512;
      break;
#endif
    case BLIS_ARCH_ZEN:
    case BLIS_ARCH_ZEN2:
    case BLIS_ARCH_ZEN3:

      /*
        Assign the AVX2 based kernel function pointers for
        ZAXPYF, ZSCAL2V, ZSCALV, ZCOPYV and corresponding fusing
        factor of ZAXPYF kernel
      */

      axpyf_kr_ptr = bli_zaxpyf_zen_int_4;
      b_fuse = 4;

      scal2v_kr_ptr = bli_zscal2v_zen_int;

      scalv_kr_ptr = bli_zscalv_zen_int;

      copyv_kr_ptr = bli_zcopyv_zen_int;

      setv_kr_ptr = bli_zsetv_zen_int;
      break;
    default:
      // For non-Zen architectures, query the context if it is NULL
      if(cntx == NULL) cntx = bli_gks_query_cntx();

      /*
        Query the context for the kernel function pointers for
        ZAXPYF, ZSCAL2V, ZSCALV, ZCOPYV and corresponding fusing
        factor of ZAXPYF kernel
      */
      axpyf_kr_ptr = bli_cntx_get_l1f_ker_dt(BLIS_DCOMPLEX, BLIS_AXPYF_KER, cntx);
      b_fuse = bli_cntx_get_blksz_def_dt(BLIS_DCOMPLEX, BLIS_AF, cntx);

      scal2v_kr_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_SCAL2V_KER, cntx);

      scalv_kr_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_SCALV_KER, cntx);

      copyv_kr_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_COPYV_KER, cntx);

      setv_kr_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_SETV_KER, cntx);
  }

  /*
    If alpha is equal to zero, y = beta * y + alpha * A * x
    becomes y = beat * y in that case packing will be costly.
    y is only scaled with SCALV and returned.
  */
  if (incy > 1 && (!bli_zeq0(*alpha)))
  {
    /*
      Initialize mem pool buffer to NULL and size to 0
      "buf" and "size" fields are assigned once memory
      is allocated from the pool in bli_pba_acquire_m().
      This will ensure bli_mem_is_alloc() will be passed on
      an allocated memory if created or a NULL .
    */
    mem_bufY.pblk.buf = NULL;
    mem_bufY.pblk.block_size = 0;
    mem_bufY.buf_type = 0;
    mem_bufY.size = 0;
    mem_bufY.pool = NULL;

    /*
      In order to get the buffer from pool via rntm access to memory broker
      is needed.Following are initializations for rntm
    */

    bli_rntm_init_from_global(&rntm);
    bli_rntm_set_num_threads_only(1, &rntm);
    bli_pba_rntm_set_pba(&rntm);

    // Calculate the size required for n_elem double elements in vector Y.
    size_t buffer_size = n_elem * sizeof(dcomplex);

#ifdef BLIS_ENABLE_MEM_TRACING
    printf("bli_zgemv_unf_var2(): get mem pool block\n");
#endif

    /*
      Acquire a Buffer(n_elem*size(double)) from the memory broker
      and save the associated mem_t entry to mem_bufY.
    */
    bli_pba_acquire_m(&rntm,
                          buffer_size,
                          BLIS_BUFFER_FOR_B_PANEL,
                          &mem_bufY);

    /* Continue packing Y if buffer memory is allocated */
    if ((bli_mem_is_alloc(&mem_bufY)))
    {
      y_buf = bli_mem_buffer(&mem_bufY);
      buf_incy = 1;

      // Invoke the ZSCAL2V function using the function pointer
      scal2v_kr_ptr
      (
        BLIS_NO_CONJUGATE,
        n_elem,
        beta,
        y, incy,
        y_buf, buf_incy,
        cntx
      );

      /*
        Set y is packed as the memory allocation was
        successful and contents have been copied
      */
      is_y_temp_buf_created = TRUE;
    }
  }
  else
  {
    /*
      Invoke the ZSETV function using the function
      pointer only when beta is 0.
    */
    if(PASTEMAC(z, eq0)(*beta))
    {
      setv_kr_ptr
      (
        BLIS_NO_CONJUGATE,
        n_elem,
        beta,
        y_buf, buf_incy,
        cntx
      );
    }
    /*
      Invoke the ZSCALV function using the function
      pointer only when beta is not 1.
    */
    else if(!PASTEMAC(z, eq1)(*beta))
    {
      scalv_kr_ptr
      (
        BLIS_NO_CONJUGATE,
        n_elem,
        beta,
        y_buf, buf_incy,
        cntx
      );
    }
  }

  // If alpha is zero(0), we only need to scalv y and return
  if (bli_zeq0(*alpha))
  {
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);

    // Return early for alpha is zero(0)
    return;
  }

  for (i = 0; i < n_iter; i += f)
  {
    f = bli_determine_blocksize_dim_f(i, n_iter, b_fuse);
    A1 = a + (0) * rs_at + (i)*cs_at;
    x1 = x + (i)*incx;
    y1 = y_buf + (0) * buf_incy;

    // Invoke the ZAXPYF function using the function pointer
    axpyf_kr_ptr
    (
      conja,
      conjx,
      n_elem,
      f,
      alpha,
      A1, rs_at, cs_at,
      x1, incx,
      y1, buf_incy,
      cntx
    );
  }

  // Check if temp y buffer was used for compute
  if (is_y_temp_buf_created)
  {
    // Store the result from unit strided y_buf to non-unit strided Y
    // Invoke the ZCOPYV function using the function pointer
    copyv_kr_ptr
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      y_buf, buf_incy,
      y, incy,
      cntx
    );

#ifdef BLIS_ENABLE_MEM_TRACING
    printf("bli_zgemv_unf_var2(): releasing mem pool block\n");
#endif

    // Return the buffer to pool
    bli_pba_release(&rntm, &mem_bufY);
  }

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}

void bli_cgemv_unf_var2
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       scomplex*  alpha,
       scomplex*  a, inc_t rs_a, inc_t cs_a,
       scomplex*  x, inc_t incx,
       scomplex*  beta,
       scomplex*  y, inc_t incy,
       cntx_t* cntx
     )
{

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
    scomplex*  A1;
    scomplex*  x1;
    scomplex*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    // For AMD these APIS are invoked skipping intermediate framework layers
    // Hence we need to ensure that cntx is set here.
    bli_init_once();
    if(cntx == NULL) cntx = bli_gks_query_cntx();

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_elem, &n_iter, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    /* If beta is zero, use setv. Otherwise, scale by beta. */
        /* y = beta * y; */
    /* beta=0 case is handled by scalv internally */
    /*bli_cscalv_zen_int10
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      beta,
      y,
      incy,
      cntx
    );*/

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == FALSE)
    {
        const num_t dt = PASTEMAC(c,type);
        /* If beta is zero, use setv. Otherwise, scale by beta. */
        if ( PASTEMAC(c,eq0)( *beta ) )
        {
            scomplex*  zero = PASTEMAC(c,0);
            /* y = 0; */
            PASTEMAC2(c,setv,BLIS_TAPI_EX_SUF)
            (
              BLIS_NO_CONJUGATE,
              n_elem,
              zero,
              y, incy,
              cntx,
              NULL
            );
        }
        else
        {
            /* y = beta * y; */
            PASTEMAC2(c,scalv,BLIS_TAPI_EX_SUF)
            (
              BLIS_NO_CONJUGATE,
              n_elem,
              beta,
              y, incy,
              cntx,
              NULL
            );
        }

        PASTECH(c,axpyf_ker_ft) kfp_af;

        /* Query the context for the kernel function pointer and fusing factor. */
        kfp_af = bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx );
        b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_AF, cntx );

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

            A1 = a + (0  )*rs_at + (i  )*cs_at;
            x1 = x + (i  )*incx;
            y1 = y + (0  )*incy;

            /* y = y + alpha * A1 * x1; */
            kfp_af
            (
              conja,
              conjx,
              n_elem,
              f,
              alpha,
              A1, rs_at, cs_at,
              x1, incx,
              y1, incy,
              cntx
            );
        }
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
        return;
    }

    bli_cscalv_ex
	    (
	     BLIS_NO_CONJUGATE,
	     n_elem,
	     beta,
	     y, incy,
	     cntx,
	     NULL
	    );



    if( bli_ceq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    // for non-unit incx, incy and rs_at and conjugate will be added in the next patch
    if( ( (incx == 1) && (incy == 1) && (rs_at == 1) ) &&
         !bli_is_conj(conja) && !bli_is_conj(conjx) &&
         !bli_is_trans(transa))
    {
        // This gemv code deals with the followint conditions only
        // 1. incx, incy, and row stride equal to one
        // 2. Non conjugate A matrix and X vector
        // 3. No Transpose for A Martix
        // Rest is taken care by the else part (axpyf implementation)
        bli_cgemv_zen_int_4x4
        (
            conja,
            conjx,
            m,
            n,
            alpha,
            a, rs_at, cs_at,
            x, incx,
            beta,
            y, incy,
            cntx
        );
    }
    else
    {
        /* fusing factor. */
        b_fuse = 4;

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );
            A1 = a + (0  )*rs_at + (i  )*cs_at;
            x1 = x + (i  )*incx;
            y1 = y + (0  )*incy;

            /* y = y + alpha * A1 * x1; */
            bli_caxpyf_zen_int_4
            (
              conja,
              conjx,
              n_elem,
              f,
              alpha,
              A1, rs_at, cs_at,
              x1, incx,
              y1, incy,
              cntx
            );
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}



