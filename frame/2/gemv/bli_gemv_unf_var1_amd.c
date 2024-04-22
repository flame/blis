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
    ctype* x_temp = x;\
    inc_t  temp_incx = incx;\
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
            x_temp = bli_mem_buffer(&mem_bufX);\
            temp_incx = 1;\
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
              x_temp, temp_incx,\
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
        x1 = x_temp + (0  )*temp_incx; \
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
          x1,   temp_incx, \
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

void bli_dgemv_unf_var1
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

    double *A1;
    double *y1;
    dim_t i;
    dim_t f;
    dim_t n_elem, n_iter;
    inc_t rs_at, cs_at;
    conj_t conja;
    //memory pool declarations for packing vector X.
    mem_t mem_bufX;
    rntm_t rntm;
    double *x_buf = x;
    inc_t buf_incx = incx;

    bli_init_once();

    if (cntx == NULL)
      cntx = bli_gks_query_cntx();

    bli_set_dims_incs_with_trans(transa,
                                 m, n, rs_a, cs_a,
                                 &n_iter, &n_elem, &rs_at, &cs_at);

    conja = bli_extract_conj(transa);

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == FALSE)
    {
        if ( cntx == NULL ) cntx = bli_gks_query_cntx();
        const num_t dt = PASTEMAC(d,type);
        double*  x1;
        double*  y1;
        PASTECH(d,dotxf_ker_ft) kfp_df;
        /* Query the context for the kernel function pointer and fusing factor. */
        kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
        dim_t b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

            A1 = a + (i  )*rs_at + (0  )*cs_at;
            x1 = x + (0  )*incy;
            y1 = y + (i  )*incy;

            /* y1 = beta * y1 + alpha * A1 * x; */
            kfp_df
            (
            conja,
            conjx,
            n_elem,
            f,
            alpha,
            A1,   cs_at, rs_at,
            x1,   incx,
            beta,
            y1,   incy,
            cntx
            );

        }

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
      return;
    }
    if (incx > 1)
    {
    /*
          Initialize mem pool buffer to NULL and size to 0
          "buf" and "size" fields are assigned once memory
          is allocated from the pool in bli_pba_acquire_m().
          This will ensure bli_mem_is_alloc() will be passed on
          an allocated memory if created or a NULL .
    */

    mem_bufX.pblk.buf = NULL;
    mem_bufX.pblk.block_size = 0;
    mem_bufX.buf_type = 0;
    mem_bufX.size = 0;
    mem_bufX.pool = NULL;

    /* In order to get the buffer from pool via rntm access to memory broker
        is needed.Following are initializations for rntm */

    bli_rntm_init_from_global(&rntm);
    bli_rntm_set_num_threads_only(1, &rntm);
    bli_pba_rntm_set_pba(&rntm);

    //calculate the size required for n_elem double elements in vector X.
    size_t buffer_size = n_elem * sizeof(double);

#ifdef BLIS_ENABLE_MEM_TRACING
    printf("bli_dgemv_unf_var1(): get mem pool block\n");
#endif

    /*acquire a Buffer(n_elem*size(double)) from the memory broker
      and save the associated mem_t entry to mem_bufX.*/
    bli_pba_acquire_m(&rntm,
                         buffer_size,
                         BLIS_BUFFER_FOR_B_PANEL,
                         &mem_bufX);

    /*Continue packing X if buffer memory is allocated*/
    if ((bli_mem_is_alloc(&mem_bufX)))
    {
      x_buf = bli_mem_buffer(&mem_bufX);

      //pack X vector with non-unit stride to a temp buffer x_buf with unit stride
      for (dim_t x_index = 0; x_index < n_elem; x_index++)
      {
        *(x_buf + x_index) = *(x + (x_index * incx));
      }
      // stride of vector x_buf =1
      buf_incx = 1;
    }
  }

  dim_t fuse_factor = 8;
  dim_t f_temp =0;

  if (n < 4)
  {
     fuse_factor = 2;
  } else if (n < 8)
  {
     fuse_factor = 4;
  }

  for (i = 0; i < n_iter; i += f)
  {
    f = bli_determine_blocksize_dim_f(i, n_iter, fuse_factor);

    //A = a + i * row_increment + 0 * column_increment
    A1 = a + (i)*rs_at;
    y1 = y + (i)*incy;

    /* y1 = beta * y1 + alpha * A1 * x; */
    switch (f)
    {
    case 8:

      bli_ddotxf_zen_int_8(
          conja,
          conjx,
          n_elem,
          f,
          alpha,
          A1, cs_at, rs_at,
          x_buf, buf_incx,
          beta,
          y1, incy,
          cntx);

      break;
    default:

      if (f < 4)
      {
        bli_ddotxf_zen_int_2(
            conja,
            conjx,
            n_elem,
            f,
            alpha,
            A1, cs_at, rs_at,
            x_buf, buf_incx,
            beta,
            y1, incy,
            cntx);
      }
      else
      {
        bli_ddotxf_zen_int_4(
            conja,
            conjx,
            n_elem,
            f,
            alpha,
            A1, cs_at, rs_at,
            x_buf, buf_incx,
            beta,
            y1, incy,
            cntx);
      }
    }

    f_temp = bli_determine_blocksize_dim_f(i + f, n_iter, fuse_factor);

    if (f_temp < fuse_factor)
    {
      switch (fuse_factor)
      {
      case 8:
        fuse_factor = 4;
        break;
      case 4:
        fuse_factor = 2;
        break;
      }
    }
  }

  if ((incx > 1) && bli_mem_is_alloc(&mem_bufX))
  {
#ifdef BLIS_ENABLE_MEM_TRACING
    printf("bli_dgemv_unf_var1(): releasing mem pool block\n");
#endif
    // Return the buffer to pool
    bli_pba_release(&rntm, &mem_bufX);
  }
  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}

// Returns the optimal number of threads for the given input sizes and fuse factor
void bli_sgemv_var1_smart_threading
    (
      dim_t m, dim_t n,
      dim_t fuse,
      dim_t* nt, dim_t nt_max
    )
{
  // Calculate the amount data processed per iteration
  dim_t n_per_loop = n / fuse;
  double data_per_iter =  n_per_loop* m;

  // Exception handling when m-dimenstion or n-dimension is zero
  if (bli_zero_dim2(m,n))
  {
    *nt = 1;
    return;
  }

  double m_n_ratio = m/n;

  // When the input value is less than the fuse factor
  if(n_per_loop < 1)
  {
    *nt = 1;
    return;
  }
  
  // Then there are two cases one
  // In m < n the thread spawning is less aggressive when compared to m > n and m = n cases
  if(m_n_ratio <= 0.6)
  {
    // Boundary units is the amount of data processed by each iteration
    // This is the variable X in the equation
    const double lower_boundary = 50000;
    const double higher_boundary = 500000;

    if(data_per_iter < lower_boundary)
    {
      double coeff_x = 0.9148;
      double constant = -1.6252;
      // Number of threads =  0.9148 * log(x) - 1.6252
      *nt = ceil(coeff_x * log(data_per_iter) + constant);
    }
    else if(data_per_iter < higher_boundary)
    {
      float coeff_x = 10.23;
      float constant = -82.332;
      // Number of threads = 10.23 * log(x) - 82.332
      *nt = ceil(coeff_x * log(data_per_iter) + constant);
    }
    else
    {
      // When the amount of data to be processed is above both of the boundaries
      // The number of threads spawned will be equal to the max number of threads set
      *nt = nt_max;
    }
  }
  else
  {
    // Boundary units is the amount of data processed by each iteration
    // This is the variable X in the equation
    const float lower_boundary = 50000;
    const float higher_boundary = 360000;

    if(data_per_iter < lower_boundary)
    {
      float coeff_x2 = -2E-09;
      float coeff_x = 0.0002;
      float constant = 1.0234;
      // Number of threads = -2E-09*x^2 + 0.0002 * x + 1.0234
      *nt = ceil(coeff_x2 * (data_per_iter * data_per_iter) + coeff_x * data_per_iter + constant);
    }
    else if(data_per_iter < higher_boundary)
    {
      float coeff_x = 16.917;
      float constant = -164.82;
      // Number of threads = 16.917 * log(x) - 164.82
      *nt = ceil(coeff_x * log(data_per_iter) + constant);
    }
    else
    {
      // When the amount of data to be processed is above both of the boundaries
      // The number of threads spawned will be equal to the max number of threads set
      *nt = nt_max;
    }
  }


  // When the number of threads calculated is greater than the user provided value
  // Choose the user provided value
  if(*nt > nt_max ) *nt = nt_max;
  if(*nt <=0 ) *nt = 1;
}

void bli_sgemv_unf_var1
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

    float*  A1;
    float*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    bli_init_once();

    if( cntx == NULL ) cntx = bli_gks_query_cntx();

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_iter, &n_elem, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == FALSE)
    {
        if ( cntx == NULL ) cntx = bli_gks_query_cntx();
        const num_t dt = PASTEMAC(s,type);
        float*  x1 ;
        PASTECH(s,dotxf_ker_ft) kfp_df;
        /* Query the context for the kernel function pointer and fusing factor. */
        kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
        b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

            A1 = a + (i  )*rs_at + (0  )*cs_at;
            x1 = x + (0  )*incy;
            y1 = y + (i  )*incy;

            /* y1 = beta * y1 + alpha * A1 * x; */
            kfp_df
            (
            conja,
            conjx,
            n_elem,
            f,
            alpha,
            A1,   cs_at, rs_at,
            x1,   incx,
            beta,
            y1,   incy,
            cntx
            );

        }

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
      return;
    }

    dim_t nt_max;

    rntm_t rnmt_obj;
    // Initialize a local runtime with global settings.
    bli_rntm_init_from_global( &rnmt_obj );

    // Query the total number of threads from the rntm_t object.
    nt_max = bli_rntm_num_threads( &rnmt_obj );

    if (nt_max<=0)
    {
        // nt is less than one if BLIS manual setting of parallelism
        // has been used. Parallelism here will be product of values.
        dim_t jc, pc, ic, jr, ir;
        jc = bli_rntm_jc_ways( &rnmt_obj );
        pc = bli_rntm_pc_ways( &rnmt_obj );
        ic = bli_rntm_ic_ways( &rnmt_obj );
        jr = bli_rntm_jr_ways( &rnmt_obj );
        ir = bli_rntm_ir_ways( &rnmt_obj );
        nt_max = jc*pc*ic*jr*ir;
    }

// If OpenMP is enabled, GEMV will multithread
#ifdef BLIS_ENABLE_OPENMP
    if ( nt_max > 1 )
    {
        b_fuse = 4;

        //Setting the thread count to the maximum number of threads provided
        dim_t nt = nt_max;

        // Enable smart threading when AOCL dynamic is enabled
        #ifdef AOCL_DYNAMIC
          bli_sgemv_var1_smart_threading(n_elem, n_iter, b_fuse, &nt, nt_max);
        #endif

        // Pass the input paramaters along with the number of threads to be used
        bli_multi_sgemv_4x2
        (
          conja,
          conjx,
          n_elem,
          n_iter,
          alpha,
          a, cs_at, rs_at,
          x, incx,
          beta,
          y, incy,
          cntx,
          nt
        );
    }
    else
    {
#endif// BLIS_ENABLE_OPENMP
        b_fuse = 8;

        for ( i = 0; i < n_iter; i += f )
        {
            float*  x1;
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

            A1 = a + (i  )*rs_at + (0  )*cs_at;
            x1 = x + (0  )*incy;
            y1 = y + (i  )*incy;

            /* y1 = beta * y1 + alpha * A1 * x; */
            bli_sdotxf_zen_int_8
            (
              conja,
              conjx,
              n_elem,
              f,
              alpha,
              A1,   cs_at, rs_at,
              x1,   incx,
              beta,
              y1,   incy,
              cntx
            );
        }
#ifdef BLIS_ENABLE_OPENMP
    }
#endif// BLIS_ENABLE_OPENMP
}

void bli_zgemv_unf_var1
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

    const num_t dt = PASTEMAC(z,type);

    dcomplex*  A1;
    dcomplex*  x1;
    dcomplex*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    /* Memory pool declarations for packing vector X. */
    mem_t  mem_bufX;
    rntm_t rntm;
    dcomplex* x_temp = x;
    inc_t  temp_incx = incx;
    /*
     Boolean to check if the X has been packed
     and memory needs to be freed in the end
    */
    bool is_x_temp_buf_created = FALSE;

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_iter, &n_elem, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    /*
      Function pointer declaration for the functions
      that will be used by this API
    */
    zdotxf_ker_ft   dotxf_kr_ptr; // ZDOTXF
    zscal2v_ker_ft  scal2v_kr_ptr; // ZSCAL2V

   /*
      Fatbinary config amdzen when run on non-AMD X86 will query for
      the support of AVX512 or AVX2, if AVX512 - arch_id will be zen4
      or for AVX2 it will be zen3.
    */
    arch_t id = bli_arch_query_id();

    switch (id)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
        /*
          Assign the AVX2 based kernel function pointers for
          DOTXF, SCAL2Vand corresponding fusing
          factor of DOTXF kernel
        */

        dotxf_kr_ptr = bli_zdotxf_zen_int_8_avx512;
        b_fuse = 8;

        scal2v_kr_ptr = bli_zscal2v_zen_int;
        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

        /*
          Assign the AVX2 based kernel function pointers for
          DOTXF, SCAL2Vand corresponding fusing
          factor of DOTXF kernel
        */

        dotxf_kr_ptr = bli_zdotxf_zen_int_6;
        b_fuse = 6;

        scal2v_kr_ptr = bli_zscal2v_zen_int;
        break;

      default:
        // For non-Zen architectures, query the context if it is NULL
        if(cntx == NULL) cntx = bli_gks_query_cntx();

        /*
          Query the context for the kernel function pointers for
          DOTXF, SCAL2V and corresponding fusing
          factor of DOTXF kernel
        */
        dotxf_kr_ptr = bli_cntx_get_l1f_ker_dt( BLIS_DCOMPLEX, BLIS_DOTXF_KER, cntx );;
        b_fuse =  bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );

        scal2v_kr_ptr = bli_cntx_get_l1v_ker_dt(dt, BLIS_SCAL2V_KER, cntx);
    }

    if( incx > 1 )
    {
        /*
        Initialize mem pool buffer to NULL and size to 0
        "buf" and "size" fields are assigned once memory
        is allocated from the pool in bli_pba_acquire_m().
        This will ensure bli_mem_is_alloc() will be passed on
        an allocated memory if created or a NULL .
        */
        mem_bufX.pblk.buf = NULL;
        mem_bufX.pblk.block_size = 0;
        mem_bufX.buf_type = 0;
        mem_bufX.size = 0;
        mem_bufX.pool = NULL;

        /*
          In order to get the buffer from pool via rntm access to memory broker
          is needed.Following are initializations for rntm
        */

        bli_rntm_init_from_global(&rntm);
        bli_rntm_set_num_threads_only(1, &rntm);
        bli_pba_rntm_set_pba(&rntm);

        /*
          Calculate the size required for n_elem double elements in vector Y.
        */
        size_t buffer_size = n_elem * sizeof(dcomplex);

        /*
          Acquire a Buffer(n_elem*size(dcomplex)) from the memory broker
          and save the associated mem_t entry to mem_bufX.
        */
        bli_pba_acquire_m(&rntm, buffer_size, BLIS_BUFFER_FOR_B_PANEL, &mem_bufX);

        /*
         Continue packing X if buffer memory is allocated
        */
        if ((bli_mem_is_alloc(&mem_bufX)))
        {
            x_temp = bli_mem_buffer(&mem_bufX);
            temp_incx = 1;
            dcomplex* alpha_passed = PASTEMAC(z,1);

            /*
              Invoke the ZSCAL2V function using the function pointer
            */
            scal2v_kr_ptr
            (
              BLIS_NO_CONJUGATE,
              n_elem,
              alpha_passed,
              x, incx,
              x_temp, temp_incx,
              cntx
            );

            /*
              Set x is packed as the memory allocation was
              successful and contents have been copied
            */
            is_x_temp_buf_created = TRUE;
        }
    }

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

        A1 = a + (i  )*rs_at + (0  )*cs_at;
        x1 = x_temp + (0  )*temp_incx;
        y1 = y + (i  )*incy;

        /* y1 = beta * y1 + alpha * A1 * x; */
        dotxf_kr_ptr
        (
          conja,
          conjx,
          n_elem,
          f,
          alpha,
          A1,   cs_at, rs_at,
          x1,   temp_incx,
          beta,
          y1,   incy,
          cntx
        );

    }
    /*
      Check if temp X buffer was used for compute
    */
    if (is_x_temp_buf_created)
    {
        /*
          Return the buffer to pool
        */
        bli_pba_release(&rntm, &mem_bufX);
    }
}


INSERT_GENTFUNC_BASIC0_C( gemv_unf_var1 )

