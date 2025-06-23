/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

// Enable fast path for small GEMV problems when AOCL_DYNAMIC is defined.
#if defined(AOCL_DYNAMIC)
  // Fast path is enabled if the total problem size is below a threshold,
  #define BLI_FAST_PATH (((n0 * m0) <= fast_path_thresh))
#else
  // Fast path is disabled if AOCL_DYNAMIC is not defined.
  #define BLI_FAST_PATH 0
#endif

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
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3)

    dim_t i;
    dim_t f;
    dim_t m0 = m, n0 = n;
    inc_t lda = cs_a, inca = rs_a;
    conj_t conja;

    double *a_buf = a;
    double *x_buf = x;
    double *y_buf = y;

    inc_t buf_incx = incx;
    inc_t buf_incy = incy;

    // Invoking the reference kernel to handle general stride.
    if ( ( rs_a != 1 ) && ( cs_a != 1 ) )
    {
        bli_dgemv_zen_ref
        (
          transa,
          m,
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          y, incy,
          NULL
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    // 'bli_dgemv_unf_var1' is dot-based kernel. This kernel is called for the following cases:
    //
    // When op(A) = n and row-storage( lda = rs_a ), we compute dot product as y[i] = <A(i,:), x>, i = 0:m-1.
    // gemv dot kernel always computes dot-product along the columns of A, we interchange m and n. Here m0 = n, n0 = m.
    //
    // op(A) = n   ->     lda  = rs_a;
    //                    inca = cs_a;
    //                    m0   = n;
    //                    n0   = m;
    //
    // when op(A) = t and col-storage( lda = cs_a ), we compute dot product as y[i] = <A(:, i), x>, i = 0:n-1. Anyways
    // the kernel computes dot along the columns of A, we don't interchange m & n, so here m0 = m and n0 = n.
    //
    // op(A) = t   ->     lda  = cs_a;
    //                    inca = rs_a;
    //                    m0   = m;
    //                    n0   = n;
    //
    bli_set_dims_incs_with_trans(transa,
                                m, n, rs_a, cs_a,
                                &n0, &m0, &lda, &inca);

    // Extract the conjugation from transa.
    conja = bli_extract_conj(transa);

    //memory pool declarations for packing vector X and Y.
    mem_t mem_bufX;
    mem_t mem_bufY;
    rntm_t rntm;

    // Boolean to check if x and y vectors are packed and memory needs to be freed.
    bool is_x_temp_buf_created = FALSE;
    bool is_y_temp_buf_created = FALSE;

    // Function pointer declaration for the functions that will be used.
    dgemv_ker_ft   gemv_kr_ptr;         // DGEMV
    dscalv_ker_ft  scalv_kr_ptr;        // DSCALV
    dcopyv_ker_ft  copyv_kr_ptr;        // DCOPYV

    /*
      Fatbinary config amdzen when run on non-AMD X86 will query for
      the support of AVX512 or AVX2, if AVX512 - arch_id will be zen4
      and zen5 or for AVX2 it will be zen3.
    */
    arch_t id = bli_arch_query_id();

#if defined(BLIS_ENABLE_OPENMP) && defined(AOCL_DYNAMIC)
    // Setting the threshold to invoke the fast-path
    // The fast-path is intended to directly call the kernel
    // in case the criteria for single threaded execution is met.
    dim_t fast_path_thresh = 0;
#endif

    switch (id)
    {
      case BLIS_ARCH_ZEN5:
#if defined(BLIS_KERNELS_ZEN5)
      gemv_kr_ptr   = bli_dgemv_t_zen4_int;     // DGEMV
      scalv_kr_ptr  = bli_dscalv_zen_int_avx512;      // DSCALV
      copyv_kr_ptr  = bli_dcopyv_zen5_asm_avx512;     // DCOPYV
#if defined(BLIS_ENABLE_OPENMP) && defined(AOCL_DYNAMIC)
      fast_path_thresh = 12000;
#endif
      break;
#endif
      case BLIS_ARCH_ZEN4:

#if defined(BLIS_KERNELS_ZEN4)
        gemv_kr_ptr   = bli_dgemv_t_zen4_int;     // DGEMV
        scalv_kr_ptr  = bli_dscalv_zen_int_avx512;      // DSCALV
        copyv_kr_ptr  = bli_dcopyv_zen4_asm_avx512;     // DCOPYV
#if defined(BLIS_ENABLE_OPENMP) && defined(AOCL_DYNAMIC)
        fast_path_thresh = 11000;
#endif
        break;
#endif

      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

        gemv_kr_ptr   = bli_dgemv_t_zen_int;       // DGEMV
        scalv_kr_ptr  = bli_dscalv_zen_int;             // DSCALV
        copyv_kr_ptr  = bli_dcopyv_zen_int;             // DCOPYV
#if defined(BLIS_ENABLE_OPENMP) && defined(AOCL_DYNAMIC)
          fast_path_thresh = 13000;
#endif
        break;

      default:
        // This function is invoked on all architectures including 'generic'.
        // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
        if ( cntx == NULL )
            cntx = bli_gks_query_cntx();

        const num_t dt = PASTEMAC(d,type);

        double*  x1;
        double*  y1;
        double*  A1;

        PASTECH(d,dotxf_ker_ft) kfp_df;

        // Query the context for the ddotxf kernel function pointer and fusing factor.
        kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
        dim_t b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );

        // 
        for ( i = 0; i < n0; i += f )
        {
            // Determine the blocksize for the current iteration.
            f  = bli_determine_blocksize_dim_f( i, n0, b_fuse );

            // Calculate the pointers to the current block of A, x, and y.
            A1 = a_buf + ( i * lda ) + ( 0 * inca );
            x1 = x_buf;
            y1 = y_buf + ( i * incy );

            // kfp_df is a function pointer to the dotxf kernel
            kfp_df
            (
                conja,
                conjx,
                m0,
                f,
                alpha,
                A1,   inca, lda,
                x1,   incx,
                beta,
                y1,   incy,
                cntx
            );
          }

          AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
          return;
    }

    /*  If y has non-unit increments and alpha is non-zero, y is packed and
        scaled by beta. The scaled contents are copied to a temp buffer (y_buf)
        and passed to the kernels. At the end, the contents of y_buf are copied
        back to y and memory is freed.

        If alpha is zero, the GEMV operation is reduced to y := beta * y, thus,
        packing of y is unnecessary so y is only scaled by beta and returned.
    */

    if ( (incy != 1) && (!bli_deq0( *alpha )))
    {
        /*  Initialize mem pool buffer to NULL and size to 0.
            "buf" and "size" fields are assigned once memory is allocated from
            the pool in bli_pba_acquire_m().

            This will ensure bli_mem_is_alloc() will be passed on an allocated
            memory if created or a NULL.
        */
        mem_bufY.pblk.buf = NULL;   mem_bufY.pblk.block_size = 0;
        mem_bufY.buf_type = 0;      mem_bufY.size = 0;
        mem_bufY.pool = NULL;

        // In order to get the buffer from pool via rntm access to memory broker
        // is needed.Following are initializations for rntm.
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        // Calculate the size required for n0 double elements in vector Y.
        size_t buffer_size = n0 * sizeof( double );

        #ifdef BLIS_ENABLE_MEM_TRACING
        printf("bli_dgemv_unf_var1(): get mem pool block for vector y\n");
        #endif

        // Acquire a Buffer(n0*size(double)) from the memory broker and save the
        // associated mem_t entry to mem_bufY.
        bli_pba_acquire_m
        (
          &rntm,
          buffer_size,
          BLIS_BUFFER_FOR_B_PANEL,
          &mem_bufY
        );

        // Continue packing Y if buffer memory is allocated.
        if ( bli_mem_is_alloc( &mem_bufY ) )
        {
            y_buf = bli_mem_buffer( &mem_bufY );

            // Using unit-stride for y_temp vector.
            buf_incy = 1;

            // Invoke the COPYV function using the function pointer.
            copyv_kr_ptr
            (
              BLIS_NO_CONJUGATE,
              n0,
              y, incy,
              y_buf, buf_incy,
              cntx
            );

            // Set y is packed as the memory allocation was successful
            // and contents have been scaled and copied to a temp buffer.
            is_y_temp_buf_created = TRUE;
        }
    }

    // If alpha is zero, the GEMV operation is reduced to y := beta * y, thus,
    // y is only scaled by beta and returned.
    if( bli_deq0( *alpha ) )
    {
        // Invoke the SCALV function using the function pointer
        scalv_kr_ptr
        (
          BLIS_NO_CONJUGATE,
          n0,
          beta,
          y_buf, buf_incy,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    // If x has non-unit increments , x is packed and copied to a temp buffer (x_buf)
    // and passed to the kernels. At the end, the memory is freed.
    if ( incx != 1 )
    {
        /*
              Initialize mem pool buffer to NULL and size to 0
              "buf" and "size" fields are assigned once memory
              is allocated from the pool in bli_pba_acquire_m().
              This will ensure bli_mem_is_alloc() will be passed on
              an allocated memory if created or a NULL .
        */

        mem_bufX.pblk.buf = NULL;         mem_bufX.pblk.block_size = 0;
        mem_bufX.buf_type = 0;            mem_bufX.size = 0;
        mem_bufX.pool = NULL;

        // In order to get the buffer from pool via rntm access to memory broker
        // is needed.Following are initializations for rntm.
        bli_rntm_init_from_global(&rntm);
        bli_rntm_set_num_threads_only(1, &rntm);
        bli_pba_rntm_set_pba(&rntm);

        //calculate the size required for m0 double elements in vector X.
        size_t buffer_size = m0 * sizeof(double);

#ifdef BLIS_ENABLE_MEM_TRACING
        printf("bli_dgemv_unf_var1(): get mem pool block for vector x\n");
#endif

        // acquire a Buffer(m0*size(double)) from the memory broker
        // and save the associated mem_t entry to mem_bufX.
        bli_pba_acquire_m(&rntm,
                            buffer_size,
                            BLIS_BUFFER_FOR_B_PANEL,
                            &mem_bufX);

        // Continue packing X if buffer memory is allocated.
        if ((bli_mem_is_alloc(&mem_bufX)))
        {
            x_buf = bli_mem_buffer(&mem_bufX);

            // stride of vector x_buf =1
            buf_incx = 1;

            // Invoke the COPYV function using the function pointer.
            copyv_kr_ptr
            (
                BLIS_NO_CONJUGATE,
                m0,
                x, incx,
                x_buf, buf_incx,
                cntx
            );

            // Set x is packed as the memory allocation was successful
            // and contents have been copied to a temp buffer.
            is_x_temp_buf_created = TRUE;
        }
    }

    // If the increments of x and y are unit stride, we can use the
    // optimized kernel path. The optimized kernel does not support
    // non-unit stride for x and y.
    if ( buf_incx == 1 && buf_incy == 1 )
    {
#if defined(BLIS_ENABLE_OPENMP)
      // If the problem size is small, we can use a fast-path to avoid
      // the overhead of threading.
        if( BLI_FAST_PATH )
      {
#endif
        // Call the DGEMV kernel directly with the packed buffers.
        gemv_kr_ptr
        (
            conja,
            conjx,
            m0,
            n0,
            alpha,
            a_buf, inca, lda,
            x_buf, buf_incx,
            beta,
            y_buf, buf_incy,
            cntx
        );

#if defined(BLIS_ENABLE_OPENMP)
      }
      else
      {
        // Initializing nt as 1 to avoid compiler warnings
        dim_t nt = 1;

        /*
        For the given problem size and architecture, the function
        returns the optimum number of threads with AOCL dynamic enabled
        else it returns the number of threads requested by the user.
        */

        bli_nthreads_l2
        (
            BLIS_GEMV_KER,
            BLIS_DOUBLE,
            BLIS_TRANSPOSE,
            id,
            n0,
            m0,
            &nt
        );

        _Pragma("omp parallel num_threads(nt)")
        {
          dim_t start, end;
          thrinfo_t thread;

          // The factor by which the size should be a multiple during thread partition.
          // The main loop of the kernel can handle 8 elements at a time hence 8 is selected for block_size.
          dim_t block_size = 8;

          // Get the thread ID
          bli_thrinfo_set_work_id( omp_get_thread_num(), &thread );

          // Get the actual number of threads spawned
          bli_thrinfo_set_n_way( omp_get_num_threads(), &thread );

          /*
          Calculate the compute range (start and end) for the current thread
          based on the actual number of threads spawned
          */

          bli_thread_range_sub
          (
              &thread,
              n0,
              block_size,
              FALSE,
              &start,
              &end
          );

          // Calculating the value of n for the particular thread
          dim_t n_thread_local = end - start;

          // Calculating thread specific pointers
          double *a_thread_local = a_buf + (start * lda);
          double *y_thread_local = y_buf + start;
          double *x_thread_local = x_buf;

          // Call the DGEMV kernel with the thread-local pointers.
          gemv_kr_ptr
          (
              conja,
              conjx,
              m0,
              n_thread_local,
              alpha,
              a_thread_local, inca, lda,
              x_thread_local, buf_incx,
              beta,
              y_thread_local, buf_incy,
              cntx
          );
        }
      }
#endif
    }

    // If the increments of x and y are not unit stride, we call the reference kernel.
    else
    {
      bli_dgemv_zen_ref
        (
          transa,
          m,
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          y, incy,
          NULL
        );
    }

    // If x was packed into x_temp, free the memory.
    if (is_x_temp_buf_created)
    {
 #ifdef BLIS_ENABLE_MEM_TRACING
        printf("bli_dgemv_unf_var1(): releasing mem pool block for vector x\n");
 #endif
        // Return the buffer to pool
        bli_pba_release(&rntm, &mem_bufX);
    }

    // If y was packed into y_temp, copy the contents back to y and free memory.
    if (is_y_temp_buf_created)
    {
        // Invoke COPYV to store the result from unit-strided y_buf to non-unit
        // strided y.
        copyv_kr_ptr
        (
          BLIS_NO_CONJUGATE,
          n0,
          y_buf, buf_incy,
          y, incy,
          cntx
        );

 #ifdef BLIS_ENABLE_MEM_TRACING
        printf("bli_dgemv_unf_var1(): releasing mem pool block for vector y\n");
 #endif
        // Return the buffer to pool.
        bli_pba_release( &rntm , &mem_bufY );
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

