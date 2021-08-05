/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.

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
#include "immintrin.h"
#include "blis.h"

#ifdef BLIS_ENABLE_FAST_MATH
/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
    __m256  v;
    float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
    __m256d v;
    double  d[4] __attribute__((aligned(64)));
} v4df_t;

// -----------------------------------------------------------------------------

void bli_dnorm2fv_unb_var1
     (
       dim_t    n,
       double*   x, inc_t incx,
       double* norm,
       cntx_t*  cntx
     )
{
    double sumsq = 0;
    double rem_sumsq = 0; /*sum of squares accumulated for n_remainder<8 cases.*/
    dim_t n_remainder = 0;
    dim_t i;
    /*memory pool declarations for packing vector X.
      Initialize mem pool buffer to NULL and size to 0
      "buf" and "size" fields are assigned once memory
      is allocated from the pool in bli_membrk_acquire_m().
      This will ensure bli_mem_is_alloc() will be passed on
      an allocated memory if created or a NULL .*/
    mem_t   mem_bufX = {0};
    rntm_t  rntm;
    double  *x_buf = x;
    
    /*early return if n<=0 or incx =0 */
    if((n <= 0) || (incx == 0))
        return;
    
    /*packing for non-unit strided Vector X*/
    if(incx != 1)
    {
        /* In order to get the buffer from pool via rntm access to memory broker
        is needed.Following are initializations for rntm */

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_membrk_rntm_set_membrk( &rntm );

        //calculate the size required for "n" double elements in vector X.
        size_t buffer_size = n * sizeof(double);

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dnorm2fv_unb_var1(): get mem pool block\n" );
        #endif

        /*acquire a Buffer(n*size(double)) from the memory broker
        and save the associated mem_t entry to mem_bufX.*/
        bli_membrk_acquire_m(&rntm,
                                buffer_size,
                                BLIS_BUFFER_FOR_B_PANEL,
                                &mem_bufX);

        /*Continue packing X if buffer memory is allocated*/
        if ((bli_mem_is_alloc( &mem_bufX )))
        {
            x_buf = bli_mem_buffer(&mem_bufX);

            /*pack X vector with non-unit stride to a temp buffer x_buf with unit stride*/
            for(dim_t x_index = 0 ; x_index < n ; x_index++)
            {
                *(x_buf + x_index) =  *(x + (x_index * incx)) ;
            }
        }
    }

    v4df_t x0v, x1v, x2v, x3v, x4v, x5v, x6v, x7v;
    /* Initialize rho vector accumulators to zero.*/
    v4df_t rho0v; rho0v.v = _mm256_setzero_pd();
    v4df_t rho1v; rho1v.v = _mm256_setzero_pd();
    v4df_t rho2v; rho2v.v = _mm256_setzero_pd();
    v4df_t rho3v; rho3v.v = _mm256_setzero_pd();
    v4df_t rho4v; rho4v.v = _mm256_setzero_pd();
    v4df_t rho5v; rho5v.v = _mm256_setzero_pd();
    v4df_t rho6v; rho6v.v = _mm256_setzero_pd();
    v4df_t rho7v; rho7v.v = _mm256_setzero_pd();

    double *x0 = x_buf;

    for(i = 0 ; i+31 < n ; i = i + 32)
    {

        x0v.v = _mm256_loadu_pd( x0 );
        x1v.v = _mm256_loadu_pd( x0 + 4 );
        x2v.v = _mm256_loadu_pd( x0 + 8 );
        x3v.v = _mm256_loadu_pd( x0 + 12 );
        x4v.v = _mm256_loadu_pd( x0 + 16 );
        x5v.v = _mm256_loadu_pd( x0 + 20 );
        x6v.v = _mm256_loadu_pd( x0 + 24 );
        x7v.v = _mm256_loadu_pd( x0 + 28 );

        rho0v.v   = _mm256_fmadd_pd(x0v.v, x0v.v, rho0v.v);
        rho1v.v   = _mm256_fmadd_pd(x1v.v, x1v.v, rho1v.v);
        rho2v.v   = _mm256_fmadd_pd(x2v.v, x2v.v, rho2v.v);
        rho3v.v   = _mm256_fmadd_pd(x3v.v, x3v.v, rho3v.v);
        rho4v.v   = _mm256_fmadd_pd(x4v.v, x4v.v, rho4v.v);
        rho5v.v   = _mm256_fmadd_pd(x5v.v, x5v.v, rho5v.v);
        rho6v.v   = _mm256_fmadd_pd(x6v.v, x6v.v, rho6v.v);
        rho7v.v   = _mm256_fmadd_pd(x7v.v, x7v.v, rho7v.v);

        x0 += 32;
    }

    n_remainder = n - i;

    if(n_remainder)
    {
        if(n_remainder >= 16)
        {
            x0v.v = _mm256_loadu_pd( x0 );
            x1v.v = _mm256_loadu_pd( x0 + 4 );
            x2v.v = _mm256_loadu_pd( x0 + 8 );
            x3v.v = _mm256_loadu_pd( x0 + 12 );

            rho0v.v   = _mm256_fmadd_pd(x0v.v, x0v.v, rho0v.v);
            rho1v.v   = _mm256_fmadd_pd(x1v.v, x1v.v, rho1v.v);
            rho2v.v   = _mm256_fmadd_pd(x2v.v, x2v.v, rho2v.v);
            rho3v.v   = _mm256_fmadd_pd(x3v.v, x3v.v, rho3v.v);

            x0 += 16;
            n_remainder -= 16;
        }
        if(n_remainder >= 8)
        {
            x0v.v = _mm256_loadu_pd( x0 );
            x1v.v = _mm256_loadu_pd( x0 + 4 );

            rho0v.v   = _mm256_fmadd_pd(x0v.v, x0v.v, rho0v.v);
            rho1v.v   = _mm256_fmadd_pd(x1v.v, x1v.v, rho1v.v);

            x0 += 8;
            n_remainder -= 8;
        }
        if(n_remainder >= 4)
        {
            x0v.v = _mm256_loadu_pd( x0 );

            rho0v.v   = _mm256_fmadd_pd(x0v.v, x0v.v, rho0v.v);

            x0 += 4;
            n_remainder -= 4;
        }
        if(n_remainder)
        {
            for(i=0; i< n_remainder ;i++)
            {
                double x_temp = *x0;
                rem_sumsq += x_temp * x_temp ;
                x0 += 1;
            }
        }
    }

    /*add all the dot product of x*x into one vector .*/
    rho0v.v = _mm256_add_pd ( rho0v.v, rho1v.v );
    rho1v.v = _mm256_add_pd ( rho2v.v, rho3v.v );
    rho2v.v = _mm256_add_pd ( rho4v.v, rho5v.v );
    rho3v.v = _mm256_add_pd ( rho6v.v, rho7v.v );

    rho4v.v = _mm256_add_pd ( rho0v.v, rho1v.v );
    rho5v.v = _mm256_add_pd ( rho2v.v, rho3v.v );

    rho6v.v = _mm256_add_pd ( rho4v.v, rho5v.v );

    rho7v.v = _mm256_hadd_pd( rho6v.v, rho6v.v );

    /*rem_sumsq will have sum of squares of n_remainder < 4 cases .
      Accumulate all the sum of squares to sumsq*/
    sumsq = rem_sumsq + rho7v.d[0] + rho7v.d[2];

    PASTEMAC(d,sqrt2s)( sumsq, *norm );

    if ((incx != 1) && bli_mem_is_alloc( &mem_bufX ))
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dnorm2fv_unb_var1(): releasing mem pool block\n" );
        #endif
        /* Return the buffer to pool*/
        bli_membrk_release(&rntm , &mem_bufX);
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
    return ;
}
#endif
