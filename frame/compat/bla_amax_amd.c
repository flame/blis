/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNC
#define GENTFUNC( ftype_x, chx, blasname, blisname ) \
\
f77_int PASTEF772S(i,chx,blasname) \
     ( \
       const f77_int* n, \
       const ftype_x* x, const f77_int* incx  \
     ) \
{ \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
    AOCL_DTL_LOG_AMAX_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(chx), *n, *incx) \
\
    dim_t    n0; \
    ftype_x* x0; \
    inc_t    incx0; \
    gint_t   bli_index; \
    f77_int  f77_index; \
\
    /* If the vector is empty, return an index of zero. This early check
       is needed to emulate netlib BLAS. Without it, bli_?amaxv() will
       return 0, which ends up getting incremented to 1 (below) before
       being returned, which is not what we want. */ \
    if ( *n < 1 || *incx <= 0 ) { \
      AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "iamax_: vector empty") \
      return 0;                                   \
    }\
\
    /* If n=1, return 1 here to emulate netlib BLAS and avoid touching vector */ \
    if ( *n == 1 ) { \
      AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "iamax_: n=1") \
      return 1;                                   \
    }\
\
    /* Initialize BLIS. */ \
    bli_init_auto(); \
\
    /* Convert/typecast negative values of n to zero. */ \
    bli_convert_blas_dim1( *n, n0 ); \
\
    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */ \
    bli_convert_blas_incv( n0, (ftype_x*)x, *incx, x0, incx0 ); \
\
    /* Call BLIS interface. */ \
    PASTEMAC2(chx,blisname,BLIS_TAPI_EX_SUF) \
    ( \
      n0, \
      x0, incx0, \
      &bli_index, \
      NULL, \
      NULL  \
    ); \
\
    /* Convert zero-based BLIS (C) index to one-based BLAS (Fortran)
       index. Also, if the BLAS integer size differs from the BLIS
       integer size, that typecast occurs here. */ \
    f77_index = bli_index + 1; \
\
    /* Finalize BLIS. */ \
    bli_finalize_auto(); \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
    return f77_index; \
}\
\
IF_BLIS_ENABLE_BLAS(\
f77_int PASTEF772(i,chx,blasname) \
     ( \
       const f77_int* n, \
       const ftype_x* x, const f77_int* incx  \
     ) \
{ \
  return PASTEF772S(i,chx,blasname)( n, x, incx );\
} \
)

f77_int isamax_blis_impl
     (
       const f77_int* n,
       const float* x, const f77_int* incx
     )
{

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_AMAX_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', *n, *incx);

    dim_t    n0;
    float* x0;
    inc_t    incx0;
    gint_t   bli_index;
    f77_int  f77_index;

    /* If the vector is empty, return an index of zero. This early check
       is needed to emulate netlib BLAS. Without it, bli_?amaxv() will
       return 0, which ends up getting incremented to 1 (below) before
       being returned, which is not what we want. */
    if ( *n < 1 || *incx <= 0 ) {
      AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "isamax_: vector empty");
      return 0;
    }

    /* If n=1, return 1 here to emulate netlib BLAS and avoid touching vector */
    if ( *n == 1 ) {
      AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "iamax_: n=1");
      return 1;
    }

    /* Initialize BLIS. */
//  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else              n0 = ( dim_t )(*n);

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */
    if ( *incx < 0 )
    {
        /* The semantics of negative stride in BLAS are that the vector
        operand be traversed in reverse order. (Another way to think
        of this is that negative strides effectively reverse the order
        of the vector, but without any explicit data movements.) This
        is also how BLIS interprets negative strides. The differences
        is that with BLAS, the caller *always* passes in the 0th (i.e.,
        top-most or left-most) element of the vector, even when the
        stride is negative. By contrast, in BLIS, negative strides are
        used *relative* to the vector address as it is given. Thus, in
        BLIS, if this backwards traversal is desired, the caller *must*
        pass in the address to the (n-1)th (i.e., the bottom-most or
        right-most) element along with a negative stride. */

        x0    = ((float*)x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((float*)x);
        incx0 = ( inc_t )(*incx);
    }

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE)
    {
        cntx_t* cntx = bli_gks_query_cntx();
        samaxv_ker_ft f = bli_cntx_get_l1v_ker_dt(BLIS_FLOAT, BLIS_AMAXV_KER, cntx );
        /* Call BLIS kernel */
        f
        (
            n0,
            x0, incx0,
            &bli_index,
            NULL
        );
    }
    else
    {
      PASTEMAC2(s,amaxv,BLIS_TAPI_EX_SUF)
      (
        n0,
        x0, incx0,
        &bli_index,
        NULL,
        NULL
      );
    }

    /* Convert zero-based BLIS (C) index to one-based BLAS (Fortran)
       index. Also, if the BLAS integer size differs from the BLIS
       integer size, that typecast occurs here. */
    f77_index = bli_index + 1;

    /* Finalize BLIS. */
//    bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return f77_index;
}
#ifdef BLIS_ENABLE_BLAS
f77_int isamax_
     (
       const f77_int* n,
       const float* x, const f77_int* incx
     )
{
  return isamax_blis_impl( n, x, incx ); 
}
#endif
f77_int idamax_blis_impl
     (
       const f77_int* n,
       const double* x, const f77_int* incx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_AMAX_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, *incx);

    dim_t    n0;
    double* x0;
    inc_t    incx0;
    gint_t   bli_index;
    f77_int  f77_index;

    /*
      If the vector is empty, return an index of zero. This early check
      is needed to emulate netlib BLAS. Without it, bli_?amaxv() will
      return 0, which ends up getting incremented to 1 (below) before
      being returned, which is not what we want.
    */
    if ( *n < 1 || *incx <= 0 ) {
      AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "idamax_: vector empty");
      return 0;
    }

    /* If n=1, return 1 here to emulate netlib BLAS and avoid touching vector */
    if ( *n == 1 ) {
      AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "iamax_: n=1");
      return 1;
    }

    /*
      When the length of the vector is one it is going to be the element with
      the maximum absolute value. This early return condition is defined in
      the BLAS standard.
    */
    if(*n == 1)
    {
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return 1;
    }

    /* Initialize BLIS. */
    //  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 )     n0 = ( dim_t )0;
    else              n0 = ( dim_t )(*n);

    /*
      If the input increments are negative, adjust the pointers so we can
      use positive increments instead.
    */
    if ( *incx < 0 )
    {
      /*
        The semantics of negative stride in BLAS are that the vector
        operand be traversed in reverse order. (Another way to think
        of this is that negative strides effectively reverse the order
        of the vector, but without any explicit data movements.) This
        is also how BLIS interprets negative strides. The differences
        is that with BLAS, the caller *always* passes in the 0th (i.e.,
        top-most or left-most) element of the vector, even when the
        stride is negative. By contrast, in BLIS, negative strides are
        used *relative* to the vector address as it is given. Thus, in
        BLIS, if this backwards traversal is desired, the caller *must*
        pass in the address to the (n-1)th (i.e., the bottom-most or
        right-most) element along with a negative stride.
      */

      x0    = ((double*)x) + (n0-1)*(-*incx);
      incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((double*)x);
        incx0 = ( inc_t )(*incx);
    }

    cntx_t* cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    damaxv_ker_ft amaxv_fun_ptr;

    // Pick the kernel based on the architecture ID
    switch (id)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          amaxv_fun_ptr = bli_damaxv_zen_int;
          break;

      default:

          // Query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          amaxv_fun_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_AMAXV_KER, cntx);
    }

    // Call BLIS kernel based on the function pointer
    amaxv_fun_ptr
    (
      n0,
      x0, incx0,
      &bli_index,
      cntx
    );

    /*
      Convert zero-based BLIS (C) index to one-based BLAS (Fortran)
      index. Also, if the BLAS integer size differs from the BLIS
      integer size, that typecast occurs here.
    */
    f77_index = bli_index + 1;

    /* Finalize BLIS. */
    // bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return f77_index;
}
#ifdef BLIS_ENABLE_BLAS
f77_int idamax_
     (
       const f77_int* n,
       const double* x, const f77_int* incx
     )
{
  return idamax_blis_impl( n, x, incx ); 
}
#endif
INSERT_GENTFUNC_BLAS_CZ( amax, amaxv )

