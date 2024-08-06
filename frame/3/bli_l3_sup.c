/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

err_t bli_gemmsup
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);

    // Return early if small matrix handling is disabled at configure-time.
    #ifdef BLIS_DISABLE_SUP_HANDLING
    AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP is Disabled.");
    return BLIS_FAILURE;
    #endif

    // Return early if this is a mixed-datatype computation.
    if ( bli_obj_dt( c ) != bli_obj_dt( a ) ||
	 bli_obj_dt( c ) != bli_obj_dt( b ) ||
	 bli_obj_comp_prec( c ) != bli_obj_prec( c ) ) {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support Mixed datatypes.");
	return BLIS_FAILURE;
    }


    const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

    /*General stride is not yet supported in sup*/
    if(BLIS_XXX==stor_id) {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support general stride.");
	return BLIS_FAILURE;
    }

    trans_t transa = bli_obj_conjtrans_status( a );
    trans_t transb = bli_obj_conjtrans_status( b );


    //Don't use sup for currently unsupported storage types in cgemmsup
    if(bli_obj_is_scomplex(c) &&
    (((stor_id == BLIS_RRC)||(stor_id == BLIS_CRC))
    || ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE))
    || ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE))
    )){
	//printf(" gemmsup: Returning with for un-supported storage types and conjugate property in cgemmsup \n");
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for cgemm");
	return BLIS_FAILURE;
    }

    //Don't use sup for currently unsupported storage types  in zgemmsup
    if(bli_obj_is_dcomplex(c) &&
    (((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE)) ||
     ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE))
    )){
	//printf(" gemmsup: Returning with for un-supported storage types and conjugate property in zgemmsup \n");
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for zgemm.");
	return BLIS_FAILURE;
    }


    // Obtain a valid context from the gks if necessary.
    // NOTE: This must be done before calling the _check() function, since
    // that function assumes the context pointer is valid.
    if ( cntx == NULL ) cntx = bli_gks_query_cntx();

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    rntm_t rntm_l;
    if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
    else                { rntm_l = *rntm;                       rntm = &rntm_l; }

#if defined(BLIS_FAMILY_ZEN5) || defined(BLIS_FAMILY_ZEN4) || defined(BLIS_FAMILY_AMDZEN) || defined(BLIS_FAMILY_X86_64)

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    if((id == BLIS_ARCH_ZEN5) || (id == BLIS_ARCH_ZEN4))
    {
        if(( bli_obj_dt(a) == BLIS_DOUBLE ) || ( bli_obj_dt(a) == BLIS_DCOMPLEX))
        {
            // Pack A to avoid RD kernels.
            if((stor_id == BLIS_CRC || stor_id == BLIS_RRC))
            {
                bli_rntm_set_pack_a(1, rntm);//packa
            }
        }
    }
#endif

#ifdef AOCL_DYNAMIC
    // Calculating optimal nt and corresponding factorization (ic,jc) here, so
    // as to determine the matrix dimensions (A - m, B - n) per thread. This
    // can be used to check if dimensions per thread falls under the SUP
    // threshold and potentially move some of the native path gemm to SUP path
    // in multi-threaded scenario.
    err_t smart_threading = bli_smart_threading_sup( a, b, c, BLIS_GEMM, rntm, cntx );

    if ( smart_threading != BLIS_SUCCESS )
    {
        thresh_func_ft func_fp;
        func_fp = bli_cntx_get_l3_thresh_func(BLIS_GEMM, cntx);

        // Return early if the sizes are beyond SUP thresholds
        if ( !func_fp( a, b, c, cntx ) )
        {
            AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2,
                            "SUP - Sizes are beyond SUP thresholds.");
            return BLIS_FAILURE;
        }
    }
#else
    thresh_func_ft func_fp;

    func_fp = bli_cntx_get_l3_thresh_func(BLIS_GEMM, cntx);

    // Return early if the sizes are beyond SUP thresholds
        if ( !func_fp( a, b, c, cntx ) ) {
            AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Sizes are beyond SUP thresholds.");
            return BLIS_FAILURE;
    }
#endif
    // We've now ruled out the following two possibilities:
    // - the ukernel prefers the operation as-is, and the sup thresholds are
    //   unsatisfied.
    // - the ukernel prefers a transposed operation, and the sup thresholds are
    //   unsatisfied after taking into account the transposition.
    // This implies that the sup thresholds (at least one of them) are met.
    // and the small/unpacked handler should be called.
    // NOTE: The sup handler is free to enforce a stricter threshold regime
    // if it so chooses, in which case it can/should return BLIS_FAILURE.

    // Query the small/unpacked handler from the context and invoke it.
    gemmsup_oft gemmsup_fp = bli_cntx_get_l3_sup_handler( BLIS_GEMM, cntx );

    err_t ret_gemmsup_fp = gemmsup_fp
    (
      alpha,
      a,
      b,
      beta,
      c,
      cntx,
      rntm
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ret_gemmsup_fp;
}

err_t bli_gemmtsup
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);
//  AOCL_DTL_LOG_GEMMT_INPUTS(AOCL_DTL_LEVEL_TRACE_2, alpha, a, b, beta, c);

    // Return early if small matrix handling is disabled at configure-time.
    #ifdef BLIS_DISABLE_SUP_HANDLING
    AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP is Disabled.");
    return BLIS_FAILURE;
    #endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    if (bli_cpuid_is_avx2fma3_supported() == FALSE){
        AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "AVX instruction is not supported");
        return BLIS_FAILURE;
    }
#else
    return BLIS_FAILURE;
#endif

    // Return early if this is a mixed-datatype computation.
    if ( bli_obj_dt( c ) != bli_obj_dt( a ) ||
	 bli_obj_dt( c ) != bli_obj_dt( b ) ||
	 bli_obj_comp_prec( c ) != bli_obj_prec( c ) ) {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support Mixed datatypes.");
	return BLIS_FAILURE;
    }


    const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

    /*General stride is not yet supported in sup*/
    if(BLIS_XXX==stor_id) {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support general stride.");
	return BLIS_FAILURE;
    }

    trans_t transa = bli_obj_conjtrans_status( a );
    trans_t transb = bli_obj_conjtrans_status( b );

    //Don't use sup for currently unsupported storage types in cgemmsup
    if(bli_obj_is_scomplex(c) &&
    (((stor_id == BLIS_RRC)||(stor_id == BLIS_CRC))
    || ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE))
    || ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE))
    )){
	//printf(" gemmtsup: Returning with for un-supported storage types and conjugate property in cgemmtsup \n");
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for cgemmt");
	return BLIS_FAILURE;
    }

    //Don't use sup for currently unsupported storage types  in zgemmsup
    if(bli_obj_is_dcomplex(c) &&
    (((stor_id == BLIS_RRC)||(stor_id == BLIS_CRC))
    || ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE))
    || ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE))
    )){
	//printf(" gemmtsup: Returning with for un-supported storage types and conjugate property in zgemmtsup \n");
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for zgemmt.");
	return BLIS_FAILURE;
    }


    // Obtain a valid (native) context from the gks if necessary.
    // NOTE: This must be done before calling the _check() function, since
    // that function assumes the context pointer is valid.
    if ( cntx == NULL ) cntx = bli_gks_query_cntx();

    thresh_func_ft func_fp;

    func_fp = bli_cntx_get_l3_thresh_func(BLIS_GEMMT, cntx);

    if ( !func_fp( a, b, c, cntx ) )
    {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Sizes beyond SUP thresholds.");
	return BLIS_FAILURE;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    rntm_t rntm_l;
    if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
    else                { rntm_l = *rntm;                       rntm = &rntm_l; }

#ifdef AOCL_DYNAMIC
	// If dynamic-threading is enabled, calculate optimum number
	// of threads and update in rntm

    // Limit the number of thread for smaller sizes.
    bli_nthreads_optimum( a, b, c, BLIS_GEMMT, rntm );
#endif

#if 0
const num_t dt = bli_obj_dt( c );
const dim_t m  = bli_obj_length( c );
const dim_t n  = bli_obj_width( c );
const dim_t k  = bli_obj_width_after_trans( a );
const dim_t tm = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_MT, cntx );
const dim_t tn = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_NT, cntx );
const dim_t tk = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_KT, cntx );

printf( "dims: %d %d %d (threshs: %d %d %d)\n",
	(int)m, (int)n, (int)k, (int)tm, (int)tn, (int)tk );
#endif

    // We've now ruled out the following two possibilities:
    // - the ukernel prefers the operation as-is, and the sup thresholds are
    // unsatisfied.
    // - the ukernel prefers a transposed operation, and the sup thresholds are
    //   unsatisfied after taking into account the transposition.
    // This implies that the sup thresholds (at least one of them) are met.
    // and the small/unpacked handler should be called.
    // NOTE: The sup handler is free to enforce a stricter threshold regime
    // if it so chooses, in which case it can/should return BLIS_FAILURE.

    // Query the small/unpacked handler from the context and invoke it.
    gemmtsup_oft gemmtsup_fp = bli_cntx_get_l3_sup_handler( BLIS_GEMMT, cntx );

    err_t ret_gemmtsup_fp = gemmtsup_fp
    (
      alpha,
      a,
      b,
      beta,
      c,
      cntx,
      rntm
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ret_gemmtsup_fp;
}

err_t bli_syrksup
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);

    // Return early if small matrix handling is disabled at configure-time.
#ifdef BLIS_DISABLE_SUP_HANDLING
    AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP is Disabled.");
    return BLIS_FAILURE;
#endif

    // Return early if this is a mixed-datatype computation.
    if ( bli_obj_dt( c ) != bli_obj_dt( a ) ||
	 bli_obj_comp_prec( c ) != bli_obj_prec( c ) )
    {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support Mixed datatypes.");
	return BLIS_FAILURE;
    }

    obj_t at_local;

    // For syrk, the right-hand "B" operand is simply A^T.
    bli_obj_alias_to( a, &at_local );
    bli_obj_induce_trans( &at_local );

    const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, &at_local );

    /*General stride is not yet supported in sup*/
    if(BLIS_XXX==stor_id) {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP doesn't support general stride.");
	return BLIS_FAILURE;
    }

    trans_t transa = bli_obj_conjtrans_status( a );

    //Don't use sup for currently unsupported storage types in cgemmsup
    if(bli_obj_is_scomplex(c) &&
    (((stor_id == BLIS_RRC)||(stor_id == BLIS_CRC))
    || ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE))
    )){
	//printf(" syrksup: Returning with for un-supported storage types and conjugate property in csyrksup \n");
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for csyrk");
	return BLIS_FAILURE;
    }

    //Don't use sup for currently unsupported storage types  in zgemmsup
    if(bli_obj_is_dcomplex(c) &&
    (((stor_id == BLIS_RRC)||(stor_id == BLIS_CRC))
    || ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE))
    )){
	//printf(" syrksup: Returning with for un-supported storage types and conjugate property in zsyrksup \n");
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - Unsuppported storage type for zsyrk.");
	return BLIS_FAILURE;
    }


    // Obtain a valid (native) context from the gks if necessary.
    // NOTE: This must be done before calling the _check() function, since
    // that function assumes the context pointer is valid.
    if ( cntx == NULL ) cntx = bli_gks_query_cntx();

    thresh_func_ft func_fp = bli_cntx_get_l3_thresh_func(BLIS_SYRK, cntx);
    if( !func_fp( a, &at_local, c, cntx))
    {
	AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_2, "SUP - sizes beyond SUP thresholds.");
	return BLIS_FAILURE;
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    rntm_t rntm_l;
    if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
    else                { rntm_l = *rntm;                       rntm = &rntm_l; }

#ifdef AOCL_DYNAMIC // Will change this name later to BLIS_SMART_THREAD
  // If dynamic-threading is enabled, calculate optimum
  // number of threads.
  // rntm will be updated with optimum number of threads.
  bli_nthreads_optimum( a, &at_local, c, BLIS_SYRK, rntm );
#endif

#if 0
const num_t dt = bli_obj_dt( c );
const dim_t m  = bli_obj_length( c );
const dim_t n  = bli_obj_width( c );
const dim_t k  = bli_obj_width_after_trans( a );
const dim_t tm = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_MT, cntx );
const dim_t tn = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_NT, cntx );
const dim_t tk = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_KT, cntx );

printf( "dims: %d %d %d (threshs: %d %d %d)\n",
	(int)m, (int)n, (int)k, (int)tm, (int)tn, (int)tk );
#endif

    // We've now ruled out the following two possibilities:
    // - the ukernel prefers the operation as-is, and the sup thresholds are
    // unsatisfied.
    // - the ukernel prefers a transposed operation, and the sup thresholds are
    //   unsatisfied after taking into account the transposition.
    // This implies that the sup thresholds (at least one of them) are met.
    // and the small/unpacked handler should be called.
    // NOTE: The sup handler is free to enforce a stricter threshold regime
    // if it so chooses, in which case it can/should return BLIS_FAILURE.

    // Query the small/unpacked handler from the context and invoke it.
    gemmtsup_oft gemmtsup_fp = bli_cntx_get_l3_sup_handler( BLIS_GEMMT, cntx );

    err_t ret_gemmtsup_fp = gemmtsup_fp
    (
      alpha,
      a,
      &at_local,
      beta,
      c,
      cntx,
      rntm
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
    return ret_gemmtsup_fp;
}
