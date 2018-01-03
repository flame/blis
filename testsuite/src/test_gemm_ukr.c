/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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
#include "test_libblis.h"


// Static variables.
static char*     op_str                    = "gemm_ukr";
static char*     o_types                   = "m"; // c
static char*     p_types                   = "";
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_gemm_ukr_deps
     (
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_gemm_ukr_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       num_t          datatype,
       char*          pc_str,
       char*          sc_str,
       unsigned int   p_cur,
       double*        perf,
       double*        resid
     );

void libblis_test_gemm_ukr_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    b,
       obj_t*    beta,
       obj_t*    c,
       cntx_t*   cntx
     );

void libblis_test_gemm_ukr_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig,
       double*        resid
     );



void libblis_test_gemm_ukr_deps
     (
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( params, &(op->ops->randv) );
	libblis_test_randm( params, &(op->ops->randm) );
	libblis_test_setv( params, &(op->ops->setv) );
	libblis_test_normfv( params, &(op->ops->normfv) );
	libblis_test_subv( params, &(op->ops->subv) );
	libblis_test_scalv( params, &(op->ops->scalv) );
	libblis_test_copym( params, &(op->ops->copym) );
	libblis_test_scalm( params, &(op->ops->scalm) );
	libblis_test_gemv( params, &(op->ops->gemv) );
}



void libblis_test_gemm_ukr
     (
       test_params_t* params,
       test_op_t*     op
     )
{

	// Return early if this test has already been done.
	if ( op->test_done == TRUE ) return;

	// Return early if operation is disabled.
	if ( op->op_switch == DISABLE_ALL ||
	     op->ops->l3ukr_over == DISABLE_ALL ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_gemm_ukr_deps( params, op );

	// Execute the test driver for each implementation requested.
	if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( params,
		                        op,
		                        BLIS_TEST_SEQ_UKERNEL,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_gemm_ukr_experiment );
	}
}



void libblis_test_gemm_ukr_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       num_t          datatype,
       char*          pc_str,
       char*          sc_str,
       unsigned int   p_cur,
       double*        perf,
       double*        resid
     )
{
	unsigned int n_repeats = params->n_repeats;
	unsigned int i;

	double       time_min  = DBL_MAX;
	double       time;

	dim_t        m, n, k;
	inc_t        ldap, ldbp;

	char         sc_a = 'c';
	char         sc_b = 'r';

	obj_t        alpha, a, b, beta, c;
	obj_t        ap, bp;
	obj_t        c_save;

	cntx_t       cntx;

	// Initialize a context.
	bli_gemm_cntx_init( datatype, &cntx );

	// Map the dimension specifier to actual dimensions.
	k = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Fix m and n to MR and NR, respectively.
	m = bli_cntx_get_blksz_def_dt( datatype, BLIS_MR, &cntx );
	n = bli_cntx_get_blksz_def_dt( datatype, BLIS_NR, &cntx );

	// Also query PACKMR and PACKNR as the leading dimensions to ap and bp,
	// respectively.
	ldap = bli_cntx_get_blksz_max_dt( datatype, BLIS_MR, &cntx );
	ldbp = bli_cntx_get_blksz_max_dt( datatype, BLIS_NR, &cntx );

	// Store the register blocksizes so that the driver can retrieve the
	// values later when printing results.
	op->dim_aux[0] = m;
	op->dim_aux[1] = n;

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );
	bli_obj_scalar_init_detached( datatype, &beta );

	// Create test operands.
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_a,      m, k, &a );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_b,      k, n, &b );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &c );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &c_save );

	// Set alpha and beta.
	if ( bli_obj_is_real( c ) )
	{
		bli_setsc(  1.2,  0.0, &alpha );
		bli_setsc( -1.0,  0.0, &beta );
		//bli_setsc( 0.0,  0.0, &beta );
	}
	else
	{
		bli_setsc(  1.2,  0.8, &alpha );
		bli_setsc( -1.0,  0.5, &beta );
	}

	// Randomize A, B, and C, and save C.
	libblis_test_mobj_randomize( params, TRUE, &a );
	libblis_test_mobj_randomize( params, TRUE, &b );
	libblis_test_mobj_randomize( params, TRUE, &c );
	bli_copym( &c, &c_save );

#if 0
	// Create pack objects for a and b, and pack them to ap and bp,
	// respectively.
	cntl_t* cntl_a = libblis_test_pobj_create
	(
	  BLIS_MR,
	  BLIS_KR,
	  BLIS_NO_INVERT_DIAG,
	  BLIS_PACKED_ROW_PANELS,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &a, &ap,
	  &cntx
	);
	cntl_t* cntl_b = libblis_test_pobj_create
	(
	  BLIS_KR,
	  BLIS_NR,
	  BLIS_NO_INVERT_DIAG,
	  BLIS_PACKED_COL_PANELS,
	  BLIS_BUFFER_FOR_B_PANEL,
	  &b, &bp,
	  &cntx
	);
#endif

	// Create the packed objects. Use packmr and packnr as the leading
	// dimensions of ap and bp, respectively.
	bli_obj_create( datatype, m, k, 1, ldap, &ap );
	bli_obj_create( datatype, k, n, ldbp, 1, &bp );

	// Set up the objects for packing. Calling packm_init_pack() does everything
	// except checkout a memory pool block and save its address to the obj_t's.
	// However, it does overwrite the buffer field of packed object with that of
	// the source object. So, we have to save the buffer address that was
	// allocated.
	void* buf_ap = bli_obj_buffer( ap );
	void* buf_bp = bli_obj_buffer( bp );
	bli_packm_init_pack( BLIS_NO_INVERT_DIAG, BLIS_PACKED_ROW_PANELS,
	                     BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
	                     BLIS_MR, BLIS_KR, &a, &ap, &cntx );
	bli_packm_init_pack( BLIS_NO_INVERT_DIAG, BLIS_PACKED_COL_PANELS,
	                     BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
	                     BLIS_KR, BLIS_NR, &b, &bp, &cntx );
	bli_obj_set_buffer( buf_ap, ap );
	bli_obj_set_buffer( buf_bp, bp );

	// Pack the data from the source objects.
	bli_packm_blk_var1( &a, &ap, &cntx, NULL, &BLIS_PACKM_SINGLE_THREADED );
	bli_packm_blk_var1( &b, &bp, &cntx, NULL, &BLIS_PACKM_SINGLE_THREADED );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &c_save, &c );

		time = bli_clock();

		libblis_test_gemm_ukr_impl( iface,
		                            &alpha, &ap, &bp, &beta, &c,
		                            &cntx );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * n * k ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( c ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_gemm_ukr_check( params, &alpha, &a, &b, &beta, &c, &c_save, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &c, perf, resid );

#if 0
	// Free the control tree nodes and release their cached mem_t entries
	// back to the memory broker.
	bli_cntl_free( cntl_a, &BLIS_PACKM_SINGLE_THREADED );
	bli_cntl_free( cntl_b, &BLIS_PACKM_SINGLE_THREADED );
#endif

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &c_save );

	// Finalize the context.
	bli_gemm_cntx_finalize( &cntx );
}



void libblis_test_gemm_ukr_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    b,
       obj_t*    beta,
       obj_t*    c,
       cntx_t*   cntx
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_UKERNEL:
		bli_gemm_ukernel( alpha, a, b, beta, c, cntx );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_gemm_ukr_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_datatype( *c );
	num_t  dt_real = bli_obj_datatype_proj_to_real( *c );

	dim_t  m       = bli_obj_length( *c );
	dim_t  n       = bli_obj_width( *c );
	dim_t  k       = bli_obj_width( *a );

	obj_t  norm;
	obj_t  t, v, w, z;

	double junk;

	//
	// Pre-conditions:
	// - a is randomized.
	// - b is randomized.
	// - c_orig is randomized.
	// Note:
	// - alpha and beta should have non-zero imaginary components in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   C := beta * C_orig + alpha * A * B
	//
	// is functioning correctly if
	//
	//   normf( v - z )
	//
	// is negligible, where
	//
	//   v = C * t
	//   z = ( beta * C_orig + alpha * A * B ) * t
	//     = beta * C_orig * t + alpha * A * B * t
	//     = beta * C_orig * t + alpha * A * w
	//     = beta * C_orig * t + z
	//

	bli_obj_scalar_init_detached( dt_real, &norm );

	bli_obj_create( dt, n, 1, 0, 0, &t );
	bli_obj_create( dt, m, 1, 0, 0, &v );
	bli_obj_create( dt, k, 1, 0, 0, &w );
	bli_obj_create( dt, m, 1, 0, 0, &z );

	libblis_test_vobj_randomize( params, TRUE, &t );

	bli_gemv( &BLIS_ONE, c, &t, &BLIS_ZERO, &v );

	bli_gemv( &BLIS_ONE, b, &t, &BLIS_ZERO, &w );
	bli_gemv( alpha, a, &w, &BLIS_ZERO, &z );
	bli_gemv( beta, c_orig, &t, &BLIS_ONE, &z );

	bli_subv( &z, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &t );
	bli_obj_free( &v );
	bli_obj_free( &w );
	bli_obj_free( &z );
}

