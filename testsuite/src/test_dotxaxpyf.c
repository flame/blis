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
static char*     op_str                    = "dotxaxpyf";
static char*     o_types                   = "mvvvv";  // A w x y z
static char*     p_types                   = "cccc";   // conjat conja conjw conjx
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_dotxaxpyf_deps
     (
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_dotxaxpyf_experiment
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

void libblis_test_dotxaxpyf_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    at,
       obj_t*    a,
       obj_t*    w,
       obj_t*    x,
       obj_t*    beta,
       obj_t*    y,
       obj_t*    z,
       cntx_t*   cntx
     );

void libblis_test_dotxaxpyf_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         at,
       obj_t*         a,
       obj_t*         w,
       obj_t*         x,
       obj_t*         beta,
       obj_t*         y,
       obj_t*         z,
       obj_t*         y_orig,
       obj_t*         z_orig,
       double*        resid
     );



void libblis_test_dotxaxpyf_deps
     (
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( params, &(op->ops->randv) );
	libblis_test_randm( params, &(op->ops->randm) );
	libblis_test_normfv( params, &(op->ops->normfv) );
	libblis_test_subv( params, &(op->ops->subv) );
	libblis_test_copyv( params, &(op->ops->copyv) );
	libblis_test_axpyv( params, &(op->ops->axpyv) );
	libblis_test_dotxv( params, &(op->ops->dotxv) );
}



void libblis_test_dotxaxpyf
     (
       test_params_t* params,
       test_op_t*     op
     )
{

	// Return early if this test has already been done.
	if ( op->test_done == TRUE ) return;

	// Return early if operation is disabled.
	if ( op->op_switch == DISABLE_ALL ||
	     op->ops->l1f_over == DISABLE_ALL ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_dotxaxpyf_deps( params, op );

	// Execute the test driver for each implementation requested.
	if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( params,
		                        op,
		                        BLIS_TEST_SEQ_FRONT_END,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_dotxaxpyf_experiment );
	}
}



void libblis_test_dotxaxpyf_experiment
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

	dim_t        m, b_n;

	conj_t       conjat, conja, conjw, conjx;

	obj_t        alpha, at, a, w, x, beta, y, z;
	obj_t        y_save, z_save;

	cntx_t       cntx;

	// Initialize a context.
	bli_dotxaxpyf_cntx_init( datatype, &cntx );

	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Query the operation's fusing factor for the current datatype.
	b_n = bli_cntx_get_blksz_def_dt( datatype, BLIS_XF, &cntx );

	// Store the fusing factor so that the driver can retrieve the value
	// later when printing results.
	op->dim_aux[0] = b_n;

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_conj( pc_str[0], &conjat );
	bli_param_map_char_to_blis_conj( pc_str[1], &conja );
	bli_param_map_char_to_blis_conj( pc_str[2], &conjw );
	bli_param_map_char_to_blis_conj( pc_str[3], &conjx );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );
	bli_obj_scalar_init_detached( datatype, &beta );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                                            sc_str[0], m, b_n, &a );
	libblis_test_vobj_create( params, datatype, sc_str[1], m, &w );
	libblis_test_vobj_create( params, datatype, sc_str[2], b_n, &x );
	libblis_test_vobj_create( params, datatype, sc_str[3], b_n, &y );
	libblis_test_vobj_create( params, datatype, sc_str[3], b_n, &y_save );
	libblis_test_vobj_create( params, datatype, sc_str[4], m, &z );
	libblis_test_vobj_create( params, datatype, sc_str[4], m, &z_save );

	// Set alpha.
	if ( bli_obj_is_real( y ) )
	{
		bli_setsc(  1.2,  0.0, &alpha );
		bli_setsc( -1.0,  0.0, &beta );
	}
	else
	{
		bli_setsc(  1.2,  0.1, &alpha );
		bli_setsc( -1.0, -0.1, &beta );
	}

	// Randomize A, w, x, y, and z, and save y and z.
	libblis_test_mobj_randomize( params, FALSE, &a );
	libblis_test_vobj_randomize( params, FALSE, &w );
	libblis_test_vobj_randomize( params, FALSE, &x );
	libblis_test_vobj_randomize( params, FALSE, &y );
	libblis_test_vobj_randomize( params, FALSE, &z );
	bli_copyv( &y, &y_save );
	bli_copyv( &z, &z_save );

	// Create an alias to a for at. (Note that it should NOT actually be
	// marked for transposition since the transposition is part of the dotxf
	// subproblem.)
	bli_obj_alias_to( a, at );

	// Apply the parameters.
	bli_obj_set_conj( conjat, at );
	bli_obj_set_conj( conja, a );
	bli_obj_set_conj( conjw, w );
	bli_obj_set_conj( conjx, x );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copyv( &y_save, &y );
		bli_copyv( &z_save, &z );

		time = bli_clock();

		libblis_test_dotxaxpyf_impl( iface,
		                             &alpha, &at, &a, &w, &x, &beta, &y, &z,
		                             &cntx );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * b_n + 2.0 * m * b_n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( y ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_dotxaxpyf_check( params, &alpha, &at, &a, &w, &x, &beta, &y, &z, &y_save, &z_save, resid );

	// Zero out performance and residual if either output vector is empty.
	libblis_test_check_empty_problem( &y, perf, resid );
	libblis_test_check_empty_problem( &z, perf, resid );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &w );
	bli_obj_free( &x );
	bli_obj_free( &y );
	bli_obj_free( &z );
	bli_obj_free( &y_save );
	bli_obj_free( &z_save );

	// Finalize the context.
	bli_dotxaxpyf_cntx_finalize( &cntx );
}



void libblis_test_dotxaxpyf_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    at,
       obj_t*    a,
       obj_t*    w,
       obj_t*    x,
       obj_t*    beta,
       obj_t*    y,
       obj_t*    z,
       cntx_t*   cntx
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_dotxaxpyf_ex( alpha, at, a, w, x, beta, y, z, cntx );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_dotxaxpyf_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         at,
       obj_t*         a,
       obj_t*         w,
       obj_t*         x,
       obj_t*         beta,
       obj_t*         y,
       obj_t*         z,
       obj_t*         y_orig,
       obj_t*         z_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_datatype( *y );
	num_t  dt_real = bli_obj_datatype_proj_to_real( *y );

	dim_t  m       = bli_obj_vector_dim( *z );
	dim_t  b_n     = bli_obj_vector_dim( *y );

	dim_t  i;

	obj_t  a1, chi1, psi1, v, q;
	obj_t  alpha_chi1;
	obj_t  norm;

	double resid1, resid2;
	double junk;

	//
	// Pre-conditions:
	// - a is randomized.
	// - w is randomized.
	// - x is randomized.
	// - y is randomized.
	// - z is randomized.
	// - at is an alias to a.
	// Note:
	// - alpha and beta should have a non-zero imaginary component in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   y := beta * y_orig + alpha * conjat(A^T) * conjw(w)
	//   z :=        z_orig + alpha * conja(A)    * conjx(x)
	//
	// is functioning correctly if
	//
	//   normf( y - v )
	//
	// and
	//
	//   normf( z - q )
	//
	// are negligible, where v and q contain y and z as computed by repeated
	// calls to dotxv and axpyv, respectively.
	//

	bli_obj_scalar_init_detached( dt_real, &norm );
	bli_obj_scalar_init_detached( dt,      &alpha_chi1 );

	bli_obj_create( dt, b_n, 1, 0, 0, &v );
	bli_obj_create( dt, m,   1, 0, 0, &q );

	bli_copyv( y_orig, &v );
	bli_copyv( z_orig, &q );

	// v := beta * v + alpha * conjat(at) * conjw(w)
	for ( i = 0; i < b_n; ++i )
	{
		bli_acquire_mpart_l2r( BLIS_SUBPART1, i, 1, at, &a1 );
		bli_acquire_vpart_f2b( BLIS_SUBPART1, i, 1, &v, &psi1 );

		bli_dotxv( alpha, &a1, w, beta, &psi1 );
	}

	// q := q + alpha * conja(a) * conjx(x)
	for ( i = 0; i < b_n; ++i )
	{
		bli_acquire_mpart_l2r( BLIS_SUBPART1, i, 1, a, &a1 );
		bli_acquire_vpart_f2b( BLIS_SUBPART1, i, 1, x, &chi1 );

		bli_copysc( &chi1, &alpha_chi1 );
		bli_mulsc( alpha, &alpha_chi1 );

		bli_axpyv( &alpha_chi1, &a1, &q );
	}


	bli_subv( y, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, &resid1, &junk );

	bli_subv( z, &q );
	bli_normfv( &q, &norm );
	bli_getsc( &norm, &resid2, &junk );


	*resid = bli_fmaxabs( resid1, resid2 );

	bli_obj_free( &v );
	bli_obj_free( &q );
}

