/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

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
#include "test_libblis.h"


// Static variables.
static char*     op_str                    = "hevd";
static char*     o_types                   = "m";   // a
static char*     p_types                   = "u";  // uploa, diaga
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-02, 1e-03 },   // warn, pass for s
                                               { 1e-02, 1e-03 },   // warn, pass for c
                                               { 1e-11, 1e-12 },   // warn, pass for d
                                               { 1e-11, 1e-12 } }; // warn, pass for z

// Local prototypes.
void libblis_test_hevd_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_hevd_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       unsigned int   p_cur,
       double*        perf,
       double*        resid
     );

void libblis_test_hevd_impl
     (
       iface_t   iface,
       obj_t*    a,
       obj_t*    v,
       obj_t*    e
     );

void libblis_test_hevd_check
     (
       test_params_t* params,
       obj_t*         v,
       obj_t*         e,
       obj_t*         a_orig,
       double*        resid
     );



void libblis_test_hevd_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randv) );
	libblis_test_randm( tdata, params, &(op->ops->randm) );
	libblis_test_setv( tdata, params, &(op->ops->setv) );
	libblis_test_normfv( tdata, params, &(op->ops->normfv) );
	libblis_test_normfm( tdata, params, &(op->ops->normfm) );
	libblis_test_subv( tdata, params, &(op->ops->subv) );
	libblis_test_subm( tdata, params, &(op->ops->subm) );
	libblis_test_scalv( tdata, params, &(op->ops->scalv) );
	libblis_test_copym( tdata, params, &(op->ops->copym) );
	libblis_test_scalm( tdata, params, &(op->ops->scalm) );
	libblis_test_hemv( tdata, params, &(op->ops->hemv) );
}



void libblis_test_hevd
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{

	// Return early if this test has already been done.
	if ( libblis_test_op_is_done( op ) ) return;

	// Return early if operation is disabled.
	if ( libblis_test_op_is_disabled( op ) ||
	     libblis_test_l4_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_hevd_deps( tdata, params, op );

	// Execute the test driver for each implementation requested.
	//if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( tdata,
		                        params,
		                        op,
		                        BLIS_TEST_SEQ_FRONT_END,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_hevd_experiment );
	}
}



void libblis_test_hevd_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
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

	num_t        datatype;
	num_t        dt_real;

	dim_t        m;

	obj_t        a, v, e;
	obj_t        a_save;

	uplo_t       uploa;

	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Project the datatype to the real domain.
	dt_real = bli_dt_proj_to_real( datatype );

	// Map the dimension specifier to actual dimensions.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m,       m,       &a );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m,       m,       &a_save );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          'c',       m,       m,       &v );
	libblis_test_vobj_create( params, dt_real, 
	                          'c',       m,                &e );

	// Set the structure and uplo properties of A and A_save.
	bli_obj_set_struc( BLIS_HERMITIAN, &a );
	bli_obj_set_uplo( uploa, &a );

	// Randomize A, load the diagonal.
	libblis_test_mobj_randomize( params, TRUE, &a );
	libblis_test_mobj_load_diag( params, &a );
	bli_setid( &BLIS_ZERO, &a );

	// Zero out the non-stored triangle to make sure the implementation
	// still works only with the stored triangle.
	bli_mktrim( &a );

	// Save A and set its structure and uplo properties.
	bli_copym( &a, &a_save );
	bli_obj_set_struc( BLIS_HERMITIAN, &a_save );
	bli_obj_set_uplo( uploa, &a_save );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &a_save, &a );

		time = bli_clock();

		libblis_test_hevd_impl( iface, &a, &v, &e );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	// f1: Cost of tridiag reduction + FormQ
	// f2: Estimated cost of tridiagonal QR algorithm (assumes two
	//     Francis steps needed per eigenvalue convergence).
	double f1 = ( ( 6.0 / 3.0 ) * m * m * m );
	double f2 = ( ( 3.0 * 2.0 ) * m * m * m );
	if ( bli_obj_is_complex( &a ) ) { f1 *= 4.0; f2 *= 2.0; }
	*perf = ( f1 + f2 ) / time_min / FLOPS_PER_UNIT_PERF;

	// Perform checks.
	libblis_test_hevd_check( params, &v, &e, &a_save, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &a, perf, resid );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &a_save );
	bli_obj_free( &v );
	bli_obj_free( &e );
}



void libblis_test_hevd_impl
     (
       iface_t   iface,
       obj_t*    a,
       obj_t*    v,
       obj_t*    e
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
			bli_hevd( a, v, e );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_hevd_check
     (
       test_params_t* params,
       obj_t*         v,
       obj_t*         e,
       obj_t*         a_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( v );
	num_t  dt_real = bli_obj_dt_proj_to_real( v );

	dim_t  m       = bli_obj_length( v );

	uplo_t uploa   = bli_obj_uplo( a_orig );

	obj_t  norm;

	obj_t  ainv, x, b, z;

	double junk;

	//
	// Pre-conditions:
	// - a_orig is randomized and Hermitian.
	// - x is randomized.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   [V,e] := hevd(A_orig)
	//
	// is functioning correctly if
	//
	//   normfm( z - x )
	//
	// is negligible, where
	//
    //   z = A_orig * ( V * I*inv(e) * V^H ) * x
    //     = A_orig * A_inv * x
    //     = A_orig * b
	//

	bli_obj_scalar_init_detached( dt_real, &norm );

	bli_obj_create( dt, m, m, 0, 0, &ainv );
	bli_obj_set_struc( BLIS_HERMITIAN, &ainv );
	bli_obj_set_uplo( uploa, &ainv );

	bli_obj_create( dt, m, 1, 0, 0, &x );
	bli_obj_create( dt, m, 1, 0, 0, &b );
	bli_obj_create( dt, m, 1, 0, 0, &z );

	libblis_test_vobj_randomize( params, TRUE, &x );

	bli_inverttv( 1.0e-13, e );
	bli_rhevd( v, e, &ainv );
	bli_mkherm( &ainv );

	bli_hemv( &BLIS_ONE, &ainv, &x, &BLIS_ZERO, &b );
	bli_hemv( &BLIS_ONE, a_orig, &b, &BLIS_ZERO, &z );

	bli_subm( &z, &x );
	bli_normfm( &x, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &ainv );
	bli_obj_free( &x );
	bli_obj_free( &b );
	bli_obj_free( &z );
}

