/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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
static char*     op_str                    = "scalm";
static char*     o_types                   = "m";  // x
static char*     p_types                   = "c";  // conjalpha
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_scalm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_scalm_experiment
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

void libblis_test_scalm_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    y
     );

void libblis_test_scalm_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         y,
       obj_t*         y_save,
       double*        resid
     );



void libblis_test_scalm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randm( tdata, params, &(op->ops->randm) );
	libblis_test_normfm( tdata, params, &(op->ops->normfm) );
	libblis_test_copym( tdata, params, &(op->ops->copym) );
}



void libblis_test_scalm
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
	     libblis_test_l1m_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_scalm_deps( tdata, params, op );

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
		                        libblis_test_scalm_experiment );
	}
}



void libblis_test_scalm_experiment
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

	dim_t        m, n;

	conj_t       conjalpha;

	obj_t        alpha, y;
	obj_t        y_save;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to actual dimensions.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );
	n = libblis_test_get_dim_from_prob_size( op->dim_spec[1], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_conj( pc_str[0], &conjalpha );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &y );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &y_save );

	// Set alpha to 0 + i.
	//bli_setsc( 0.0, 1.0, &alpha );
	if ( bli_obj_is_real( &y ) )
		bli_setsc( -2.0,  0.0, &alpha );
	else
		bli_setsc(  0.0, -2.0, &alpha );

	// Randomize and save y.
	libblis_test_mobj_randomize( params, FALSE, &y );
	bli_copym( &y, &y_save );

	// Apply the parameters.
	bli_obj_set_conj( conjalpha, &alpha );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &y_save, &y );

		time = bli_clock();

		libblis_test_scalm_impl( iface, &alpha, &y );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 1.0 * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &y ) ) *perf *= 6.0;

	// Perform checks.
	libblis_test_scalm_check( params, &alpha, &y, &y_save, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &y, perf, resid );

	// Free the test objects.
	bli_obj_free( &y );
	bli_obj_free( &y_save );
}



void libblis_test_scalm_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    y
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_scalm( alpha, y );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_scalm_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         y,
       obj_t*         y_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( y );
	num_t  dt_real = bli_obj_dt_proj_to_real( y );

	dim_t  m       = bli_obj_length( y );
	dim_t  n       = bli_obj_width( y );

	obj_t  norm_y_r;
	obj_t  nalpha;

	obj_t  y2;

	double junk;

	//
	// Pre-conditions:
	// - y_orig is randomized.
	// Note:
	// - alpha should have a non-zero imaginary component in the complex
	//   cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   y := conjalpha(alpha) * y_orig
	//
	// is functioning correctly if
	//
	//   normfm( y + -conjalpha(alpha) * y_orig )
	//
	// is negligible.
	//

	bli_obj_create( dt, m, n, 0, 0, &y2 );
	bli_copym( y_orig, &y2 );

	bli_obj_scalar_init_detached( dt,      &nalpha );
	bli_obj_scalar_init_detached( dt_real, &norm_y_r );

	bli_copysc( alpha, &nalpha );
	bli_mulsc( &BLIS_MINUS_ONE, &nalpha );

	bli_scalm( &nalpha, &y2 );
	bli_addm( &y2, y );
	
	bli_normfm( y, &norm_y_r );

	bli_getsc( &norm_y_r, resid, &junk );

	bli_obj_free( &y2 );
}

