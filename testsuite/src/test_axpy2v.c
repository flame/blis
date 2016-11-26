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
static char*     op_str                    = "axpy2v";
static char*     o_types                   = "vvv";  // x y z
static char*     p_types                   = "cc";   // conjx conjy
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_axpy2v_deps
     (
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_axpy2v_experiment
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

void libblis_test_axpy2v_impl
     (
       iface_t   iface,
       obj_t*    alpha1,
       obj_t*    alpha2,
       obj_t*    x,
       obj_t*    y,
       obj_t*    z,
       cntx_t*   cntx
     );

void libblis_test_axpy2v_check
     (
       test_params_t* params,
       obj_t*         alpha1,
       obj_t*         alpha2,
       obj_t*         x,
       obj_t*         y,
       obj_t*         z,
       obj_t*         z_orig,
       double*        resid
     );



void libblis_test_axpy2v_deps
     (
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( params, &(op->ops->randv) );
	libblis_test_normfv( params, &(op->ops->normfv) );
	libblis_test_addv( params, &(op->ops->addv) );
	libblis_test_subv( params, &(op->ops->subv) );
	libblis_test_copyv( params, &(op->ops->copyv) );
	libblis_test_scalv( params, &(op->ops->scalv) );
}



void libblis_test_axpy2v
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
	if ( TRUE ) libblis_test_axpy2v_deps( params, op );

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
		                        libblis_test_axpy2v_experiment );
	}
}



void libblis_test_axpy2v_experiment
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

	dim_t        m;

	conj_t       conjx, conjy;

	obj_t        alpha1, alpha2, x, y, z;
	obj_t        z_save;

	cntx_t       cntx;

	// Initialize a context.
	bli_axpy2v_cntx_init( datatype, &cntx );

	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_conj( pc_str[0], &conjx );
	bli_param_map_char_to_blis_conj( pc_str[1], &conjy );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha1 );
	bli_obj_scalar_init_detached( datatype, &alpha2 );

	// Create test operands (vectors and/or matrices).
	libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
	libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
	libblis_test_vobj_create( params, datatype, sc_str[2], m, &z );
	libblis_test_vobj_create( params, datatype, sc_str[2], m, &z_save );

	// Set alpha.
	if ( bli_obj_is_real( z ) )
	{
		bli_setsc( -1.0,  0.0, &alpha1 );
		bli_setsc( -0.9,  0.0, &alpha2 );
	}
	else
	{
		bli_setsc(  0.0, -1.0, &alpha1 );
		bli_setsc(  0.0, -0.9, &alpha2 );
	}

	// Randomize x and y, and save y.
	libblis_test_vobj_randomize( params, TRUE, &x );
	libblis_test_vobj_randomize( params, TRUE, &y );
	libblis_test_vobj_randomize( params, TRUE, &z );
	bli_copyv( &z, &z_save );

	// Apply the parameters.
	bli_obj_set_conj( conjx, x );
	bli_obj_set_conj( conjy, y );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copyv( &z_save, &z );

		time = bli_clock();

		libblis_test_axpy2v_impl( iface,
		                          &alpha1, &alpha2, &x, &y, &z,
		                          &cntx );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m + 2.0 * m ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( z ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_axpy2v_check( params, &alpha1, &alpha2, &x, &y, &z, &z_save, resid );

	// Zero out performance and residual if output vector is empty.
	libblis_test_check_empty_problem( &z, perf, resid );

	// Free the test objects.
	bli_obj_free( &x );
	bli_obj_free( &y );
	bli_obj_free( &z );
	bli_obj_free( &z_save );

	// Finalize the context.
	bli_axpy2v_cntx_finalize( &cntx );
}



void libblis_test_axpy2v_impl
     (
       iface_t   iface,
       obj_t*    alpha1,
       obj_t*    alpha2,
       obj_t*    x,
       obj_t*    y,
       obj_t*    z,
       cntx_t*   cntx
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_axpy2v_ex( alpha1, alpha2, x, y, z, cntx );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_axpy2v_check
     (
       test_params_t* params,
       obj_t*         alpha1,
       obj_t*         alpha2,
       obj_t*         x,
       obj_t*         y,
       obj_t*         z,
       obj_t*         z_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_datatype( *z );
	num_t  dt_real = bli_obj_datatype_proj_to_real( *z );

	dim_t  m       = bli_obj_vector_dim( *z );

	obj_t  x_temp, y_temp, z_temp;
	obj_t  norm;

	double junk;

	//
	// Pre-conditions:
	// - x is randomized.
	// - y is randomized.
	// - z_orig is randomized.
	// Note:
	// - alpha1, alpha2 should have a non-zero imaginary component in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   z := z_orig + alpha1 * conjx(x) + alpha2 * conjy(y)
	//
	// is functioning correctly if
	//
	//   normf( z - v )
	//
	// is negligible, where v contains z as computed by two calls to axpyv.
	//

	bli_obj_scalar_init_detached( dt_real, &norm );

	bli_obj_create( dt, m, 1, 0, 0, &x_temp );
	bli_obj_create( dt, m, 1, 0, 0, &y_temp );
	bli_obj_create( dt, m, 1, 0, 0, &z_temp );

	bli_copyv( x,      &x_temp );
	bli_copyv( y,      &y_temp );
	bli_copyv( z_orig, &z_temp );

	bli_scalv( alpha1, &x_temp );
	bli_scalv( alpha2, &y_temp );
	bli_addv( &x_temp, &z_temp );
	bli_addv( &y_temp, &z_temp );

	bli_subv( &z_temp, z );
	bli_normfv( z, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &x_temp );
	bli_obj_free( &y_temp );
	bli_obj_free( &z_temp );
}

