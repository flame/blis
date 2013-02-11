/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"
#include "test_libblis.h"


// Static variables.
static char*     op_str                    = "dotxv";
static char*     o_types                   = "vv";  // x y
static char*     p_types                   = "cc";  // conjx conjy
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_dotxv_deps( test_params_t* params,
                              test_op_t*     op );

void libblis_test_dotxv_experiment( test_params_t* params,
                                    test_op_t*     op,
                                    mt_impl_t      impl,
                                    num_t          datatype,
                                    char*          pc_str,
                                    char*          sc_str,
                                    dim_t          p_cur,
                                    double*        perf,
                                    double*        resid );

void libblis_test_dotxv_impl( mt_impl_t impl,
                              obj_t*    alpha,
                              obj_t*    x,
                              obj_t*    y,
                              obj_t*    beta,
                              obj_t*    rho );

void libblis_test_dotxv_check( obj_t*  alpha,
                               obj_t*  x,
                               obj_t*  y,
                               obj_t*  beta,
                               obj_t*  rho,
                               obj_t*  rho_orig,
                               double* resid );



void libblis_test_dotxv_deps( test_params_t* params, test_op_t* op )
{
	libblis_test_randv( params, &(op->ops->randv) );
	libblis_test_fnormv( params, &(op->ops->fnormv) );
	libblis_test_copyv( params, &(op->ops->copyv) );
}



void libblis_test_dotxv( test_params_t* params, test_op_t* op )
{

	// Return early if this test has already been done.
	if ( op->test_done == TRUE ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_dotxv_deps( params, op );

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
		                        libblis_test_dotxv_experiment );
	}
}



void libblis_test_dotxv_experiment( test_params_t* params,
                                    test_op_t*     op,
                                    mt_impl_t      impl,
                                    num_t          datatype,
                                    char*          pc_str,
                                    char*          sc_str,
                                    dim_t          p_cur,
                                    double*        perf,
                                    double*        resid )
{
	unsigned int n_repeats = params->n_repeats;
	unsigned int i;

	double       time_min  = 1e9;
	double       time;

	dim_t        m;

	conj_t       conjx, conjy, conjconjxy;

	obj_t        alpha, x, y, beta, rho;
	obj_t        rho_save;


	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Map parameter characters to BLIS constants.
	bl2_param_map_char_to_blis_conj( pc_str[0], &conjx );
	bl2_param_map_char_to_blis_conj( pc_str[1], &conjy );

	// Create test scalars.
	bl2_obj_init_scalar( datatype, &alpha );
	bl2_obj_init_scalar( datatype, &beta );
	bl2_obj_init_scalar( datatype, &rho );
	bl2_obj_init_scalar( datatype, &rho_save );

	// Create test operands (vectors and/or matrices).
	libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
	libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );

	// Initialize alpha, beta, and rho.
	bl2_copysc( &BLIS_TWO, &alpha );
	bl2_copysc( &BLIS_ZERO, &beta );
	bl2_copysc( &BLIS_MINUS_ONE, &rho );
	bl2_copysc( &rho, &rho_save );

	// Randomize x.
	bl2_randv( &x );

	// Determine whether to make a copy of x with or without conjugation.
	// 
	//  conjx conjy  ~conjx^conjy   y is initialized as
	//  n     n      c              y = conj(x)
	//  n     c      n              y = x
	//  c     n      n              y = x
	//  c     c      c              y = conj(x)
	//
	conjconjxy = bl2_apply_conj( conjx, conjy );
	conjconjxy = bl2_conj_toggled( conjconjxy );
	bl2_obj_set_conj( conjconjxy, x );
	bl2_copyv( &x, &y );

	// Apply the parameters.
	bl2_obj_set_conj( conjx, x );
	bl2_obj_set_conj( conjy, y );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bl2_copysc( &rho_save, &rho );

		time = bl2_clock();

		libblis_test_dotxv_impl( impl, &alpha, &x, &y, &beta, &rho );

		time_min = bl2_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bl2_obj_is_complex( y ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_dotxv_check( &alpha, &x, &y, &beta, &rho, &rho_save, resid );

	// Free the test objects.
	bl2_obj_free( &x );
	bl2_obj_free( &y );
}



void libblis_test_dotxv_impl( mt_impl_t impl,
                              obj_t*    alpha,
                              obj_t*    x,
                              obj_t*    y,
                              obj_t*    beta,
                              obj_t*    rho )
{
	switch ( impl )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bl2_dotxv( alpha, x, y, beta, rho );
		break;

		default:
		libblis_test_printf_error( "Invalid implementation type.\n" );
	}
}



void libblis_test_dotxv_check( obj_t*  alpha,
                               obj_t*  x,
                               obj_t*  y,
                               obj_t*  beta,
                               obj_t*  rho,
                               obj_t*  rho_orig,
                               double* resid )
{
	num_t  dt_real = bl2_obj_datatype_proj_to_real( *y );

	obj_t  rho_r, rho_i;
	obj_t  norm_x_r, norm_xy_r;
	obj_t  temp_r;

	double zero;
	double junk;

	//
	// Pre-conditions:
	// - x is randomized.
	// - y is equal to conjx(conjy(x)).
	// - alpha must be real-valued.
	// - beta must be zero.
	// Note:
	// - We forgo fully exercising beta scaling in order to simplify the
	//   test.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   rho := beta * rho_orig + alpha * conjx(x)^T conjy(y)
	//
	// is functioning correctly if
	//
	//   sqrtsc( rho.real ) - sqrtsc( alpha ) * fnorm( x )
	//
	// and
	//
	//   rho.imag
	//
	// are negligible.
	//

	bl2_obj_init_scalar( dt_real, &rho_r );
	bl2_obj_init_scalar( dt_real, &rho_i );
	bl2_obj_init_scalar( dt_real, &norm_x_r );
	bl2_obj_init_scalar( dt_real, &norm_xy_r );
	bl2_obj_init_scalar( dt_real, &temp_r );

	bl2_copysc( alpha, &temp_r );
	bl2_sqrtsc( &temp_r, &temp_r );

	bl2_fnormv( x, &norm_x_r );
	bl2_mulsc( &temp_r, &norm_x_r );

	bl2_unzipsc( rho, &rho_r, &rho_i );

	bl2_sqrtsc( &rho_r, &norm_xy_r );

	bl2_subsc( &norm_x_r, &norm_xy_r );
	bl2_getsc( &norm_xy_r, resid, &junk );
	bl2_getsc( &rho_i,     &zero, &junk );

	*resid = bl2_fmaxabs( *resid, zero );
}

