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
static char*     op_str                    = "gemmsup_ukr";
static char*     o_types                   = "mmm"; // ccc
static char*     p_types                   = "";
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_gemmsup_ukr_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

bool libblis_test_gemmsup_ukr_experiment
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

void libblis_test_gemmsup_ukr_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    b,
       obj_t*    beta,
       obj_t*    c,
       cntx_t*   cntx
     );

void libblis_test_gemmsup_ukr_check
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



void libblis_test_gemmsup_ukr_deps
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
	libblis_test_subv( tdata, params, &(op->ops->subv) );
	libblis_test_scalv( tdata, params, &(op->ops->scalv) );
	libblis_test_copym( tdata, params, &(op->ops->copym) );
	libblis_test_scalm( tdata, params, &(op->ops->scalm) );
	libblis_test_gemv( tdata, params, &(op->ops->gemv) );
}



void libblis_test_gemmsup_ukr
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
	     libblis_test_l3ukr_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_gemmsup_ukr_deps( tdata, params, op );

	op->dim_spec[1] = 1; // m
	op->dim_spec[2] = 1; // n

	// Execute the test driver for each implementation requested.
	//if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( tdata,
		                        params,
		                        op,
		                        BLIS_TEST_SEQ_SUP_UKERNEL,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_gemmsup_ukr_experiment );
	}
}



bool libblis_test_gemmsup_ukr_experiment
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

	dim_t        m, n, k;

	obj_t        alpha, a, b, beta, c;
	obj_t        c_save;

	cntx_t*      cntx;


	// Query a context.
	cntx = ( cntx_t* )bli_gks_query_cntx();

	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	dim_t MRM = bli_cntx_get_l3_sup_blksz_max_dt( datatype, BLIS_MR, cntx );
	dim_t NRM = bli_cntx_get_l3_sup_blksz_max_dt( datatype, BLIS_NR, cntx );
	if ( MRM == 0) MRM = bli_cntx_get_blksz_def_dt( datatype, BLIS_MR, cntx );
	if ( NRM == 0) NRM = bli_cntx_get_blksz_def_dt( datatype, BLIS_NR, cntx );

	// Map the dimension specifier to actual dimensions.
	k = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );
	m = op->dim_spec[1];
	n = op->dim_spec[2];

	// Store the register blocksizes so that the driver can retrieve the
	// values later when printing results.
	op->dim_aux[0] = m;
	op->dim_aux[1] = n;

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );
	bli_obj_scalar_init_detached( datatype, &beta );

	stor3_t stor_id = sc_str[0] == 'r' ?
	                    sc_str[1] == 'r' ?
	                      sc_str[2] == 'r' ? BLIS_RRR : BLIS_RRC :
	                      sc_str[2] == 'r' ? BLIS_RCR : BLIS_RCC :
	                    sc_str[1] == 'r' ?
	                      sc_str[2] == 'r' ? BLIS_CRR : BLIS_CRC :
	                      sc_str[2] == 'r' ? BLIS_CCR : BLIS_CCC;

	const bool    is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	const bool    is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;
	const bool    row_pref   = bli_cntx_ukr_prefers_rows_dt( datatype, bli_stor3_ukr( stor_id ), cntx );

	const bool    is_primary = ( row_pref ? is_rrr_rrc_rcr_crr
	                                      : is_rcc_crc_ccr_ccc );

	dim_t m_use = is_primary ? m : n;
	dim_t n_use = is_primary ? n : m;

	// Create test operands.
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          bli_stor3_stora( stor_id ), m_use, k, &a );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          bli_stor3_storb( stor_id ), k, n_use, &b );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          bli_stor3_storc( stor_id ), m_use, n_use, &c );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          bli_stor3_storc( stor_id ), m_use, n_use, &c_save );

	// Set alpha and beta.
	if ( bli_obj_is_real( &c ) )
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

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &c_save, &c );

		time = bli_clock();

		libblis_test_gemmsup_ukr_impl( iface,
		                            &alpha, &a, &b, &beta, &c,
		                            cntx );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * n * k ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &c ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_gemmsup_ukr_check( params, &alpha, &a, &b, &beta, &c, &c_save, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &c, perf, resid );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &c_save );

	if ( n == NRM  )
	{
		if ( m == MRM )
		{
			return true;
		}
		else
		{
			op->dim_spec[1] = m + 1;
			op->dim_spec[2] = 1;
		}
	}
	else
	{
		op->dim_spec[2] = n + 1;
	}

	return false;
}



void libblis_test_gemmsup_ukr_impl
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
	num_t dt = bli_obj_dt( c );

	dim_t m = bli_obj_length( c );
	dim_t n = bli_obj_width( c );
	dim_t k = bli_obj_width( a );

	inc_t rs_a = bli_obj_row_stride( a );
	inc_t cs_a = bli_obj_col_stride( a );
	inc_t rs_b = bli_obj_row_stride( b );
	inc_t cs_b = bli_obj_col_stride( b );
	inc_t rs_c = bli_obj_row_stride( c );
	inc_t cs_c = bli_obj_col_stride( c );

	const void* buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	const void* buf_beta  = bli_obj_buffer_for_1x1( dt, beta );
	const void* buf_a     = bli_obj_buffer_at_off( a );
	const void* buf_b     = bli_obj_buffer_at_off( b );
	      void* buf_c     = bli_obj_buffer_at_off( c );

	conj_t conja = bli_obj_conj_status( a );
	conj_t conjb = bli_obj_conj_status( b );

	stor3_t stor_id = bli_stor3_from_strides( rs_c, cs_c, rs_a, cs_a, rs_b, cs_b );

	const bool    is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	const bool    is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;
	const bool    row_pref   = bli_cntx_ukr_prefers_rows_dt( dt, bli_stor3_ukr( stor_id ), cntx );

	const bool    is_primary = ( row_pref ? is_rrr_rrc_rcr_crr
	                                      : is_rcc_crc_ccr_ccc );

	if ( !is_primary )
	{
		      conj_t conjtmp = conja; conja = conjb; conjb = conjtmp;
		      dim_t  len_tmp =     m;     m =     n;     n = len_tmp;
		const void*  buf_tmp = buf_a; buf_a = buf_b; buf_b = buf_tmp;
		      inc_t  str_tmp =  rs_a;  rs_a =  cs_b;  cs_b = str_tmp;
		             str_tmp =  cs_a;  cs_a =  rs_b;  rs_b = str_tmp;
		             str_tmp =  rs_c;  rs_c =  cs_c;  cs_c = str_tmp;

		//stor_id = bli_stor3_trans( stor_id );
	}

	gemmsup_ker_ft f = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx );

	dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
	dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );

	auxinfo_t auxinfo;
	bli_auxinfo_set_ps_a( MR * rs_a, &auxinfo );
	bli_auxinfo_set_ps_b( NR * cs_b, &auxinfo );

	switch ( iface )
	{
		case BLIS_TEST_SEQ_SUP_UKERNEL:
		f
		(
		  BLIS_NO_CONJUGATE,
		  BLIS_NO_CONJUGATE,
		  m, n, k,
		  buf_alpha,
		  buf_a,
		  rs_a,
		  cs_a,
		  buf_b,
		  rs_b,
		  cs_b,
		  buf_beta,
		  buf_c,
		  rs_c,
		  cs_c,
		  &auxinfo,
		  cntx
		);
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_gemmsup_ukr_check
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
	num_t  dt      = bli_obj_dt( c );
	num_t  dt_real = bli_obj_dt_proj_to_real( c );

	dim_t  m       = bli_obj_length( c );
	dim_t  n       = bli_obj_width( c );
	dim_t  k       = bli_obj_width( a );

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
	//   normfv( v - z )
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

