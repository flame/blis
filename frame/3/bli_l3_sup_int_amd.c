/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019-23, Advanced Micro Devices, Inc. All rights reserved.

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

err_t bli_gemmsup_int
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4);

	const num_t  dt          = bli_obj_dt( c );
	const dim_t  m           = bli_obj_length( c );
	const dim_t  n           = bli_obj_width( c );
	const dim_t  k           = bli_obj_width( a );
	const dim_t  MR          = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t  NR          = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
	const dim_t  KC          = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );
	const bool   auto_factor = bli_rntm_auto_factor( rntm );
	const dim_t  n_threads   = bli_rntm_num_threads( rntm );
	bool         use_pb      = FALSE;
	dim_t        jc_new;
	dim_t        ic_new;

	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );
	const bool    is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	const bool    is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;
	const bool    row_pref = bli_cntx_l3_sup_ker_prefers_rows_dt( dt, stor_id, cntx );
	const bool    col_pref = !row_pref;

	// For row-preferred kernels, rrr_rrc_rcr_crr becomes primary
	// For col-preferred kernels, rcc_crc_ccr_ccc becomes primary
	const bool    is_primary = ( row_pref && is_rrr_rrc_rcr_crr ) ||
		                   ( col_pref && is_rcc_crc_ccr_ccc );

	#ifdef TRACEVAR
	if ( bli_thread_am_ochief( thread ) )
	  printf( "bli_l3_sup_int(): var2m primary\n" );
	#endif

	// Don't use the small/unpacked implementation if one of the matrices
	// uses general stride.
	if ( stor_id == BLIS_XXX ) {
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_4, "SUP doesn't support general stide.");
		return BLIS_FAILURE;
	}

	if ( is_primary )
	{
	  // This branch handles:
	  //  - rrr rrc rcr crr for row-preferential kernels
	  //  - rcc crc ccr ccc for column-preferential kernels

	  // calculate number of micropanels in m and n dimensions and
	  // recalculate the automatic thread factorization based on these number of  micropanels
	  const dim_t mu = m / MR;
	  const dim_t nu = n / NR;

	  // Heuristic to decide whether to use 1n variant or not for sgemm.
	  use_pb = ( ( nu >= ( 4 * mu ) ) && ( k >= KC ) ) ? TRUE : FALSE;

	  // If the parallel thread factorization was automatic, we update it
	  // with a new factorization based on the matrix dimensions in units
	  // of micropanels. However in case smart threading is enabled,
	  // auto_factor will be false.
	  if ( auto_factor )
	  {
	      // In the block-panel algorithm, the m dimension is parallelized
	      // with ic_nt and the n dimension is parallelized with jc_nt.
	      bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );

	      // Update the ways of parallelism for the jc and ic loops, and then
	      // update the current thread's root thrinfo_t node according to the
	      // new ways of parallelism value for the jc loop.
	      bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, rntm );
	      bli_l3_sup_thrinfo_update_root( rntm, thread );
	  }

	  //Enable packing for B matrix for higher sizes
	  if(bli_is_float(dt) && (n_threads==1)) {
              if((m > 240) &&  (k > 240) && (n > 240))
	          bli_rntm_set_pack_b( 1, rntm );//packb
	  }

	  //Enable packing of B matrix for complex data type
	  if (bli_is_dcomplex(dt) && (n_threads == 1))
	  {
		  if ((m > 55) && (k > 55) && (n > 55))
		  {
				if ( row_pref )
					bli_rntm_set_pack_b(1, rntm);//packb
		  }
	  }

#if defined(BLIS_FAMILY_ZEN3) || defined(BLIS_FAMILY_AMDZEN)

	  //Enable packing of B matrix for double data type when dims at per
	  //thread level are above caches and enable packing of A when transA
	  //(RRC or CRC storage ids) to avoid rd kernels
	  if(bli_is_double(dt) && (bli_arch_query_id() == BLIS_ARCH_ZEN3))
	  {
		  dim_t m_pt = (m/bli_rntm_ways_for( BLIS_MC, rntm ));
		  dim_t n_pt = (n/bli_rntm_ways_for( BLIS_NC, rntm ));

		  if(k > 120)
		  {
			  if(((m_pt > 320) && (n_pt > 120)) || ((m_pt > 120) && (n_pt > 320)))
			  {
				  bli_rntm_set_pack_b(1, rntm);//packb

				  if(( stor_id==BLIS_RRC ) || ( stor_id==BLIS_CRC ))
					bli_rntm_set_pack_a(1, rntm);//packa
			  }
		  }
	  }
#endif
	  // Using the 1n kernel (B broadcast) gave better performance for sgemm
	  // in single-thread scenario, given the number of n panels are
	  // sufficiently larger than m panels.
	  if ( bli_is_float( dt ) && ( n_threads == 1 ) && ( use_pb == TRUE ) )
	  {
		bli_gemmsup_ref_var1n( BLIS_NO_TRANSPOSE,
			                 alpha, a, b, beta, c,
			                 stor_id, cntx, rntm, thread );
	  }
	  else
	  {
	  	bli_gemmsup_ref_var2m( BLIS_NO_TRANSPOSE,
				             alpha, a, b, beta, c,
				             stor_id, cntx, rntm, thread );
	  }
	}
	else
	{
	  // This branch handles:
	  //  - rrr rrc rcr crr for column-preferential kernels
	  //  - rcc crc ccr ccc for row-preferential kernels
	  const dim_t mu = n / MR; // the n becomes m after a transposition
	  const dim_t nu = m / NR; // the m becomes n after a transposition

	  use_pb = ( ( nu >= ( 4 * mu ) ) && ( k >= KC ) ) ? TRUE : FALSE;

	  if ( auto_factor )
	  {
	      // In the block-panel algorithm, the m dimension is parallelized
	      // with ic_nt and the n dimension is parallelized with jc_nt.
	      bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );

	      // Update the ways of parallelism for the jc and ic loops, and then
	      // update the current thread's root thrinfo_t node according to the
	      // new ways of parallelism value for the jc loop.
	      bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, rntm );
	      bli_l3_sup_thrinfo_update_root( rntm, thread );
	  }

	  /* Enable packing for B matrix for higher sizes. Note that pack A
	   * becomes pack B inside var2m because this is transpose case*/
	  if(bli_is_float(dt) && (n_threads==1)) {
              if((m > 240) &&  (k > 240) && (n > 240))
	          bli_rntm_set_pack_a( 1, rntm );//packb
	  }

	  //Enable packing of A matrix for complex data type
	  if (bli_is_dcomplex(dt) && (n_threads == 1))
	  {
		  if ((m > 55) && (k > 55) && (n > 55))
		  {
				if ( row_pref )
					bli_rntm_set_pack_a(1, rntm);//packb
		  }
	  }

#if defined(BLIS_FAMILY_ZEN3) || defined(BLIS_FAMILY_AMDZEN)

	  //Enable packing of B matrix for double data type when dims at per
	  //thread level are above caches and enable packing of A when transA
	  //(RRC or CRC storage ids) to avoid rd kernels
	  if(bli_is_double(dt) && (bli_arch_query_id() == BLIS_ARCH_ZEN3))
	  {
		  dim_t m_pt = (m/bli_rntm_ways_for( BLIS_NC, rntm ));
		  dim_t n_pt = (n/bli_rntm_ways_for( BLIS_MC, rntm ));

		  if(k > 120)
		  {
			  if(((m_pt > 320) && (n_pt > 120)) || ((m_pt > 120) && (n_pt > 320)))
			  {
				  bli_rntm_set_pack_a(1, rntm);//packb

				  if(( stor_id==BLIS_RRC ) || ( stor_id==BLIS_CRC ))
					bli_rntm_set_pack_b(1, rntm);//packa
			  }
		  }
	  }
#endif
	  if ( bli_is_float( dt ) && ( n_threads == 1 ) && ( use_pb == TRUE ) )
	  {
		bli_gemmsup_ref_var1n( BLIS_TRANSPOSE,
			                 alpha, a, b, beta, c,
			                 stor_id, cntx, rntm, thread );
	  }
	  else
	  {
	  	bli_gemmsup_ref_var2m( BLIS_TRANSPOSE,
			                 alpha, a, b, beta, c,
			                 stor_id, cntx, rntm, thread );
	  }
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4);
	return BLIS_SUCCESS;


}

// -----------------------------------------------------------------------------

err_t bli_gemmtsup_int
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4);
//	AOCL_DTL_LOG_GEMMT_INPUTS(AOCL_DTL_LEVEL_TRACE_4, alpha, a, b, beta, c);


	const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

	// Don't use the small/unpacked implementation if one of the matrices
	// uses general stride.
	if ( stor_id == BLIS_XXX ) {
		AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_4, "SUP doesn't support general stide.");
		return BLIS_FAILURE;
	}

	const bool    is_rrr_rrc_rcr_crr = ( stor_id == BLIS_RRR ||
	                                     stor_id == BLIS_RRC ||
	                                     stor_id == BLIS_RCR ||
	                                     stor_id == BLIS_CRR );
	const bool    is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;

	const num_t   dt         = bli_obj_dt( c );
	const bool    row_pref   = bli_cntx_l3_sup_ker_prefers_rows_dt( dt, stor_id, cntx );

	const bool    is_primary = ( row_pref ? is_rrr_rrc_rcr_crr
	                                      : is_rcc_crc_ccr_ccc );

	const dim_t  m           = bli_obj_length( c );
	const dim_t  n           = m;
	const dim_t  k           = bli_obj_width( a );
	const dim_t  MR          = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t  NR          = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
	const bool   auto_factor = bli_rntm_auto_factor( rntm );
	const dim_t  n_threads   = bli_rntm_num_threads( rntm );
	bool         use_bp      = TRUE;
	dim_t        jc_new;
	dim_t        ic_new;


	if ( is_primary )
	{
		// This branch handles:
		//  - rrr rrc rcr crr for row-preferential kernels
		//  - rcc crc ccr ccc for column-preferential kernels

		const dim_t mu = m / MR;
		const dim_t nu = n / NR;

		// Decide which algorithm to use (block-panel var2m or panel-block
		// var1n) based on the number of micropanels in the m and n dimensions.
		// Also, recalculate the automatic thread factorization.

		if         ( mu >= nu )    use_bp = TRUE;
		else /* if ( mu <  nu ) */ use_bp = TRUE;// var1n is not implemented for GEMMT

		// If the parallel thread factorization was automatic, we update it
		// with a new factorization based on the matrix dimensions in units
		// of micropanels.
		if ( auto_factor )
		{
			if ( use_bp )
			{
				// In the block-panel algorithm, the m dimension is parallelized
				// with ic_nt and the n dimension is parallelized with jc_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );
			}
			else // if ( !use_bp )
			{
				// In the panel-block algorithm, the m dimension is parallelized
				// with jc_nt and the n dimension is parallelized with ic_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &jc_new, &ic_new );
			}

			// Update the ways of parallelism for the jc and ic loops, and then
			// update the current thread's root thrinfo_t node according to the
			// new ways of parallelism value for the jc loop.
			bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, rntm );
			bli_l3_sup_thrinfo_update_root( rntm, thread );
		}

		/* Enable packing for B matrix for higher sizes. Note that pack B
		 * * becomes pack A inside var2m because this is transpose case*/
		if(bli_is_double(dt) && ((n_threads==1)))
		{
			if((m > 320) &&  (k > 50))
				bli_rntm_set_pack_b( 1, rntm );
		}

		if ( use_bp )
		{
			#ifdef TRACEVAR
			if ( bli_thread_am_ochief( thread ) )
			printf( "bli_l3_sup_int(): var2m primary\n" );
			#endif
			// block-panel macrokernel; m -> mc, mr; n -> nc, nr: var2()
			bli_gemmtsup_ref_var2m( BLIS_NO_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
		}
		else // use_pb
		{
			#ifdef TRACEVAR
			if ( bli_thread_am_ochief( thread ) )
			printf( "bli_l3_sup_int(): var1n primary\n" );
			#endif
			// panel-block macrokernel; m -> nc*,mr; n -> mc*,nr: var1()
			bli_gemmtsup_ref_var1n( BLIS_NO_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
			// *requires nudging of nc up to be a multiple of mr.
		}
	}
	else
	{
		// This branch handles:
		//  - rrr rrc rcr crr for column-preferential kernels
		//  - rcc crc ccr ccc for row-preferential kernels

		const dim_t mu = n / MR; // the n becomes m after a transposition
		const dim_t nu = m / NR; // the m becomes n after a transposition

		// Decide which algorithm to use (block-panel var2m or panel-block
		// var1n) based on the number of micropanels in the m and n dimensions.
		// Also, recalculate the automatic thread factorization.

		if         ( mu >= nu )    use_bp = TRUE;
		else /* if ( mu <  nu ) */ use_bp = TRUE; //var1n is not implemented for gemmt

		// If the parallel thread factorization was automatic, we update it
		// with a new factorization based on the matrix dimensions in units
		// of micropanels.
		if ( auto_factor )
		{
			if ( use_bp )
			{
				// In the block-panel algorithm, the m dimension is parallelized
				// with ic_nt and the n dimension is parallelized with jc_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &ic_new, &jc_new );
			}
			else // if ( !use_bp )
			{
				// In the panel-block algorithm, the m dimension is parallelized
				// with jc_nt and the n dimension is parallelized with ic_nt.
				bli_thread_partition_2x2( n_threads, mu, nu, &jc_new, &ic_new );
			}

			// Update the ways of parallelism for the jc and ic loops, and then
			// update the current thread's root thrinfo_t node according to the
			// new ways of parallelism value for the jc loop.
			bli_rntm_set_ways_only( jc_new, 1, ic_new, 1, 1, rntm );
			bli_l3_sup_thrinfo_update_root( rntm, thread );
		}

		/* Enable packing for A matrix for higher sizes. Note that pack A
		 * * becomes pack B inside var2m because this is transpose case*/
		if(bli_is_double(dt) && (n_threads==1))
		{
			if((m > 320) &&  (k > 50))
				bli_rntm_set_pack_a( 1, rntm );
		}


		if ( use_bp )
		{
			#ifdef TRACEVAR
			if ( bli_thread_am_ochief( thread ) )
			printf( "bli_l3_sup_int(): var2m non-primary\n" );
			#endif
			// panel-block macrokernel; m -> nc, nr; n -> mc, mr: var2() + trans
			bli_gemmtsup_ref_var2m( BLIS_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
		}
		else // use_pb
		{
			#ifdef TRACEVAR
			if ( bli_thread_am_ochief( thread ) )
			printf( "bli_l3_sup_int(): var1n non-primary\n" );
			#endif
			// block-panel macrokernel; m -> mc*,nr; n -> nc*,mr: var1() + trans
			bli_gemmtsup_ref_var1n( BLIS_TRANSPOSE,
			                        alpha, a, b, beta, c,
			                        stor_id, cntx, rntm, thread );
			// *requires nudging of mc up to be a multiple of nr.
		}
	}

	// Return success so that the caller knows that we computed the solution.
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
	return BLIS_SUCCESS;
}

