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

//
// -- blksz_t structure --------------------------------------------------------
//

static blksz_t bli_gks_blkszs[BLIS_NUM_BLKSZS] =
{
         /*           float (0)       scomplex (1)         double (2)       dcomplex (3) */
/* kr */ { { BLIS_DEFAULT_KR_S, BLIS_DEFAULT_KR_C, BLIS_DEFAULT_KR_D, BLIS_DEFAULT_KR_Z, },
           { BLIS_PACKDIM_KR_S, BLIS_PACKDIM_KR_C, BLIS_PACKDIM_KR_D, BLIS_PACKDIM_KR_Z, }
         },
/* mr */ { { BLIS_DEFAULT_MR_S, BLIS_DEFAULT_MR_C, BLIS_DEFAULT_MR_D, BLIS_DEFAULT_MR_Z, },
           { BLIS_PACKDIM_MR_S, BLIS_PACKDIM_MR_C, BLIS_PACKDIM_MR_D, BLIS_PACKDIM_MR_Z, }
         },
/* nr */ { { BLIS_DEFAULT_NR_S, BLIS_DEFAULT_NR_C, BLIS_DEFAULT_NR_D, BLIS_DEFAULT_NR_Z, },
           { BLIS_PACKDIM_NR_S, BLIS_PACKDIM_NR_C, BLIS_PACKDIM_NR_D, BLIS_PACKDIM_NR_Z, }
         },
/* mc */ { { BLIS_DEFAULT_MC_S, BLIS_DEFAULT_MC_C, BLIS_DEFAULT_MC_D, BLIS_DEFAULT_MC_Z, },
           { BLIS_MAXIMUM_MC_S, BLIS_MAXIMUM_MC_C, BLIS_MAXIMUM_MC_D, BLIS_MAXIMUM_MC_Z, }
         },
/* kc */ { { BLIS_DEFAULT_KC_S, BLIS_DEFAULT_KC_C, BLIS_DEFAULT_KC_D, BLIS_DEFAULT_KC_Z, },
           { BLIS_MAXIMUM_KC_S, BLIS_MAXIMUM_KC_C, BLIS_MAXIMUM_KC_D, BLIS_MAXIMUM_KC_Z, }
         },
/* nc */ { { BLIS_DEFAULT_NC_S, BLIS_DEFAULT_NC_C, BLIS_DEFAULT_NC_D, BLIS_DEFAULT_NC_Z, },
           { BLIS_MAXIMUM_NC_S, BLIS_MAXIMUM_NC_C, BLIS_MAXIMUM_NC_D, BLIS_MAXIMUM_NC_Z, }
         },
/* m2 */ { { BLIS_DEFAULT_M2_S, BLIS_DEFAULT_M2_C, BLIS_DEFAULT_M2_D, BLIS_DEFAULT_M2_Z, },
           { BLIS_DEFAULT_M2_S, BLIS_DEFAULT_M2_C, BLIS_DEFAULT_M2_D, BLIS_DEFAULT_M2_Z, }
         },
/* n2 */ { { BLIS_DEFAULT_N2_S, BLIS_DEFAULT_N2_C, BLIS_DEFAULT_N2_D, BLIS_DEFAULT_N2_Z, },
           { BLIS_DEFAULT_N2_S, BLIS_DEFAULT_N2_C, BLIS_DEFAULT_N2_D, BLIS_DEFAULT_N2_Z, }
         },
/* 1f */ { { BLIS_DEFAULT_1F_S, BLIS_DEFAULT_1F_C, BLIS_DEFAULT_1F_D, BLIS_DEFAULT_1F_Z, },
           { BLIS_DEFAULT_1F_S, BLIS_DEFAULT_1F_C, BLIS_DEFAULT_1F_D, BLIS_DEFAULT_1F_Z, }
         },
/* af */ { { BLIS_DEFAULT_AF_S, BLIS_DEFAULT_AF_C, BLIS_DEFAULT_AF_D, BLIS_DEFAULT_AF_Z, },
           { BLIS_DEFAULT_AF_S, BLIS_DEFAULT_AF_C, BLIS_DEFAULT_AF_D, BLIS_DEFAULT_AF_Z, }
         },
/* df */ { { BLIS_DEFAULT_DF_S, BLIS_DEFAULT_DF_C, BLIS_DEFAULT_DF_D, BLIS_DEFAULT_DF_Z, },
           { BLIS_DEFAULT_DF_S, BLIS_DEFAULT_DF_C, BLIS_DEFAULT_DF_D, BLIS_DEFAULT_DF_Z, }
         },
/* xf */ { { BLIS_DEFAULT_XF_S, BLIS_DEFAULT_XF_C, BLIS_DEFAULT_XF_D, BLIS_DEFAULT_XF_Z, },
           { BLIS_DEFAULT_XF_S, BLIS_DEFAULT_XF_C, BLIS_DEFAULT_XF_D, BLIS_DEFAULT_XF_Z, }
         },
/* vf */ { { BLIS_DEFAULT_VF_S, BLIS_DEFAULT_VF_C, BLIS_DEFAULT_VF_D, BLIS_DEFAULT_VF_Z, },
           { BLIS_DEFAULT_VF_S, BLIS_DEFAULT_VF_C, BLIS_DEFAULT_VF_D, BLIS_DEFAULT_VF_Z, }
         },
};

// -----------------------------------------------------------------------------

void bli_gks_get_blksz( bszid_t  bs_id,
                        blksz_t* blksz )
{
	*blksz = bli_gks_blkszs[ bs_id ];
}

void bli_gks_cntx_set_blkszs( ind_t method, dim_t n_bs, ... )
{
	/* Example prototypes:

	   void
	   bli_gks_cntx_set_blkszs(

	             ind_t   method = BLIS_NAT,
	             dim_t   n_bs,
	             bszid_t bs0_id, bszid_t bm0_id,
	             bszid_t bs1_id, bszid_t bm1_id,
	             bszid_t bs2_id, bszid_t bm2_id,
	             ...
	             cntx_t* cntx );

	   void
	   bli_gks_cntx_set_blkszs(

	             ind_t   method != BLIS_NAT,
	             dim_t   n_bs,
	             bszid_t bs0_id, bszid_t bm0_id, dim_t scalr0,
	             bszid_t bs1_id, bszid_t bm1_id, dim_t scalr1,
	             bszid_t bs2_id, bszid_t bm2_id, dim_t scalr2,
	             ...
	             cntx_t* cntx );
	*/
	va_list   args;
	dim_t     i;

	bszid_t*  bszids;
	bszid_t*  bmults;
	double*   scalrs;

	cntx_t*   cntx;

	blksz_t*  cntx_blkszs;
	bszid_t*  cntx_bmults;

	bszid_t   bs_id;
	bszid_t   bm_id;
	double    scalr;

	// Allocate some temporary local arrays.
	bszids = bli_malloc_intl( n_bs * sizeof( bszid_t  ) );
	bmults = bli_malloc_intl( n_bs * sizeof( bszid_t  ) );
	scalrs = bli_malloc_intl( n_bs * sizeof( double   ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_bs );

	// Handle native and induced method cases separately.
	if ( method == BLIS_NAT )
	{
		// Process n_bs tuples.
		for ( i = 0; i < n_bs; ++i )
		{
			// Here, we query the variable argument list for:
			// - the bszid_t of the blocksize we're about to process,
			// - the bszid_t of the multiple we need to associate with
			//   the blksz_t object.
			bs_id = va_arg( args, bszid_t  );
			bm_id = va_arg( args, bszid_t  );

			// Store the values in our temporary arrays.
			bszids[ i ] = bs_id;
			bmults[ i ] = bm_id;
		}
	}
	else // if induced method execution was indicated
	{
		// Process n_bs tuples.
		for ( i = 0; i < n_bs; ++i )
		{
			// Here, we query the variable argument list for:
			// - the bszid_t of the blocksize we're about to process,
			// - the bszid_t of the multiple we need to associate with
			//   the blksz_t object.
			// - the scalar we wish to apply to the real blocksizes to
			//   come up with the induced complex blocksizes.
			bs_id = va_arg( args, bszid_t  );
			bm_id = va_arg( args, bszid_t  );
			scalr = va_arg( args, double   );

			// Store the values in our temporary arrays.
			bszids[ i ] = bs_id;
			bmults[ i ] = bm_id;
			scalrs[ i ] = scalr;
		}
	}

	// The last argument should be the context pointer.
	cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Save the execution type into the context.
	bli_cntx_set_method( method, cntx );

	// Query the context for the addresses of:
	// - the blocksize object array
	// - the blocksize multiple array
	cntx_blkszs = bli_cntx_blkszs_buf( cntx );
	cntx_bmults = bli_cntx_bmults_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context.

	// Handle native and induced method cases separately.
	if ( method == BLIS_NAT )
	{
		// Process each blocksize id tuple provided.
		for ( i = 0; i < n_bs; ++i )
		{
			// Read the current blocksize id, blocksize multiple id.
			      bszid_t  bs_id = bszids[ i ];
			      bszid_t  bm_id = bmults[ i ];

			      blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];

			// Query the blocksizes (blksz_t) associated with bs_id and save
			// them directly into the appropriate location in the context's
			// blksz_t array.
			bli_gks_get_blksz( bs_id, cntx_blksz );

			// Copy the blocksize multiple id into the context.
			cntx_bmults[ bs_id ] = bm_id;
		}
	}
	else
	{
		// Process each blocksize id tuple provided.
		for ( i = 0; i < n_bs; ++i )
		{
			// Read the current blocksize id, blocksize multiple id,
			// and blocksize scalar.
			      bszid_t  bs_id = bszids[ i ];
			      bszid_t  bm_id = bmults[ i ];
			      double   scalr = scalrs[ i ];

			      blksz_t  blksz;
			      blksz_t  bmult;

			      blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];

			// Query the blocksizes (blksz_t) associated with bs_id and bm_id
			// and use them to populate a pair of local blksz_t objects.
			bli_gks_get_blksz( bs_id, &blksz );
			bli_gks_get_blksz( bm_id, &bmult );

			// Copy the real domain values of the source blksz_t object into
			// the context, duplicating into the complex domain fields.
			bli_blksz_copy_dt( BLIS_FLOAT,  &blksz, BLIS_FLOAT,    cntx_blksz );
			bli_blksz_copy_dt( BLIS_DOUBLE, &blksz, BLIS_DOUBLE,   cntx_blksz );
			bli_blksz_copy_dt( BLIS_FLOAT,  &blksz, BLIS_SCOMPLEX, cntx_blksz );
			bli_blksz_copy_dt( BLIS_DOUBLE, &blksz, BLIS_DCOMPLEX, cntx_blksz );

			// The next steps apply only to cache blocksizes, and not register
			// blocksizes (ie: they only apply to blocksizes for which the
			// blocksize multiple id is different than the blocksize id) and
			// only when the scalar provided is non-unit.
			if ( bs_id != bm_id && scalr != 1.0 ) 
			{
				// Scale the complex domain values in the blocksize object.
				bli_blksz_scale_dt_by( 1, (dim_t)scalr, BLIS_SCOMPLEX, cntx_blksz );
				bli_blksz_scale_dt_by( 1, (dim_t)scalr, BLIS_DCOMPLEX, cntx_blksz );

				// Finally, round the newly-scaled blocksizes down to their
				// respective multiples.
				bli_blksz_reduce_dt_to( BLIS_FLOAT,  &bmult, BLIS_SCOMPLEX, cntx_blksz );
				bli_blksz_reduce_dt_to( BLIS_DOUBLE, &bmult, BLIS_DCOMPLEX, cntx_blksz );
			}

			// Copy the blocksize multiple id into the context.
			cntx_bmults[ bs_id ] = bm_id;
		}
	}

	// Free the temporary local arrays.
	bli_free_intl( bszids );
	bli_free_intl( bmults );
	bli_free_intl( scalrs );
}


//
// -- level-3 micro-kernel structure -------------------------------------------
//

static func_t bli_gks_l3_ind_ukrs[BLIS_NUM_IND_METHODS]
                                 [BLIS_NUM_LEVEL3_UKRS] =
{
              /*      s(0)  c(1)                         d(2)  z(3)                        */
/* 3mh        */  {
/* gemm       */  { { NULL, BLIS_CGEMM3MH_UKERNEL,       NULL, BLIS_ZGEMM3MH_UKERNEL,       } },
/* gemmtrsm_l */  { { NULL, NULL,                        NULL, NULL,                        } },
/* gemmtrsm_u */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_l     */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_u     */  { { NULL, NULL,                        NULL, NULL,                        } },
                  },
/* 3m3        */  {
/* gemm       */  { { NULL, BLIS_CGEMM3M3_UKERNEL,       NULL, BLIS_ZGEMM3M3_UKERNEL,       } },
/* gemmtrsm_l */  { { NULL, NULL,                        NULL, NULL,                        } },
/* gemmtrsm_u */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_l     */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_u     */  { { NULL, NULL,                        NULL, NULL,                        } },
                  },
/* 3m2        */  {
/* gemm       */  { { NULL, BLIS_CGEMM3M2_UKERNEL,       NULL, BLIS_ZGEMM3M2_UKERNEL,       } },
/* gemmtrsm_l */  { { NULL, NULL,                        NULL, NULL,                        } },
/* gemmtrsm_u */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_l     */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_u     */  { { NULL, NULL,                        NULL, NULL,                        } },
                  },
/* 3m1        */  {
/* gemm       */  { { NULL, BLIS_CGEMM3M1_UKERNEL,       NULL, BLIS_ZGEMM3M1_UKERNEL,       } },
/* gemmtrsm_l */  { { NULL, BLIS_CGEMMTRSM3M1_L_UKERNEL, NULL, BLIS_ZGEMMTRSM3M1_L_UKERNEL, } },
/* gemmtrsm_u */  { { NULL, BLIS_CGEMMTRSM3M1_U_UKERNEL, NULL, BLIS_ZGEMMTRSM3M1_U_UKERNEL, } },
/* trsm_l     */  { { NULL, BLIS_CTRSM3M1_L_UKERNEL,     NULL, BLIS_ZTRSM3M1_L_UKERNEL,     } },
/* trsm_u     */  { { NULL, BLIS_CTRSM3M1_U_UKERNEL,     NULL, BLIS_ZTRSM3M1_U_UKERNEL,     } },
                  },
/* 4mh        */  {
/* gemm       */  { { NULL, BLIS_CGEMM4MH_UKERNEL,       NULL, BLIS_ZGEMM4MH_UKERNEL,       } },
/* gemmtrsm_l */  { { NULL, NULL,                        NULL, NULL,                        } },
/* gemmtrsm_u */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_l     */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_u     */  { { NULL, NULL,                        NULL, NULL,                        } },
                  },
/* 4m1b       */  {
/* gemm       */  { { NULL, BLIS_CGEMM4MB_UKERNEL,       NULL, BLIS_ZGEMM4MB_UKERNEL,       } },
/* gemmtrsm_l */  { { NULL, NULL,                        NULL, NULL,                        } },
/* gemmtrsm_u */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_l     */  { { NULL, NULL,                        NULL, NULL,                        } },
/* trsm_u     */  { { NULL, NULL,                        NULL, NULL,                        } },
                  },
/* 4m1a       */  {
/* gemm       */  { { NULL, BLIS_CGEMM4M1_UKERNEL,       NULL, BLIS_ZGEMM4M1_UKERNEL,       } },
/* gemmtrsm_l */  { { NULL, BLIS_CGEMMTRSM4M1_L_UKERNEL, NULL, BLIS_ZGEMMTRSM4M1_L_UKERNEL, } },
/* gemmtrsm_u */  { { NULL, BLIS_CGEMMTRSM4M1_U_UKERNEL, NULL, BLIS_ZGEMMTRSM4M1_U_UKERNEL, } },
/* trsm_l     */  { { NULL, BLIS_CTRSM4M1_L_UKERNEL,     NULL, BLIS_ZTRSM4M1_L_UKERNEL,     } },
/* trsm_u     */  { { NULL, BLIS_CTRSM4M1_U_UKERNEL,     NULL, BLIS_ZTRSM4M1_U_UKERNEL,     } },
                  },
/* nat        */  {
/* gemm       */  { { BLIS_SGEMM_UKERNEL,       BLIS_CGEMM_UKERNEL,
                      BLIS_DGEMM_UKERNEL,       BLIS_ZGEMM_UKERNEL,       } },
/* gemmtrsm_l */  { { BLIS_SGEMMTRSM_L_UKERNEL, BLIS_CGEMMTRSM_L_UKERNEL,
                      BLIS_DGEMMTRSM_L_UKERNEL, BLIS_ZGEMMTRSM_L_UKERNEL, } },
/* gemmtrsm_u */  { { BLIS_SGEMMTRSM_U_UKERNEL, BLIS_CGEMMTRSM_U_UKERNEL,
                      BLIS_DGEMMTRSM_U_UKERNEL, BLIS_ZGEMMTRSM_U_UKERNEL, } },
/* trsm_l     */  { { BLIS_STRSM_L_UKERNEL,     BLIS_CTRSM_L_UKERNEL,
                      BLIS_DTRSM_L_UKERNEL,     BLIS_ZTRSM_L_UKERNEL,     } },
/* trsm_u     */  { { BLIS_STRSM_U_UKERNEL,     BLIS_CTRSM_U_UKERNEL,
                      BLIS_DTRSM_U_UKERNEL,     BLIS_ZTRSM_U_UKERNEL,     } },
                  },
};

static func_t bli_gks_l3_ref_ukrs[BLIS_NUM_LEVEL3_UKRS] =
{
                /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
/* gemm       */  { { BLIS_SGEMM_UKERNEL_REF,       BLIS_CGEMM_UKERNEL_REF,
                      BLIS_DGEMM_UKERNEL_REF,       BLIS_ZGEMM_UKERNEL_REF,       } },
/* gemmtrsm_l */  { { BLIS_SGEMMTRSM_L_UKERNEL_REF, BLIS_CGEMMTRSM_L_UKERNEL_REF,
                      BLIS_DGEMMTRSM_L_UKERNEL_REF, BLIS_ZGEMMTRSM_L_UKERNEL_REF, } },
/* gemmtrsm_u */  { { BLIS_SGEMMTRSM_U_UKERNEL_REF, BLIS_CGEMMTRSM_U_UKERNEL_REF,
                      BLIS_DGEMMTRSM_U_UKERNEL_REF, BLIS_ZGEMMTRSM_U_UKERNEL_REF, } },
/* trsm_l     */  { { BLIS_STRSM_L_UKERNEL_REF,     BLIS_CTRSM_L_UKERNEL_REF,
                      BLIS_DTRSM_L_UKERNEL_REF,     BLIS_ZTRSM_L_UKERNEL_REF,     } },
/* trsm_u     */  { { BLIS_STRSM_U_UKERNEL_REF,     BLIS_CTRSM_U_UKERNEL_REF,
                      BLIS_DTRSM_U_UKERNEL_REF,     BLIS_ZTRSM_U_UKERNEL_REF,     } },
};

// -----------------------------------------------------------------------------

void bli_gks_get_l3_nat_ukr( l3ukr_t ukr,
                             func_t* func )
{
	*func = bli_gks_l3_ind_ukrs[ BLIS_NAT ][ ukr ];
}

void bli_gks_get_l3_vir_ukr( ind_t   method,
                             l3ukr_t ukr,
                             func_t* func )
{
	*func = bli_gks_l3_ind_ukrs[ method ][ ukr ];
}

void bli_gks_get_l3_ref_ukr( l3ukr_t ukr,
                             func_t* func )
{
	*func = bli_gks_l3_ref_ukrs[ ukr ];
}

void bli_gks_cntx_set_l3_nat_ukr( l3ukr_t ukr,
                                  cntx_t* cntx )
{
	func_t* cntx_l3_nat_ukrs = bli_cntx_l3_nat_ukrs_buf( cntx );
	func_t* cntx_l3_nat_ukr  = &cntx_l3_nat_ukrs[ ukr ];

	bli_gks_get_l3_nat_ukr( ukr, cntx_l3_nat_ukr );
}

void bli_gks_cntx_set_l3_nat_ukrs( dim_t n_uk, ... )
{
	/* Example prototype:

	   void
	   bli_gks_cntx_set_l3_nat_ukrs( dim_t   n_uk,
	                                 l3ukr_t ukr0_id,
	                                 l3ukr_t ukr1_id,
	                                 l3ukr_t ukr2_id,
	                                 ...
	                                 cntx_t* cntx );
	*/

	va_list   args;
	dim_t     i;
	l3ukr_t*  l3_ukrs;
	cntx_t*   cntx;

	// Allocate some temporary local arrays.
	l3_ukrs = bli_malloc_intl( n_uk * sizeof( l3ukr_t ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_uk );

	// Process n_uk kernel ids.
	for ( i = 0; i < n_uk; ++i )
	{
		// Here, we query the variable argument list for the kernel id.
		const l3ukr_t uk_id = va_arg( args, l3ukr_t  );

		// Store the value in our temporary array.
		l3_ukrs[ i ] = uk_id;
	}

	// The last argument should be the context pointer.
	cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Process each kernel id provided.
	for ( i = 0; i < n_uk; ++i )
	{
		// Read the current kernel id.
		const l3ukr_t uk_id = l3_ukrs[ i ];

		// Query the func_t associated with uk_id and save it directly into
		// the context.
		bli_gks_cntx_set_l3_nat_ukr( uk_id, cntx );
	}

	// Free the temporary local array.
	bli_free_intl( l3_ukrs );
}

void bli_gks_cntx_set_l3_vir_ukr( ind_t   method,
                                  l3ukr_t ukr,
                                  cntx_t* cntx )
{
	func_t* cntx_l3_vir_ukrs = bli_cntx_l3_vir_ukrs_buf( cntx );
	func_t* cntx_l3_vir_ukr  = &cntx_l3_vir_ukrs[ ukr ];

	bli_gks_get_l3_vir_ukr( method, ukr, cntx_l3_vir_ukr );
}

void bli_gks_cntx_set_l3_vir_ukrs( ind_t method, dim_t n_uk, ... )
{
	/* Example prototype:

	   void
	   bli_gks_cntx_set_l3_vir_ukrs( ind_t   method,
                                     dim_t   n_uk,
	                                 l3ukr_t ukr0_id,
	                                 l3ukr_t ukr1_id,
	                                 l3ukr_t ukr2_id,
	                                 ...
	                                 cntx_t* cntx );
	*/

	va_list   args;
	dim_t     i;
	l3ukr_t*  l3_ukrs;
	cntx_t*   cntx;

	// Allocate some temporary local arrays.
	l3_ukrs = bli_malloc_intl( n_uk * sizeof( l3ukr_t ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_uk );

	// Process n_uk kernel ids.
	for ( i = 0; i < n_uk; ++i )
	{
		// Here, we query the variable argument list for the kernel id.
		const l3ukr_t uk_id = va_arg( args, l3ukr_t  );

		// Store the value in our temporary array.
		l3_ukrs[ i ] = uk_id;
	}

	// The last argument should be the context pointer.
	cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Process each kernel id provided.
	for ( i = 0; i < n_uk; ++i )
	{
		// Read the current kernel id.
		const l3ukr_t uk_id = l3_ukrs[ i ];

		// Query the func_t associated with uk_id and save it directly into
		// the context.
		bli_gks_cntx_set_l3_vir_ukr( method, uk_id, cntx );
	}

	// Free the temporary local array.
	bli_free_intl( l3_ukrs );
}


//
// -- level-3 micro-kernel preferences -----------------------------------------
//

static mbool_t bli_gks_l3_ukrs_prefs[BLIS_NUM_LEVEL3_UKRS] =
{
/* gemm       */  { { BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS,
                      BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS,
                      BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS,
                      BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS, } },
/* gemmtrsm_l */  { { FALSE, FALSE, FALSE, FALSE, } },
/* gemmtrsm_u */  { { FALSE, FALSE, FALSE, FALSE, } },
/* trsm_l     */  { { FALSE, FALSE, FALSE, FALSE, } },
/* trsm_u     */  { { FALSE, FALSE, FALSE, FALSE, } },
};

// -----------------------------------------------------------------------------

void bli_gks_get_l3_nat_ukr_prefs( l3ukr_t  ukr,
                                   mbool_t* mbool )
{
	*mbool = bli_gks_l3_ukrs_prefs[ ukr ];
}

void bli_gks_cntx_set_l3_nat_ukr_prefs( l3ukr_t ukr,
                                        cntx_t* cntx )
{
	mbool_t* cntx_l3_nat_ukr_prefs = bli_cntx_l3_nat_ukrs_prefs_buf( cntx );
	mbool_t* cntx_l3_nat_ukr_pref  = &cntx_l3_nat_ukr_prefs[ ukr ];

	bli_gks_get_l3_nat_ukr_prefs( ukr, cntx_l3_nat_ukr_pref );
}


#if 0
//
// -- packm structure-aware kernel structure -----------------------------------
//

static func_t bli_gks_packm_struc_kers[BLIS_NUM_PACK_SCHEMA_TYPES] =
{
                /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
// row/col vectors
                 { NULL,                      NULL,                
                   NULL,                      NULL,                      },
// row/col panels
                 { bli_spackm_struc_cxk,      bli_cpackm_struc_cxk,
                   bli_dpackm_struc_cxk,      bli_zpackm_struc_cxk,      },
// row/col panels: 4m interleaved
                 { NULL,                      bli_cpackm_struc_cxk_4mi,
                   NULL,                      bli_zpackm_struc_cxk_4mi,  },
// row/col panels: 4m separated (NOT IMPLEMENTED)
                 { NULL,                      NULL,                    
                   NULL,                      NULL,                      },
// row/col panels: 3m interleaved
                 { NULL,                      bli_cpackm_struc_cxk_3mis,
                   NULL,                      bli_zpackm_struc_cxk_3mis, },
// row/col panels: 3m separated
                 { NULL,                      bli_cpackm_struc_cxk_3mis,
                   NULL,                      bli_zpackm_struc_cxk_3mis, },
// row/col panels: real only
                 { NULL,                      bli_cpackm_struc_cxk_rih,
                   NULL,                      bli_zpackm_struc_cxk_rih,  },
// row/col panels: imaginary only
                 { NULL,                      bli_cpackm_struc_cxk_rih,
                   NULL,                      bli_zpackm_struc_cxk_rih,  },
// row/col panels: real+imaginary only
                 { NULL,                      bli_cpackm_struc_cxk_rih,
                   NULL,                      bli_zpackm_struc_cxk_rih,  },
};

// -----------------------------------------------------------------------------

void bli_gks_get_packm_struc_ker( pack_t  schema,
                                  func_t* func )
{
	const dim_t i = bli_pack_schema_index( schema );

	*func = bli_gks_packm_struc_kers[ i ];
}

void bli_gks_cntx_set_packm_struc_ker( pack_t  schema,
                                       cntx_t* cntx )
{
	func_t* cntx_packm_ukr = bli_cntx_packm_ukrs( cntx );

	bli_gks_get_packm_struc_kers( schema, cntx_packm_ukr );
}
#endif


//
// -- level-1f kernel structure ------------------------------------------------
//

static func_t bli_gks_l1f_kers[BLIS_NUM_LEVEL1F_KERS] =
{
                /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
/* axpy2v     */ { { BLIS_SAXPY2V_KERNEL, BLIS_CAXPY2V_KERNEL,
                     BLIS_DAXPY2V_KERNEL, BLIS_ZAXPY2V_KERNEL, }
                 },
/* dotaxpyv   */ { { BLIS_SDOTAXPYV_KERNEL, BLIS_CDOTAXPYV_KERNEL,
                     BLIS_DDOTAXPYV_KERNEL, BLIS_ZDOTAXPYV_KERNEL, }
                 },
/* axpyf      */ { { BLIS_SAXPYF_KERNEL, BLIS_CAXPYF_KERNEL,
                     BLIS_DAXPYF_KERNEL, BLIS_ZAXPYF_KERNEL, }
                 },
/* dotxf      */ { { BLIS_SDOTXF_KERNEL, BLIS_CDOTXF_KERNEL,
                     BLIS_DDOTXF_KERNEL, BLIS_ZDOTXF_KERNEL, }
                 },
/* dotxaxpyf  */ { { BLIS_SDOTXAXPYF_KERNEL, BLIS_CDOTXAXPYF_KERNEL,
                     BLIS_DDOTXAXPYF_KERNEL, BLIS_ZDOTXAXPYF_KERNEL, }
                 },
};

static func_t bli_gks_l1f_ref_kers[BLIS_NUM_LEVEL1F_KERS] =
{
                /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
/* axpy2v     */ { { BLIS_SAXPY2V_KERNEL_REF, BLIS_CAXPY2V_KERNEL_REF,
                     BLIS_DAXPY2V_KERNEL_REF, BLIS_ZAXPY2V_KERNEL_REF, }
                 },
/* dotaxpyv   */ { { BLIS_SDOTAXPYV_KERNEL_REF, BLIS_CDOTAXPYV_KERNEL_REF,
                     BLIS_DDOTAXPYV_KERNEL_REF, BLIS_ZDOTAXPYV_KERNEL_REF, }
                 },
/* axpyf      */ { { BLIS_SAXPYF_KERNEL_REF, BLIS_CAXPYF_KERNEL_REF,
                     BLIS_DAXPYF_KERNEL_REF, BLIS_ZAXPYF_KERNEL_REF, }
                 },
/* dotxf      */ { { BLIS_SDOTXF_KERNEL_REF, BLIS_CDOTXF_KERNEL_REF,
                     BLIS_DDOTXF_KERNEL_REF, BLIS_ZDOTXF_KERNEL_REF, }
                 },
/* dotxaxpyf  */ { { BLIS_SDOTXAXPYF_KERNEL_REF, BLIS_CDOTXAXPYF_KERNEL_REF,
                     BLIS_DDOTXAXPYF_KERNEL_REF, BLIS_ZDOTXAXPYF_KERNEL_REF, }
                 },
};

// -----------------------------------------------------------------------------

void bli_gks_get_l1f_ker( l1fkr_t ker,
                          func_t* func )
{
	*func = bli_gks_l1f_kers[ ker ];
}

void bli_gks_get_l1f_ref_ker( l1fkr_t ker,
                              func_t* func )
{
	*func = bli_gks_l1f_ref_kers[ ker ];
}

void bli_gks_cntx_set_l1f_ker( l1fkr_t ker,
                               cntx_t* cntx )
{
	func_t* cntx_l1f_kers = bli_cntx_l1f_kers_buf( cntx );
	func_t* cntx_l1f_ker  = &cntx_l1f_kers[ ker ];

	bli_gks_get_l1f_ker( ker, cntx_l1f_ker );
}

void bli_gks_cntx_set_l1f_kers( dim_t n_kr, ... )
{
	/* Example prototype:

	   void
	   bli_gks_cntx_set_l1f_kers( dim_t   n_kr,
	                              l1fkr_t ker0_id,
	                              l1fkr_t ker1_id,
	                              l1fkr_t ker2_id,
	                              ...
	                              cntx_t* cntx );
	*/

	va_list   args;
	dim_t     i;
	l1fkr_t*  l1f_kers;
	cntx_t*   cntx;

	// Allocate some temporary local arrays.
	l1f_kers = bli_malloc_intl( n_kr * sizeof( l1fkr_t ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_kr );

	// Process n_kr kernel ids.
	for ( i = 0; i < n_kr; ++i )
	{
		// Here, we query the variable argument list for the kernel id.
		const l1fkr_t kr_id = va_arg( args, l1fkr_t  );

		// Store the value in our temporary array.
		l1f_kers[ i ] = kr_id;
	}

	// The last argument should be the context pointer.
	cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Process each kernel id provided.
	for ( i = 0; i < n_kr; ++i )
	{
		// Read the current kernel id.
		const l1fkr_t kr_id = l1f_kers[ i ];

		// Query the func_t associated with kr_id and save it directly into
		// the context.
		bli_gks_cntx_set_l1f_ker( kr_id, cntx );
	}

	// Free the temporary local array.
	bli_free_intl( l1f_kers );
}


//
// -- level-1v kernel structure ------------------------------------------------
//

static func_t bli_gks_l1v_kers[BLIS_NUM_LEVEL1V_KERS] =
{
                /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
/* addv       */ { { BLIS_SADDV_KERNEL, BLIS_CADDV_KERNEL,
                     BLIS_DADDV_KERNEL, BLIS_ZADDV_KERNEL, }
                 },
/* axpbyv     */ { { BLIS_SAXPBYV_KERNEL, BLIS_CAXPBYV_KERNEL,
                     BLIS_DAXPBYV_KERNEL, BLIS_ZAXPBYV_KERNEL, }
                 },
/* axpyv      */ { { BLIS_SAXPYV_KERNEL, BLIS_CAXPYV_KERNEL,
                     BLIS_DAXPYV_KERNEL, BLIS_ZAXPYV_KERNEL, }
                 },
/* copyv      */ { { BLIS_SCOPYV_KERNEL, BLIS_CCOPYV_KERNEL,
                     BLIS_DCOPYV_KERNEL, BLIS_ZCOPYV_KERNEL, }
                 },
/* dotv       */ { { BLIS_SDOTV_KERNEL, BLIS_CDOTV_KERNEL,
                     BLIS_DDOTV_KERNEL, BLIS_ZDOTV_KERNEL, }
                 },
/* dotxv      */ { { BLIS_SDOTXV_KERNEL, BLIS_CDOTXV_KERNEL,
                     BLIS_DDOTXV_KERNEL, BLIS_ZDOTXV_KERNEL, }
                 },
/* invertv    */ { { BLIS_SINVERTV_KERNEL, BLIS_CINVERTV_KERNEL,
                     BLIS_DINVERTV_KERNEL, BLIS_ZINVERTV_KERNEL, }
                 },
/* scalv      */ { { BLIS_SSCALV_KERNEL, BLIS_CSCALV_KERNEL,
                     BLIS_DSCALV_KERNEL, BLIS_ZSCALV_KERNEL, }
                 },
/* scal2v     */ { { BLIS_SSCAL2V_KERNEL, BLIS_CSCAL2V_KERNEL,
                     BLIS_DSCAL2V_KERNEL, BLIS_ZSCAL2V_KERNEL, }
                 },
/* setv       */ { { BLIS_SSETV_KERNEL, BLIS_CSETV_KERNEL,
                     BLIS_DSETV_KERNEL, BLIS_ZSETV_KERNEL, }
                 },
/* subv       */ { { BLIS_SSUBV_KERNEL, BLIS_CSUBV_KERNEL,
                     BLIS_DSUBV_KERNEL, BLIS_ZSUBV_KERNEL, }
                 },
/* swapv      */ { { BLIS_SSWAPV_KERNEL, BLIS_CSWAPV_KERNEL,
                     BLIS_DSWAPV_KERNEL, BLIS_ZSWAPV_KERNEL, }
                 },
/* xpbyv      */ { { BLIS_SXPBYV_KERNEL, BLIS_CXPBYV_KERNEL,
                     BLIS_DXPBYV_KERNEL, BLIS_ZXPBYV_KERNEL, }
                 },
};

static func_t bli_gks_l1v_ref_kers[BLIS_NUM_LEVEL1V_KERS] =
{
                /* float (0)  scomplex (1)  double (2)  dcomplex (3) */
/* addv       */ { { BLIS_SADDV_KERNEL_REF, BLIS_CADDV_KERNEL_REF,
                     BLIS_DADDV_KERNEL_REF, BLIS_ZADDV_KERNEL_REF, }
                 },
/* axpbyv     */ { { BLIS_SAXPBYV_KERNEL_REF, BLIS_CAXPBYV_KERNEL_REF,
                     BLIS_DAXPBYV_KERNEL_REF, BLIS_ZAXPBYV_KERNEL_REF, }
                 },
/* axpyv      */ { { BLIS_SAXPYV_KERNEL_REF, BLIS_CAXPYV_KERNEL_REF,
                     BLIS_DAXPYV_KERNEL_REF, BLIS_ZAXPYV_KERNEL_REF, }
                 },
/* copyv      */ { { BLIS_SCOPYV_KERNEL_REF, BLIS_CCOPYV_KERNEL_REF,
                     BLIS_DCOPYV_KERNEL_REF, BLIS_ZCOPYV_KERNEL_REF, }
                 },
/* dotv       */ { { BLIS_SDOTV_KERNEL_REF, BLIS_CDOTV_KERNEL_REF,
                     BLIS_DDOTV_KERNEL_REF, BLIS_ZDOTV_KERNEL_REF, }
                 },
/* dotxv      */ { { BLIS_SDOTXV_KERNEL_REF, BLIS_CDOTXV_KERNEL_REF,
                     BLIS_DDOTXV_KERNEL_REF, BLIS_ZDOTXV_KERNEL_REF, }
                 },
/* invertv    */ { { BLIS_SINVERTV_KERNEL_REF, BLIS_CINVERTV_KERNEL_REF,
                     BLIS_DINVERTV_KERNEL_REF, BLIS_ZINVERTV_KERNEL_REF, }
                 },
/* scalv      */ { { BLIS_SSCALV_KERNEL_REF, BLIS_CSCALV_KERNEL_REF,
                     BLIS_DSCALV_KERNEL_REF, BLIS_ZSCALV_KERNEL_REF, }
                 },
/* scal2v     */ { { BLIS_SSCAL2V_KERNEL_REF, BLIS_CSCAL2V_KERNEL_REF,
                     BLIS_DSCAL2V_KERNEL_REF, BLIS_ZSCAL2V_KERNEL_REF, }
                 },
/* setv       */ { { BLIS_SSETV_KERNEL_REF, BLIS_CSETV_KERNEL_REF,
                     BLIS_DSETV_KERNEL_REF, BLIS_ZSETV_KERNEL_REF, }
                 },
/* subv       */ { { BLIS_SSUBV_KERNEL_REF, BLIS_CSUBV_KERNEL_REF,
                     BLIS_DSUBV_KERNEL_REF, BLIS_ZSUBV_KERNEL_REF, }
                 },
/* swapv      */ { { BLIS_SSWAPV_KERNEL_REF, BLIS_CSWAPV_KERNEL_REF,
                     BLIS_DSWAPV_KERNEL_REF, BLIS_ZSWAPV_KERNEL_REF, }
                 },
/* xpbyv      */ { { BLIS_SXPBYV_KERNEL_REF, BLIS_CXPBYV_KERNEL_REF,
                     BLIS_DXPBYV_KERNEL_REF, BLIS_ZXPBYV_KERNEL_REF, }
                 },
};

// -----------------------------------------------------------------------------

void bli_gks_get_l1v_ker( l1vkr_t ker,
                          func_t* func )
{
	*func = bli_gks_l1v_kers[ ker ];
}

void bli_gks_get_l1v_ref_ker( l1vkr_t ker,
                              func_t* func )
{
	*func = bli_gks_l1v_ref_kers[ ker ];
}

void bli_gks_cntx_set_l1v_ker( l1vkr_t ker,
                               cntx_t* cntx )
{
	func_t* cntx_l1v_kers = bli_cntx_l1v_kers_buf( cntx );
	func_t* cntx_l1v_ker  = &cntx_l1v_kers[ ker ];

	bli_gks_get_l1v_ker( ker, cntx_l1v_ker );
}


void bli_gks_cntx_set_l1v_kers( dim_t n_kr, ... )
{
	/* Example prototype:

	   void
	   bli_gks_cntx_set_l1v_kers( dim_t   n_kr,
	                              l1vkr_t ker0_id,
	                              l1vkr_t ker1_id,
	                              l1vkr_t ker2_id,
	                              ...
	                              cntx_t* cntx );
	*/

	va_list   args;
	dim_t     i;
	l1vkr_t*  l1v_kers;
	cntx_t*   cntx;

	// Allocate some temporary local arrays.
	l1v_kers = bli_malloc_intl( n_kr * sizeof( l1vkr_t ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_kr );

	// Process n_kr kernel ids.
	for ( i = 0; i < n_kr; ++i )
	{
		// Here, we query the variable argument list for the kernel id.
		const l1vkr_t kr_id = va_arg( args, l1vkr_t  );

		// Store the value in our temporary array.
		l1v_kers[ i ] = kr_id;
	}

	// The last argument should be the context pointer.
	cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Process each kernel id provided.
	for ( i = 0; i < n_kr; ++i )
	{
		// Read the current kernel id.
		const l1vkr_t kr_id = l1v_kers[ i ];

		// Query the func_t associated with kr_id and save it directly into
		// the context.
		bli_gks_cntx_set_l1v_ker( kr_id, cntx );
	}

	// Free the temporary local array.
	bli_free_intl( l1v_kers );
}


//
// -- level-3 micro-kernel implementation strings ------------------------------
//

static char* bli_gks_l3_ukr_impl_str[BLIS_NUM_UKR_IMPL_TYPES] =
{
	"refrnce",
	"virtual",
	"optimzd",
	"notappl",
};

// -----------------------------------------------------------------------------

char* bli_gks_l3_ukr_impl_string( l3ukr_t ukr, ind_t method, num_t dt )
{
	func_t  p;
	kimpl_t ki;

	// Query the func_t for the given ukr type and method.
	bli_gks_get_l3_vir_ukr( method, ukr, &p );

	// Check whether the ukrs func_t is NULL for the given ukr type and
	// datatype. If the queried ukr func_t is NULL, return the string
	// for not applicable. Otherwise, query the ukernel implementation
	// type using the method provided and return the associated string.
	if ( bli_func_is_null_dt( dt, &p ) )
		ki = BLIS_NOTAPPLIC_UKERNEL;
	else
		ki = bli_gks_l3_ukr_impl_type( ukr, method, dt );

	return bli_gks_l3_ukr_impl_str[ ki ];
}

#if 0
char* bli_gks_l3_ukr_avail_impl_string( l3ukr_t ukr, num_t dt )
{
	opid_t  oper;
	ind_t   method;
	kimpl_t ki;

	// We need to decide which operation we will use to query the
	// current available induced method. If the ukr type given is
	// BLIS_GEMM_UKR, we use gemm. Otherwise, we use trsm (since
	// the four other defined ukr types are trsm-related).
	if ( ukr == BLIS_GEMM_UKR ) oper = BLIS_GEMM;
	else                        oper = BLIS_TRSM;

	// Query the current available induced method using the
	// chosen operation id type.
	method = bli_l3_ind_oper_find_avail( oper, dt );

	// Query the ukernel implementation type using the current
	// available method.
	ki = bli_gks_l3_ukr_impl_type( ukr, method, dt );

	return bli_ukr_impl_str[ ki ];
}
#endif

kimpl_t bli_gks_l3_ukr_impl_type( l3ukr_t ukr, ind_t method, num_t dt )
{
	// If the current available induced method is not native, it
	// must be virtual.
	if ( method != BLIS_NAT ) return BLIS_VIRTUAL_UKERNEL;
	else
	{
		// If the current available induced method for the gemm
		// operation is native, then it might be reference or
		// optimized. To determine which, we compare the
		// datatype-specific function pointer within the ukrs
		// object corresponding to the current available induced
		// method to the typed function pointer within the known
		// reference ukrs object.

		func_t  funcs;
		func_t  ref_funcs;
		void*   p;
		void*   ref_p;

		bli_gks_get_l3_vir_ukr( method, ukr, &funcs );
		bli_gks_get_l3_ref_ukr( ukr, &ref_funcs );

		p     = bli_func_get_dt( dt, &funcs );
		ref_p = bli_func_get_dt( dt, &ref_funcs );
	
		if ( p == ref_p ) return BLIS_REFERENCE_UKERNEL;
		else              return BLIS_OPTIMIZED_UKERNEL;
	}
}

