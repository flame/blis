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

// 3mh blocksizes
extern blksz_t* gemm3mh_mc;
extern blksz_t* gemm3mh_nc;
extern blksz_t* gemm3mh_kc;
extern blksz_t* gemm3mh_mr;
extern blksz_t* gemm3mh_nr;
extern blksz_t* gemm3mh_kr;

// 3m3 blocksizes
extern blksz_t* gemm3m3_mc;
extern blksz_t* gemm3m3_nc;
extern blksz_t* gemm3m3_kc;
extern blksz_t* gemm3m3_mr;
extern blksz_t* gemm3m3_nr;
extern blksz_t* gemm3m3_kr;

// 3m2 blocksizes
extern blksz_t* gemm3m2_mc;
extern blksz_t* gemm3m2_nc;
extern blksz_t* gemm3m2_kc;
extern blksz_t* gemm3m2_mr;
extern blksz_t* gemm3m2_nr;
extern blksz_t* gemm3m2_kr;

// 3m1 blocksizes
extern blksz_t* gemm3m1_mc;
extern blksz_t* gemm3m1_nc;
extern blksz_t* gemm3m1_kc;
extern blksz_t* gemm3m1_mr;
extern blksz_t* gemm3m1_nr;
extern blksz_t* gemm3m1_kr;

// 4mh blocksizes
extern blksz_t* gemm4mh_mc;
extern blksz_t* gemm4mh_nc;
extern blksz_t* gemm4mh_kc;
extern blksz_t* gemm4mh_mr;
extern blksz_t* gemm4mh_nr;
extern blksz_t* gemm4mh_kr;

// 4m1b blocksizes
extern blksz_t* gemm4mb_mc;
extern blksz_t* gemm4mb_nc;
extern blksz_t* gemm4mb_kc;
extern blksz_t* gemm4mb_mr;
extern blksz_t* gemm4mb_nr;
extern blksz_t* gemm4mb_kr;

// 4m1a blocksizes
extern blksz_t* gemm4m1_mc;
extern blksz_t* gemm4m1_nc;
extern blksz_t* gemm4m1_kc;
extern blksz_t* gemm4m1_mr;
extern blksz_t* gemm4m1_nr;
extern blksz_t* gemm4m1_kr;

// Native blocksizes
extern blksz_t* gemm_mc;
extern blksz_t* gemm_nc;
extern blksz_t* gemm_kc;
extern blksz_t* gemm_mr;
extern blksz_t* gemm_nr;
extern blksz_t* gemm_kr;

//
// NOTE: We have to use the address of the blksz_t*, since the value
// will not yet be set at compile-time (since they are allocated at
// runtime).
//
static blksz_t** bli_bsizes[BLIS_NUM_IND_METHODS][BLIS_NUM_LEVEL3_BLKSZS] =
{
        /*   mc/mr        nc/nr        kc/kr   */
/* 3mh  */ { &gemm3mh_mc, &gemm3mh_nc, &gemm3mh_kc,
             &gemm3mh_mr, &gemm3mh_nr, &gemm3mh_kr },
/* 3m3  */ { &gemm3m3_mc, &gemm3m3_nc, &gemm3m3_kc,
             &gemm3m3_mr, &gemm3m3_nr, &gemm3m3_kr },
/* 3m2  */ { &gemm3m2_mc, &gemm3m2_nc, &gemm3m2_kc,
             &gemm3m2_mr, &gemm3m2_nr, &gemm3m2_kr },
/* 3m1  */ { &gemm3m1_mc, &gemm3m1_nc, &gemm3m1_kc,
             &gemm3m1_mr, &gemm3m1_nr, &gemm3m1_kr },
/* 4mh  */ { &gemm4mh_mc, &gemm4mh_nc, &gemm4mh_kc,
             &gemm4mh_mr, &gemm4mh_nr, &gemm4mh_kr },
/* 4mb  */ { &gemm4mb_mc, &gemm4mb_nc, &gemm4mb_kc,
             &gemm4mb_mr, &gemm4mb_nr, &gemm4mb_kr },
/* 4m1  */ { &gemm4m1_mc, &gemm4m1_nc, &gemm4m1_kc,
             &gemm4m1_mr, &gemm4m1_nr, &gemm4m1_kr },
/* nat  */ { &gemm_mc,    &gemm_nc,    &gemm_kc,
             &gemm_mr,    &gemm_nr,    &gemm_kr },
};

// -----------------------------------------------------------------------------

dim_t bli_bsv_get_avail_blksz_dt( bszid_t bsv, opid_t oper, num_t dt )
{
	// Query the blksz_t object corresponding to the requested
	// blocksize id type and datatype (for the current available
	// induced method of the given operation).
	blksz_t* b = bli_bsv_get_avail_blksz( bsv, oper, dt );

	// Return the default blocksize associated with the given datatype.
	return bli_blksz_get_def( dt, b );
}

// -----------------------------------------------------------------------------

dim_t bli_bsv_get_avail_blksz_max_dt( bszid_t bsv, opid_t oper, num_t dt )
{
	// Query the blksz_t object corresponding to the requested
	// blocksize id type and datatype (for the current available
	// induced method of the given operation).
	blksz_t* b = bli_bsv_get_avail_blksz( bsv, oper, dt );

	// Return the maximum blocksize associated with the given datatype.
	return bli_blksz_get_max( dt, b );
}

// -----------------------------------------------------------------------------

blksz_t* bli_bsv_get_avail_blksz( bszid_t bsv, opid_t oper, num_t dt )
{
	// Query the current available induced method for the operation
	// and datatype given.
	ind_t method = bli_ind_oper_find_avail( oper, dt );

	// Return a pointer to the blksz_t object corresponding to the
	// blocksize id type for the current available induced method.
	return bli_bsv_get_blksz( bsv, method );
}

// -----------------------------------------------------------------------------

blksz_t* bli_bsv_get_blksz( bszid_t bsv, ind_t method )
{
	// Initialize the cntl API, if it isn't already initialized. This is
	// needed because we have to ensure that the blksz_t objects have
	// been created, otherwise this function could return a NULL (or
	// garbage) address.
	bli_cntl_init();

	return *(bli_bsizes[ method ][ bsv ]);
}

