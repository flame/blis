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

extern scalv_t*   scalv_cntl;
extern packm_t*   packm_cntl_noscale;
extern packv_t*   packv_cntl;
extern unpackv_t* unpackv_cntl;

static blksz_t*   gemv_mc;
static blksz_t*   gemv_nc;

gemv_t*           gemv_cntl_bs_ke_dot;
gemv_t*           gemv_cntl_bs_ke_axpy;

gemv_t*           gemv_cntl_rp_bs_dot;
gemv_t*           gemv_cntl_rp_bs_axpy;

gemv_t*           gemv_cntl_cp_bs_dot;
gemv_t*           gemv_cntl_cp_bs_axpy;

gemv_t*           gemv_cntl_ge_dot;
gemv_t*           gemv_cntl_ge_axpy;

// Cache blocksizes.

#define BLIS_GEMV_MC_S 1000
#define BLIS_GEMV_MC_D 1000
#define BLIS_GEMV_MC_C 1000
#define BLIS_GEMV_MC_Z 1000

#define BLIS_GEMV_NC_S 1000
#define BLIS_GEMV_NC_D 1000
#define BLIS_GEMV_NC_C 1000
#define BLIS_GEMV_NC_Z 1000



void bl2_gemv_cntl_init()
{
	// Create blocksize objects for each dimension.
	gemv_mc = bl2_blksz_obj_create( BLIS_GEMV_MC_S,
	                                BLIS_GEMV_MC_D,
	                                BLIS_GEMV_MC_C,
	                                BLIS_GEMV_MC_Z );

	gemv_nc = bl2_blksz_obj_create( BLIS_GEMV_NC_S,
	                                BLIS_GEMV_NC_D,
	                                BLIS_GEMV_NC_C,
	                                BLIS_GEMV_NC_Z );


	// Create control trees for the lowest-level kernels. These trees induce
	// operations on (persumably) relatively small block-subvector problems.
	gemv_cntl_bs_ke_dot
	=
	bl2_gemv_cntl_obj_create( BLIS_UNB_FUSED,
	                          BLIS_VARIANT1,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL,
	                          NULL );
	gemv_cntl_bs_ke_axpy
	=
	bl2_gemv_cntl_obj_create( BLIS_UNB_FUSED,
	                          BLIS_VARIANT2,
	                          NULL, NULL, NULL,
	                          NULL, NULL, NULL,
	                          NULL );


	// Create control trees for problems with relatively small m dimension
	// (ie: where trans(A) is a row panel problem).
	gemv_cntl_rp_bs_dot
	=
	bl2_gemv_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemv_nc,
	                          scalv_cntl,         // scale y up-front
	                          packm_cntl_noscale, // pack A1 (if needed)
	                          packv_cntl,         // pack x1 (if needed)
	                          NULL,               // y is not partitioned in var2
	                          gemv_cntl_bs_ke_dot,
	                          NULL );             // y is not partitioned in var2
	gemv_cntl_rp_bs_axpy
	=
	bl2_gemv_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT2,
	                          gemv_nc,
	                          scalv_cntl,         // scale y up-front
	                          packm_cntl_noscale, // pack A1 (if needed)
	                          packv_cntl,         // pack x1 (if needed)
	                          NULL,               // y is not partitioned in var2
	                          gemv_cntl_bs_ke_axpy,
	                          NULL );             // y is not partitioned in var2


	// Create control trees for problems with relatively small n dimension
	// (ie: where trans(A) is a column panel problem).
	gemv_cntl_cp_bs_dot
	=
	bl2_gemv_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemv_mc,
	                          NULL,               // no scaling in blk_var1
	                          packm_cntl_noscale, // pack A1 (if needed)
	                          NULL,               // x is not partitioned in var1
	                          packv_cntl,         // pack y1 (if needed)
	                          gemv_cntl_bs_ke_dot,
	                          unpackv_cntl );     // unpack y1 (if packed)
	gemv_cntl_cp_bs_axpy
	=
	bl2_gemv_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemv_mc,
	                          NULL,               // no scaling in blk_var1
	                          packm_cntl_noscale, // pack A1 (if needed)
	                          NULL,               // x is not partitioned in var1
	                          packv_cntl,         // pack y1 (if needed)
	                          gemv_cntl_bs_ke_axpy,
	                          unpackv_cntl );     // unpack y1 (if packed)


	// Create control trees for generally large problems. Here, we choose a
	// variant that partitions subproblems into row panels.
	gemv_cntl_ge_dot
	=
	bl2_gemv_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemv_mc,
	                          NULL,               // no scaling in blk_var1
	                          NULL,               // do not pack A1
	                          NULL,               // x is not partitioned in var1
	                          packv_cntl,         // pack y1 (if needed)
	                          gemv_cntl_rp_bs_dot,
	                          unpackv_cntl );     // unpack y1 (if packed)
	gemv_cntl_ge_axpy
	=
	bl2_gemv_cntl_obj_create( BLIS_BLOCKED,
	                          BLIS_VARIANT1,
	                          gemv_mc,
	                          NULL,               // no scaling in blk_var1
	                          NULL,               // do not pack A1
	                          NULL,               // x is not partitioned in var1
	                          packv_cntl,         // pack y1 (if needed)
	                          gemv_cntl_rp_bs_axpy,
	                          unpackv_cntl );     // unpack y1 (if packed)
}

void bl2_gemv_cntl_finalize()
{
	bl2_blksz_obj_free( gemv_mc );
	bl2_blksz_obj_free( gemv_nc );

	bl2_cntl_obj_free( gemv_cntl_bs_ke_dot );
	bl2_cntl_obj_free( gemv_cntl_bs_ke_axpy );

	bl2_cntl_obj_free( gemv_cntl_rp_bs_dot );
	bl2_cntl_obj_free( gemv_cntl_rp_bs_axpy );

	bl2_cntl_obj_free( gemv_cntl_cp_bs_dot );
	bl2_cntl_obj_free( gemv_cntl_cp_bs_axpy );

	bl2_cntl_obj_free( gemv_cntl_ge_dot );
	bl2_cntl_obj_free( gemv_cntl_ge_axpy );
}


gemv_t* bl2_gemv_cntl_obj_create( impl_t     impl_type,
                                  varnum_t   var_num,
                                  blksz_t*   b,
                                  scalv_t*   sub_scalv,
                                  packm_t*   sub_packm_a,
                                  packv_t*   sub_packv_x,
                                  packv_t*   sub_packv_y,
                                  gemv_t*    sub_gemv,
                                  unpackv_t* sub_unpackv_y )
{
	gemv_t* cntl;

	cntl = ( gemv_t* ) bl2_malloc( sizeof(gemv_t) );	

	cntl->impl_type     = impl_type;
	cntl->var_num       = var_num;
	cntl->b             = b;
	cntl->sub_scalv     = sub_scalv;
	cntl->sub_packm_a   = sub_packm_a;
	cntl->sub_packv_x   = sub_packv_x;
	cntl->sub_packv_y   = sub_packv_y;
	cntl->sub_gemv      = sub_gemv;
	cntl->sub_unpackv_y = sub_unpackv_y;

	return cntl;
}

void bl2_gemv_cntl_obj_init( gemv_t*    cntl,
                             impl_t     impl_type,
                             varnum_t   var_num,
                             blksz_t*   b,
                             scalv_t*   sub_scalv,
                             packm_t*   sub_packm_a,
                             packv_t*   sub_packv_x,
                             packv_t*   sub_packv_y,
                             gemv_t*    sub_gemv,
                             unpackv_t* sub_unpackv_y )
{
	cntl->impl_type     = impl_type;
	cntl->var_num       = var_num;
	cntl->b             = b;
	cntl->sub_scalv     = sub_scalv;
	cntl->sub_packm_a   = sub_packm_a;
	cntl->sub_packv_x   = sub_packv_x;
	cntl->sub_packv_y   = sub_packv_y;
	cntl->sub_gemv      = sub_gemv;
	cntl->sub_unpackv_y = sub_unpackv_y;
}

