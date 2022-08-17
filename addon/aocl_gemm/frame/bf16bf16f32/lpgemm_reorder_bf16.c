/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_utils.h"
#include "lpgemm_reorder_bf16.h"
#include "lpgemm_packb_bf16.h"
#include "lpgemm_config.h"
#include "aocl_bf16_type.h"

void reorderb_nr64_bf16bf16f32of32
  (
    lpgemm_obj_t *b,
    lpgemm_obj_t *b_reorder
  )
{   
    dim_t NC = lpgemm_get_block_size_NC_global_cntx( BF16BF16F32OF32 );
    dim_t KC = lpgemm_get_block_size_KC_global_cntx( BF16BF16F32OF32 );

    // Extracting the matrix properties from the lpgemm object
    dim_t rs_b = b->rs;
    dim_t n = b->width;
    dim_t k = b->length;

    dim_t rs_b_reorder;
    dim_t cs_b_reorder;

    // k needs to be a multiple of 2 so that it can be used with vpdpbusd
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    dim_t k_updated = k;
    k_updated += (k_updated & 0x1);

	for ( dim_t jc = 0; jc < n; jc += NC )
	{
		dim_t nc0 = ( ( jc + NC ) <= n ) ? NC : ( n % NC );

		dim_t nc0_mod16 = nc0 % 16;
		dim_t nc0_updated = nc0;
		if ( nc0_mod16 != 0 )
		{
			nc0_updated += ( 16 - nc0_mod16 );
		}
		for ( dim_t pc = 0; pc < k; pc += KC )
		{
			dim_t kc0 = ( ( pc + KC ) <= k ) ? KC : ( k % KC );      
			// B should always be packed.		
			packb_nr64_bf16bf16f32of32                                         
			(
			 ( ( ( bfloat16* )b_reorder->storage.aligned_buffer ) + ( jc * k_updated ) +
			   ( nc0_updated * pc ) ),
			 ( ( ( bfloat16* )b->storage.aligned_buffer ) + ( rs_b * pc ) + jc ),
			 rs_b, nc0, kc0, &rs_b_reorder, &cs_b_reorder
			);		
		}
	}

	b_reorder->rs = rs_b_reorder;
	b_reorder->cs = cs_b_reorder;
	b_reorder->mtag = REORDERED;
}  
