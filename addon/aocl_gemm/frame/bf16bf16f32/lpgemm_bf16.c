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
#include "lpgemm_5loop_interface_apis.h"
#include "lpgemm_packb_bf16.h"
#include "lpgemm_kernels.h"
#include "lpgemm_utils.h"
#include "lpgemm_thrinfo_utils.h"
#include "lpgemm_config.h"

// B should always be packed.
LPGEMM_5LOOP(bfloat16,bfloat16,float,bf16bf16f32of32)
{
    dim_t NC = lpgemm_get_block_size_NC_global_cntx( BF16BF16F32OF32 );
    dim_t KC = lpgemm_get_block_size_KC_global_cntx( BF16BF16F32OF32 );
    dim_t MC = lpgemm_get_block_size_MC_global_cntx( BF16BF16F32OF32 );
    dim_t NR = lpgemm_get_block_size_NR_global_cntx( BF16BF16F32OF32 );
	
    const int16_t* a_use = NULL;
    dim_t cs_a_use = cs_a;
    dim_t a_block_stride = 0;

    const int16_t* b_use = NULL;
    dim_t rs_b_use = rs_b;
    dim_t cs_b_use = cs_b;
	
    float* c_use_jc = NULL;
    float* c_use_ic = NULL;

    // kc needs to be a multiple of 2 so that it can be used with dpbf16_ps
    // instruction. Padding is added in cases this condition is not
    // satisfied, and therefore the k offset used for packed/reordered
    // buffer needs to be updated.
    dim_t k_updated = k;
    k_updated += (k_updated & 0x1);

    // Is required to decide whether to apply post ops or not.
    bool is_last_k = FALSE;

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
			float beta0 = ( pc == 0 ) ? beta : 1;
			dim_t kc0 = ( ( pc + KC ) <= k ) ? KC : ( k % KC );

			// kc0 needs to be a multiple of 2 so that it can be
			// used with dpbf16_ps instruction. Padding is added in
			// cases this condition is not satisfied, and therefore
			// the kc0 offsets used for packed/reordered buffers
			// needs to be updated.
			dim_t kc0_updated = kc0;
			kc0_updated += (kc0_updated & 0x1);

      		is_last_k = ( ( pc + KC ) >= k ) ? ( TRUE ) : ( FALSE );

			// B part getting processed
      		if ( mtag_b == REORDERED )
			{
				b_use = b + ( jc * k_updated ) + ( pc * nc0_updated );
				get_packb_nr64_bf16bf16f32of32_strides( &rs_b_use, &cs_b_use );
			}

			for ( dim_t ic = 0; ic < m; ic += MC )
			{
				dim_t mc0 = ( ( ic + MC ) <= m ) ? MC : ( m % MC );

				a_use = a + ( rs_a * ic ) + ( cs_a * pc );

				// bf16 kernel reads 2 elements, totalling 4 bytes in a
				// single broadcast for use in bf16 instruction.
				// Non bf16 based kernel requires update to this code.
				cs_a_use = 2;
				a_block_stride = rs_a;

				for ( dim_t jr = 0; jr < nc0; jr += NR )
				{
					dim_t nr0 = ( ( jr + NR ) <= nc0 ) ? NR : ( nc0 % NR );

					// Reorder/Packed B, Reorder/Packed/Unpacked A call.
					lpgemm_rowvar_bf16bf16f32of32_6x64 
          			(
            		 mc0, nr0, kc0, 
                     a_use, rs_a, cs_a_use, a_block_stride, 
                     ( b_use + ( jr * kc0_updated ) ), rs_b_use, cs_b_use, 
                     ( c + ( rs_c * ic ) + jc + jr ), rs_c, 1, 
                     alpha, beta0, 
                     is_last_k, ic, ( jc + jr ), post_op_list
          			);
				}
			}
		}
	}
}
