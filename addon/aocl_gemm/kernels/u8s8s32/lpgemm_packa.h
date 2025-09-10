/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_GEMM_INT8_PACKA
#define BLIS_GEMM_INT8_PACKA

// The strides needs to be updated based on the m_fringe value to account
// for different schemas used to pack A fringe cases.
BLIS_INLINE void get_packa_strides_mfringe_u8s8s32os32
     (
       const dim_t rs,
       const dim_t cs,
       dim_t* rs_use,
       dim_t* cs_use,
       dim_t MR,
       dim_t m_fringe
     )
{
	// Only applicable for row major packing.
	if ( ( rs != 1 ) && ( cs == 1 ) && ( ( *cs_use ) > MR ))
	{
		( *rs_use ) = 4;
		( *cs_use ) = ( ( *cs_use ) / MR ) * m_fringe;
	}
}

typedef void (*packa_s32)
     (
       uint8_t*,
       const uint8_t*,
       const dim_t,
       const dim_t,
       const dim_t,
       const dim_t,
       dim_t*,
       dim_t*
     );

void packa_u8s8s32os32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    rs,
       const dim_t    cs,
       const dim_t    MC,
       const dim_t    KC,
       dim_t*         rs_a,
       dim_t*         cs_a
     );

#endif //BLIS_GEMM_INT8_PACKA
