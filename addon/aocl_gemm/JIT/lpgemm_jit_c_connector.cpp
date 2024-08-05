/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

//#include "libjit_c_connector.h"
#include "blis.h"
#include "lpgemm_jit_bf16.h"


#ifdef __cplusplus
extern "C" {
#endif

static bli_lpgemm_jit *lpgemm_jit_objs[LPGEMM_BF16_MR][LPGEMM_BF16_NR];

void get_jit_kernel( lpgemm_jit_inputs_t *params,
                     void* buffer,
                     dim_t bufferSize
                   )
{
    dim_t m_idx = ( params->MR ) % LPGEMM_BF16_MR;
    dim_t n_idx = ( params->NR ) / NUM_F32_ELEMS_PER_ZMM;
    lpgemm_jit_objs[m_idx][n_idx] = new bli_lpgemm_jit( buffer, bufferSize );
    lpgemm_jit_objs[m_idx][n_idx]->generate_kernel( params );
}

void* get_jit_code( lpgemm_jit_inputs_t *params )
{
    dim_t m_idx = ( params->MR ) % LPGEMM_BF16_MR;
    dim_t n_idx = ( params->NR ) / NUM_F32_ELEMS_PER_ZMM;
    return ((void*) lpgemm_jit_objs[m_idx][n_idx]->get_code() );
}

dim_t get_kernel_size( lpgemm_jit_inputs_t *params )
{
    dim_t m_idx = ( params->MR ) % LPGEMM_BF16_MR;
    dim_t n_idx = ( params->NR ) / NUM_F32_ELEMS_PER_ZMM;
    return lpgemm_jit_objs[m_idx][n_idx]->get_size();
}
#ifdef __cplusplus
}
#endif
