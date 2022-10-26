/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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


void bli_packm_sup_init_mem
     (
       bool       will_pack,
       packbuf_t  pack_buf_type,
       num_t      dt,
       dim_t      m,
       dim_t      k,
       dim_t      mr,
       thrinfo_t* thread
     );

void bli_packm_sup_finalize_mem
     (
       bool       did_pack,
       thrinfo_t* thread
     );

void bli_packm_sup_init
     (
             bool       will_pack,
             stor3_t    stor_id,
             pack_t*    schema,
             dim_t      m,
             dim_t      k,
             dim_t      mr,
             dim_t*     m_max,
             dim_t*     k_max,
       const void*      x, inc_t  rs_x, inc_t  cs_x,
             void**     p, inc_t* rs_p, inc_t* cs_p,
                           dim_t* pd_p, inc_t* ps_p,
             thrinfo_t* thread
     );

void bli_packm_sup
     (
             bool       will_pack,
             packbuf_t  pack_buf_type,
             stor3_t    stor_id,
             trans_t    transc,
             num_t      dt,
             dim_t      m_alloc,
             dim_t      k_alloc,
             dim_t      m,
             dim_t      k,
             dim_t      mr,
       const void*      kappa,
       const void*      a, inc_t  rs_a, inc_t  cs_a,
             void**     p, inc_t* rs_p, inc_t* cs_p,
                           inc_t* ps_p,
       const cntx_t*    cntx,
             thrinfo_t* thread
     );

