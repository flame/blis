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


//
// Prototype object-based interfaces.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC0(opname) \
     ( \
       obj_t*  a, \
       obj_t*  ah, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       cntl_t* cntl, \
       thrinfo_t* thread  \
     );

GENPROT( gemmt_x_ker_var2 )

GENPROT( gemmt_l_ker_var2 )
GENPROT( gemmt_u_ker_var2 )


//
// Prototype BLAS-like interfaces with void pointer operands.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t  diagoffc, \
       pack_t  schema_a, \
       pack_t  schema_b, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       void*   alpha, \
       void*   a, inc_t cs_a, inc_t is_a, \
                  dim_t pd_a, inc_t ps_a, \
       void*   b, inc_t rs_b, inc_t is_b, \
                  dim_t pd_b, inc_t ps_b, \
       void*   beta, \
       void*   c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       thrinfo_t* thread  \
     );

INSERT_GENTPROT_BASIC0( gemmt_l_ker_var2 )
INSERT_GENTPROT_BASIC0( gemmt_u_ker_var2 )

