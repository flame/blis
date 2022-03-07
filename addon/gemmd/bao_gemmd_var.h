/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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
// Prototype the object-based variant interfaces.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTECH(bao_,opname) \
     ( \
       const obj_t*  alpha, \
       const obj_t*  a, \
       const obj_t*  d, \
       const obj_t*  b, \
       const obj_t*  beta, \
       const obj_t*  c, \
       const cntx_t* cntx, \
       const rntm_t* rntm, \
       const thrinfo_t* thread  \
     );

GENPROT( gemmd_bp_var1 )


//
// Prototype the typed variant interfaces.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTECH2(bao_,ch,varname) \
     ( \
       conj_t        conja, \
       conj_t        conjb, \
       dim_t         m, \
       dim_t         n, \
       dim_t         k, \
       const void*   alpha, \
       const void*   a, inc_t rs_a, inc_t cs_a, \
       const void*   d, inc_t incd, \
       const void*   b, inc_t rs_b, inc_t cs_b, \
       const void*   beta, \
             void*   c, inc_t rs_c, inc_t cs_c, \
       const cntx_t* cntx, \
       const rntm_t* rntm, \
       const thrinfo_t* thread  \
     );

//INSERT_GENTPROT_BASIC0( gemmd_bp_var1 )
GENTPROT( float,    s, gemmd_bp_var1 )
GENTPROT( double,   d, gemmd_bp_var1 )
GENTPROT( scomplex, c, gemmd_bp_var1 )
GENTPROT( dcomplex, z, gemmd_bp_var1 )


//
// Prototype the typed kernel interfaces.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTECH2(bao_,ch,varname) \
     ( \
       const dim_t      MR, \
       const dim_t      NR, \
       dim_t            mr_cur, \
       dim_t            nr_cur, \
       dim_t            k, \
       const ctype*     alpha, \
       const ctype*     a, inc_t rs_a, inc_t cs_a, \
       const ctype*     b, inc_t rs_b, inc_t cs_b, \
       const ctype*     beta, \
             ctype*     c, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* aux, \
       const cntx_t*    cntx  \
     );

//INSERT_GENTPROT_BASIC0( gemm_kernel )
GENTPROT( float,    s, gemm_kernel )
GENTPROT( double,   d, gemm_kernel )
GENTPROT( scomplex, c, gemm_kernel )
GENTPROT( dcomplex, z, gemm_kernel )

