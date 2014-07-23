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

/*
void bli_dotxaxpyf_ref_var1( obj_t*  alpha,
                             obj_t*  at,
                             obj_t*  a,
                             obj_t*  w,
                             obj_t*  x,
                             obj_t*  beta,
                             obj_t*  y,
                             obj_t*  z );
*/


#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_a, ctype_b, ctype_c, ctype_ab, cha, chb, chc, chab, varname ) \
\
void PASTEMAC3(cha,chb,chc,varname) \
     ( \
       conj_t             conjat, \
       conj_t             conja, \
       conj_t             conjw, \
       conj_t             conjx, \
       dim_t              m, \
       dim_t              b_n, \
       ctype_ab* restrict alpha, \
       ctype_a*  restrict a, inc_t inca, inc_t lda, \
       ctype_b*  restrict w, inc_t incw, \
       ctype_b*  restrict x, inc_t incx, \
       ctype_c*  restrict beta, \
       ctype_c*  restrict y, inc_t incy, \
       ctype_c*  restrict z, inc_t incz  \
     );

INSERT_GENTPROT3U12_BASIC( dotxaxpyf_ref_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT3U12_MIX_D( dotxaxpyf_ref_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT3U12_MIX_P( dotxaxpyf_ref_var1 )
#endif

