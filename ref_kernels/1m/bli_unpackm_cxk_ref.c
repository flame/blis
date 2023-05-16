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


#define UNPACKM_BODY( ctype, ch, pragma, cdim, inca, op ) \
\
do \
{ \
	for ( dim_t k = n; k != 0; --k ) \
	{ \
		pragma \
		for ( dim_t mn = 0; mn < cdim; mn++ ) \
			PASTEMAC(ch,op)( *kappa_cast, *(pi1 + mn*dfac), *(alpha1 + mn*inca) ); \
\
		alpha1 += lda; \
		pi1    += ldp; \
	} \
} while(0)


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mnr0, bb0, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             conj_t  conja, \
             pack_t  schema, \
             dim_t   cdim, \
             dim_t   n, \
       const void*   kappa, \
       const void*   p,             inc_t ldp, \
             void*   a, inc_t inca, inc_t lda, \
       const cntx_t* cntx  \
     ) \
{ \
	const dim_t     mnr        = PASTECH2(mnr0, _, ch); \
    /* It's not clear if unpack needs to care about BB storage... */ \
	const dim_t     dfac       = PASTECH2(bb0, _, ch); \
\
	const ctype* restrict kappa_cast = kappa; \
	const ctype* restrict pi1        = p; \
	      ctype* restrict alpha1     = a; \
\
	if ( cdim == mnr && mnr != -1 ) \
	{ \
		if ( inca == 1 ) \
		{ \
			if ( bli_is_conj( conja ) ) UNPACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, 1, scal2js ); \
			else                        UNPACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, 1, scal2s ); \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) UNPACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca, scal2js ); \
			else                        UNPACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca, scal2s ); \
		} \
	} \
	else /* if ( cdim < mnr ) */ \
	{ \
			if ( bli_is_conj( conja ) ) UNPACKM_BODY( ctype, ch, , cdim, inca, scal2js ); \
			else                        UNPACKM_BODY( ctype, ch, , cdim, inca, scal2s ); \
	} \
}

INSERT_GENTFUNC_BASIC( unpackm_mrxk, BLIS_MR, BLIS_BBM, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNC_BASIC( unpackm_nrxk, BLIS_NR, BLIS_BBN, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

