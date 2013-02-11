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

#ifndef BLIS_SUBJS_H
#define BLIS_SUBJS_H

// subjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.

#define bl2_sssubjs( a, y ) \
{ \
	(y)      -= ( float  )(a); \
}
#define bl2_sdsubjs( a, y ) \
{ \
	(y)      -= ( double )(a); \
}
#define bl2_scsubjs( a, y ) \
{ \
	(y).real -= ( float  )(a); \
	/*(y).imag -= 0.0F;*/ \
}
#define bl2_szsubjs( a, y ) \
{ \
	(y).real -= ( double )(a); \
	/*(y).imag -= 0.0F;*/ \
}

#define bl2_dssubjs( a, y ) \
{ \
	(y)      -= ( float  )(a); \
}
#define bl2_ddsubjs( a, y ) \
{ \
	(y)      -= ( double )(a); \
}
#define bl2_dcsubjs( a, y ) \
{ \
	(y).real -= ( float  )(a); \
	/*(y).imag -= 0.0F;*/ \
}
#define bl2_dzsubjs( a, y ) \
{ \
	(y).real -= ( double )(a); \
	/*(y).imag -= 0.0F;*/ \
}

#define bl2_cssubjs( a, y ) \
{ \
	(y)      -= ( float  )(a).real; \
}
#define bl2_cdsubjs( a, y ) \
{ \
	(y)      -= ( double )(a).real; \
}
#define bl2_ccsubjs( a, y ) \
{ \
	(y).real -= ( float  )(a).real; \
	(y).imag += ( float  )(a).imag; \
}
#define bl2_czsubjs( a, y ) \
{ \
	(y).real -= ( double )(a).real; \
	(y).imag += ( double )(a).imag; \
}

#define bl2_zssubjs( a, y ) \
{ \
	(y)      -= ( float  )(a).real; \
}
#define bl2_zdsubjs( a, y ) \
{ \
	(y)      -= ( double )(a).real; \
}
#define bl2_zcsubjs( a, y ) \
{ \
	(y).real -= ( float  )(a).real; \
	(y).imag += ( float  )(a).imag; \
}
#define bl2_zzsubjs( a, y ) \
{ \
	(y).real -= ( double )(a).real; \
	(y).imag += ( double )(a).imag; \
}


#define bl2_ssubjs( a, y ) \
{ \
	bl2_sssubjs( a, y ); \
}
#define bl2_dsubjs( a, y ) \
{ \
	bl2_ddsubjs( a, y ); \
}
#define bl2_csubjs( a, y ) \
{ \
	bl2_ccsubjs( a, y ); \
}
#define bl2_zsubjs( a, y ) \
{ \
	bl2_zzsubjs( a, y ); \
}


#endif
