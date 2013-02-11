/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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

#ifndef BLIS_ADDJS_H
#define BLIS_ADDJS_H

// addjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.

#define bl2_ssaddjs( a, y ) \
{ \
	(y)      += ( float  )(a); \
}
#define bl2_sdaddjs( a, y ) \
{ \
	(y)      += ( double )(a); \
}
#define bl2_scaddjs( a, y ) \
{ \
	(y).real += ( float  )(a); \
	/*(y).imag += 0.0F;*/ \
}
#define bl2_szaddjs( a, y ) \
{ \
	(y).real += ( double )(a); \
	/*(y).imag += 0.0F;*/ \
}

#define bl2_dsaddjs( a, y ) \
{ \
	(y)      += ( float  )(a); \
}
#define bl2_ddaddjs( a, y ) \
{ \
	(y)      += ( double )(a); \
}
#define bl2_dcaddjs( a, y ) \
{ \
	(y).real += ( float  )(a); \
	/*(y).imag += 0.0F;*/ \
}
#define bl2_dzaddjs( a, y ) \
{ \
	(y).real += ( double )(a); \
	/*(y).imag += 0.0F;*/ \
}

#define bl2_csaddjs( a, y ) \
{ \
	(y)      += ( float  )(a).real; \
}
#define bl2_cdaddjs( a, y ) \
{ \
	(y)      += ( double )(a).real; \
}
#define bl2_ccaddjs( a, y ) \
{ \
	(y).real += ( float  )(a).real; \
	(y).imag -= ( float  )(a).imag; \
}
#define bl2_czaddjs( a, y ) \
{ \
	(y).real += ( double )(a).real; \
	(y).imag -= ( double )(a).imag; \
}

#define bl2_zsaddjs( a, y ) \
{ \
	(y)      += ( float  )(a).real; \
}
#define bl2_zdaddjs( a, y ) \
{ \
	(y)      += ( double )(a).real; \
}
#define bl2_zcaddjs( a, y ) \
{ \
	(y).real += ( float  )(a).real; \
	(y).imag -= ( float  )(a).imag; \
}
#define bl2_zzaddjs( a, y ) \
{ \
	(y).real += ( double )(a).real; \
	(y).imag -= ( double )(a).imag; \
}


#define bl2_saddjs( a, y ) \
{ \
	bl2_ssaddjs( a, y ); \
}
#define bl2_daddjs( a, y ) \
{ \
	bl2_ddaddjs( a, y ); \
}
#define bl2_caddjs( a, y ) \
{ \
	bl2_ccaddjs( a, y ); \
}
#define bl2_zaddjs( a, y ) \
{ \
	bl2_zzaddjs( a, y ); \
}


#endif
