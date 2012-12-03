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

#ifndef BLIS_SCALCJS_H
#define BLIS_SCALCJS_H

// scalcjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - a is (conditionally) used in conjugated form.

#define bl2_ssscalcjs( conj, a, x ) \
{ \
	(x) *= ( float  ) (a); \
}
#define bl2_dsscalcjs( conj, a, x ) \
{ \
	(x) *= ( float  ) (a); \
}
#define bl2_csscalcjs( conj, a, x ) \
{ \
	(x) *= ( float  ) (a).real; \
}
#define bl2_zsscalcjs( conj, a, x ) \
{ \
	(x) *= ( float  ) (a).real; \
}

#define bl2_sdscalcjs( conj, a, x ) \
{ \
	(x) *= ( double ) (a); \
}
#define bl2_ddscalcjs( conj, a, x ) \
{ \
	(x) *= ( double ) (a); \
}
#define bl2_cdscalcjs( conj, a, x ) \
{ \
	(x) *= ( double ) (a).real; \
}
#define bl2_zdscalcjs( conj, a, x ) \
{ \
	(x) *= ( double ) (a).real; \
}

#define bl2_scscalcjs( conj, a, x ) \
{ \
	(x).real *= ( float  ) (a); \
	(x).imag *= ( float  ) (a); \
}
#define bl2_dcscalcjs( conj, a, x ) \
{ \
	(x).real *= ( float  ) (a); \
	(x).imag *= ( float  ) (a); \
}
#define bl2_ccscalcjs( conj, a, x ) \
{ \
	float  aimag = ( bl2_is_conj( conj ) ? ( float  ) -(a).imag : \
	                                       ( float  )  (a).imag ); \
	float  tempr = ( float  ) (a).real * (x).real - ( float  ) aimag * (x).imag; \
	float  tempi = ( float  ) (a).real * (x).imag + ( float  ) aimag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}
#define bl2_zcscalcjs( conj, a, x ) \
{ \
	float  aimag = ( bl2_is_conj( conj ) ? ( float  ) -(a).imag : \
	                                       ( float  )  (a).imag ); \
	float  tempr = ( float  ) (a).real * (x).real - ( float  ) aimag * (x).imag; \
	float  tempi = ( float  ) (a).real * (x).imag + ( float  ) aimag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}

#define bl2_szscalcjs( conj, a, x ) \
{ \
	(x).real *= ( double ) (a); \
	(x).imag *= ( double ) (a); \
}
#define bl2_dzscalcjs( conj, a, x ) \
{ \
	(x).real *= ( double ) (a); \
	(x).imag *= ( double ) (a); \
}
#define bl2_czscalcjs( conj, a, x ) \
{ \
	double aimag = ( bl2_is_conj( conj ) ? ( double ) -(a).imag : \
	                                       ( double )  (a).imag ); \
	double tempr = ( double ) (a).real * (x).real - ( double ) aimag * (x).imag; \
	double tempi = ( double ) (a).real * (x).imag + ( double ) aimag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}
#define bl2_zzscalcjs( conj, a, x ) \
{ \
	double aimag = ( bl2_is_conj( conj ) ? ( double ) -(a).imag : \
	                                       ( double )  (a).imag ); \
	double tempr = ( double ) (a).real * (x).real - ( double ) aimag * (x).imag; \
	double tempi = ( double ) (a).real * (x).imag + ( double ) aimag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}


#define bl2_sscalcjs( conj, a, x ) \
{ \
	bl2_ssscalcjs( conj, a, x ); \
}
#define bl2_dscalcjs( conj, a, x ) \
{ \
	bl2_ddscalcjs( conj, a, x ); \
}
#define bl2_cscalcjs( conj, a, x ) \
{ \
	bl2_ccscalcjs( conj, a, x ); \
}
#define bl2_zscalcjs( conj, a, x ) \
{ \
	bl2_zzscalcjs( conj, a, x ); \
}


#endif
