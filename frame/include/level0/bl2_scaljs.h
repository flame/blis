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

#ifndef BLIS_SCALJS_H
#define BLIS_SCALJS_H

// scaljs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - a is used in conjugated form.

#define bl2_ssscaljs( a, x ) \
{ \
	(x) *= ( float  ) (a); \
}
#define bl2_dsscaljs( a, x ) \
{ \
	(x) *= ( float  ) (a); \
}
#define bl2_csscaljs( a, x ) \
{ \
	(x) *= ( float  ) (a).real; \
}
#define bl2_zsscaljs( a, x ) \
{ \
	(x) *= ( float  ) (a).real; \
}

#define bl2_sdscaljs( a, x ) \
{ \
	(x) *= ( double ) (a); \
}
#define bl2_ddscaljs( a, x ) \
{ \
	(x) *= ( double ) (a); \
}
#define bl2_cdscaljs( a, x ) \
{ \
	(x) *= ( double ) (a).real; \
}
#define bl2_zdscaljs( a, x ) \
{ \
	(x) *= ( double ) (a).real; \
}

#define bl2_scscaljs( a, x ) \
{ \
	(x).real *= ( float  ) (a); \
	(x).imag *= ( float  ) (a); \
}
#define bl2_dcscaljs( a, x ) \
{ \
	(x).real *= ( float  ) (a); \
	(x).imag *= ( float  ) (a); \
}
#define bl2_ccscaljs( a, x ) \
{ \
	float tempr = ( float  ) (a).real * (x).real + ( float  ) (a).imag * (x).imag; \
	float tempi = ( float  ) (a).real * (x).imag - ( float  ) (a).imag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}
#define bl2_zcscaljs( a, x ) \
{ \
	float tempr = ( float  ) (a).real * (x).real + ( float  ) (a).imag * (x).imag; \
	float tempi = ( float  ) (a).real * (x).imag - ( float  ) (a).imag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}

#define bl2_szscaljs( a, x ) \
{ \
	(x).real *= ( double ) (a); \
	(x).imag *= ( double ) (a); \
}
#define bl2_dzscaljs( a, x ) \
{ \
	(x).real *= ( double ) (a); \
	(x).imag *= ( double ) (a); \
}
#define bl2_czscaljs( a, x ) \
{ \
	double tempr = ( double ) (a).real * (x).real + ( double ) (a).imag * (x).imag; \
	double tempi = ( double ) (a).real * (x).imag - ( double ) (a).imag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}
#define bl2_zzscaljs( a, x ) \
{ \
	double tempr = ( double ) (a).real * (x).real + ( double ) (a).imag * (x).imag; \
	double tempi = ( double ) (a).real * (x).imag - ( double ) (a).imag * (x).real; \
	(x).real = tempr; \
	(x).imag = tempi; \
}


#define bl2_sscaljs( a, x ) \
{ \
	bl2_ssscaljs( a, x ); \
}
#define bl2_dscaljs( a, x ) \
{ \
	bl2_ddscaljs( a, x ); \
}
#define bl2_cscaljs( a, x ) \
{ \
	bl2_ccscaljs( a, x ); \
}
#define bl2_zscaljs( a, x ) \
{ \
	bl2_zzscaljs( a, x ); \
}


#endif
