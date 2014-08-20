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

#ifndef BLIS_INVSCALCJS_H
#define BLIS_INVSCALCJS_H

// invscalcjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.

#define bli_ssinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_dsinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_csinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}
#define bli_zsinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}

#define bli_sdinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_ddinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_cdinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}
#define bli_zdinvscalcjs( conj, a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}

#define bli_scinvscalcjs( conj, a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bli_dcinvscalcjs( conj, a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bli_ccinvscalcjs( conj, a, x ) \
{ \
	float  aimag = ( bli_is_conj( conj ) ? ( float  ) -(a).imag : \
	                                       ( float  )  (a).imag ); \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) aimag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real + ( float  ) aimag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag - ( float  ) aimag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bli_zcinvscalcjs( conj, a, x ) \
{ \
	float  aimag = ( bli_is_conj( conj ) ? ( float  ) -(a).imag : \
	                                       ( float  )  (a).imag ); \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) aimag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real + ( float  ) aimag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag - ( float  ) aimag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}

#define bli_szinvscalcjs( conj, a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bli_dzinvscalcjs( conj, a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bli_czinvscalcjs( conj, a, x ) \
{ \
	double aimag = ( bli_is_conj( conj ) ? ( double ) -(a).imag : \
	                                       ( double )  (a).imag ); \
	double temp =              ( double ) (a).real * (a).real + ( double ) aimag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real + ( double ) aimag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag - ( double ) aimag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bli_zzinvscalcjs( conj, a, x ) \
{ \
	double aimag = ( bli_is_conj( conj ) ? ( double ) -(a).imag : \
	                                       ( double )  (a).imag ); \
	double temp =              ( double ) (a).real * (a).real + ( double ) aimag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real + ( double ) aimag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag - ( double ) aimag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}


#define bli_sinvscalcjs( conj, a, x ) \
{ \
	bli_ssinvscalcjs( conj, a, x ); \
}
#define bli_dinvscalcjs( conj, a, x ) \
{ \
	bli_ddinvscalcjs( conj, a, x ); \
}
#define bli_cinvscalcjs( conj, a, x ) \
{ \
	bli_ccinvscalcjs( conj, a, x ); \
}
#define bli_zinvscalcjs( conj, a, x ) \
{ \
	bli_zzinvscalcjs( conj, a, x ); \
}


#endif
