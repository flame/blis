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

#ifndef BLIS_INVSCALJS_H
#define BLIS_INVSCALJS_H

// invscaljs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.


#define bli_ssinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_dsinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_csinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) bli_creal(a); \
}
#define bli_zsinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) bli_zreal(a); \
}


#define bli_sdinvscaljs( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_ddinvscaljs( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_cdinvscaljs( a, x ) \
{ \
	(x) /= ( double ) bli_creal(a); \
}
#define bli_zdinvscaljs( a, x ) \
{ \
	(x) /= ( double ) bli_zreal(a); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scinvscaljs( a, x ) \
{ \
	bli_creal(x) /= ( float  ) (a); \
	bli_cimag(x) /= ( float  ) (a); \
}
#define bli_dcinvscaljs( a, x ) \
{ \
	bli_creal(x) /= ( float  ) (a); \
	bli_cimag(x) /= ( float  ) (a); \
}
#define bli_ccinvscaljs( a, x ) \
{ \
	float  temp  = ( float  ) bli_ccimulnc_r( (a), (a) ); \
	float  xr    = ( float  ) bli_ccimulnn_r( (a), (x) ) / temp; \
	float  xi    = ( float  ) bli_ccimulnn_i( (x), (a) ) / temp; \
	bli_creal(x) = xr; \
	bli_cimag(x) = xi; \
}
#define bli_zcinvscaljs( a, x ) \
{ \
	float  temp  = ( float  ) bli_zzimulnc_r( (a), (a) ); \
	float  xr    = ( float  ) bli_zcimulnn_r( (a), (x) ) / temp; \
	float  xi    = ( float  ) bli_czimulnn_i( (x), (a) ) / temp; \
	bli_creal(x) = xr; \
	bli_cimag(x) = xi; \
}


#define bli_szinvscaljs( a, x ) \
{ \
	bli_zreal(x) /= ( double ) (a); \
	bli_zimag(x) /= ( double ) (a); \
}
#define bli_dzinvscaljs( a, x ) \
{ \
	bli_zreal(x) /= ( double ) (a); \
	bli_zimag(x) /= ( double ) (a); \
}
#define bli_czinvscaljs( a, x ) \
{ \
	double temp  = ( double ) bli_ccimulnc_r( (a), (a) ); \
	double xr    = ( double ) bli_czimulnn_r( (a), (x) ) / temp; \
	double xi    = ( double ) bli_zcimulnn_i( (x), (a) ) / temp; \
	bli_zreal(x) = xr; \
	bli_zimag(x) = xi; \
}
#define bli_zzinvscaljs( a, x ) \
{ \
	double temp  = ( double ) bli_zzimulnc_r( (a), (a) ); \
	double xr    = ( double ) bli_zzimulnn_r( (a), (x) ) / temp; \
	double xi    = ( double ) bli_zzimulnn_i( (x), (a) ) / temp; \
	bli_zreal(x) = xr; \
	bli_zimag(x) = xi; \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scinvscaljs( a, x )  { (x) /= (a); }
#define bli_dcinvscaljs( a, x )  { (x) /= (a); }
#define bli_ccinvscaljs( a, x )  { (x) /= conjf(a); }
#define bli_zcinvscaljs( a, x )  { (x) /= conj(a); }

#define bli_szinvscaljs( a, x )  { (x) /= (a); }
#define bli_dzinvscaljs( a, x )  { (x) /= (a); }
#define bli_czinvscaljs( a, x )  { (x) /= conjf(a); }
#define bli_zzinvscaljs( a, x )  { (x) /= conj(a); }


#endif // BLIS_ENABLE_C99_COMPLEX



#define bli_sinvscaljs( a, x )  bli_ssinvscaljs( a, x )
#define bli_dinvscaljs( a, x )  bli_ddinvscaljs( a, x )
#define bli_cinvscaljs( a, x )  bli_ccinvscaljs( a, x )
#define bli_zinvscaljs( a, x )  bli_zzinvscaljs( a, x )


#endif
