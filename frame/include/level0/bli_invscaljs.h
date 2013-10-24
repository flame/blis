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
	float  s     = ( float  )bli_fmax( bli_creal(a), bli_cimag(a) ); \
	float  ar_s  = ( float  )bli_creal(a) / s; \
	float  ai_s  = ( float  )bli_cimag(a) / s; \
	float  temp  = ( ar_s * ( float  )bli_creal(a) + ai_s * ( float  )bli_cimag(a) ); \
	bli_creal(x) = ( bli_creal(x) * ar_s - bli_cimag(x) * ai_s ) / temp; \
	bli_cimag(x) = ( bli_cimag(x) * ar_s + bli_creal(x) * ai_s ) / temp; \
}
#define bli_zcinvscaljs( a, x ) \
{ \
	double s     = ( double )bli_fmax( bli_zreal(a), bli_zimag(a) ); \
	double ar_s  = ( double )bli_zreal(a) / s; \
	double ai_s  = ( double )bli_zimag(a) / s; \
	double temp  = ( ar_s * ( double )bli_zreal(a) + ai_s * ( double )bli_zimag(a) ); \
	bli_creal(x) = ( bli_creal(x) * ar_s - bli_cimag(x) * ai_s ) / temp; \
	bli_cimag(x) = ( bli_cimag(x) * ar_s + bli_creal(x) * ai_s ) / temp; \
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
	double s     = ( double )bli_fmax( bli_creal(a), bli_cimag(a) ); \
	double ar_s  = ( double )bli_creal(a) / s; \
	double ai_s  = ( double )bli_cimag(a) / s; \
	double temp  = ( ar_s * ( double )bli_creal(a) + ai_s * ( double )bli_cimag(a) ); \
	bli_zreal(x) = ( bli_zreal(x) * ar_s - bli_zimag(x) * ai_s ) / temp; \
	bli_zimag(x) = ( bli_zimag(x) * ar_s + bli_zreal(x) * ai_s ) / temp; \
}
#define bli_zzinvscaljs( a, x ) \
{ \
	double s     = ( double )bli_fmax( bli_zreal(a), bli_zimag(a) ); \
	double ar_s  = ( double )bli_zreal(a) / s; \
	double ai_s  = ( double )bli_zimag(a) / s; \
	double temp  = ( ar_s * ( double )bli_zreal(a) + ai_s * ( double )bli_zimag(a) ); \
	bli_zreal(x) = ( bli_zreal(x) * ar_s - bli_zimag(x) * ai_s ) / temp; \
	bli_zimag(x) = ( bli_zimag(x) * ar_s + bli_zreal(x) * ai_s ) / temp; \
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
