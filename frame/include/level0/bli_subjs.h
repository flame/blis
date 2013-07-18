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

#ifndef BLIS_SUBJS_H
#define BLIS_SUBJS_H

// subjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.


#define bli_sssubjs( a, y ) \
{ \
	(y) -= bli_sreal(a); \
}
#define bli_dssubjs( a, y ) \
{ \
	(y) -= bli_dreal(a); \
}
#define bli_cssubjs( a, y ) \
{ \
	(y) -= bli_creal(a); \
}
#define bli_zssubjs( a, y ) \
{ \
	(y) -= bli_zreal(a); \
}


#define bli_sdsubjs( a, y ) \
{ \
	(y) -= bli_sreal(a); \
}
#define bli_ddsubjs( a, y ) \
{ \
	(y) -= bli_dreal(a); \
}
#define bli_cdsubjs( a, y ) \
{ \
	(y) -= bli_creal(a); \
}
#define bli_zdsubjs( a, y ) \
{ \
	(y) -= bli_zreal(a); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scsubjs( a, y ) \
{ \
	bli_creal(y) -= bli_sreal(a); \
}
#define bli_dcsubjs( a, y ) \
{ \
	bli_creal(y) -= bli_dreal(a); \
}
#define bli_ccsubjs( a, y ) \
{ \
	bli_creal(y) -= bli_creal(a); \
	bli_cimag(y) -= -bli_cimag(a); \
}
#define bli_zcsubjs( a, y ) \
{ \
	bli_creal(y) -= bli_zreal(a); \
	bli_cimag(y) -= -bli_zimag(a); \
}


#define bli_szsubjs( a, y ) \
{ \
	bli_zreal(y) -= bli_sreal(a); \
}
#define bli_dzsubjs( a, y ) \
{ \
	bli_zreal(y) -= bli_dreal(a); \
}
#define bli_czsubjs( a, y ) \
{ \
	bli_zreal(y) -= bli_creal(a); \
	bli_zimag(y) -= -bli_cimag(a); \
}
#define bli_zzsubjs( a, y ) \
{ \
	bli_zreal(y) -= bli_zreal(a); \
	bli_zimag(y) -= -bli_zimag(a); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scsubjs( a, y )  { (y) -=      (a); }
#define bli_dcsubjs( a, y )  { (y) -=      (a); }
#define bli_ccsubjs( a, y )  { (y) -= conjf(a); }
#define bli_zcsubjs( a, y )  { (y) -=  conj(a); }

#define bli_szsubjs( a, y )  { (y) -=      (a); }
#define bli_dzsubjs( a, y )  { (y) -=      (a); }
#define bli_czsubjs( a, y )  { (y) -= conjf(a); }
#define bli_zzsubjs( a, y )  { (y) -=  conj(a); }


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_ssubjs( a, y )  bli_sssubjs( a, y )
#define bli_dsubjs( a, y )  bli_ddsubjs( a, y )
#define bli_csubjs( a, y )  bli_ccsubjs( a, y )
#define bli_zsubjs( a, y )  bli_zzsubjs( a, y )


#endif
