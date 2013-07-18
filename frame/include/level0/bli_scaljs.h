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

#ifndef BLIS_SCALJS_H
#define BLIS_SCALJS_H

// scaljs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.


#define bli_ssscaljs( a, x ) \
{ \
	(x)          = bli_ssimulcn_r( (a), (x) ); \
}
#define bli_dsscaljs( a, x ) \
{ \
	(x)          = bli_dsimulcn_r( (a), (x) ); \
}
#define bli_csscaljs( a, x ) \
{ \
	(x)          = bli_csimulcn_r( (a), (x) ); \
}
#define bli_zsscaljs( a, x ) \
{ \
	(x)          = bli_zsimulcn_r( (a), (x) ); \
}


#define bli_sdscaljs( a, x ) \
{ \
	(x)          = bli_sdimulcn_r( (a), (x) ); \
}
#define bli_ddscaljs( a, x ) \
{ \
	(x)          = bli_ddimulcn_r( (a), (x) ); \
}
#define bli_cdscaljs( a, x ) \
{ \
	(x)          = bli_cdimulcn_r( (a), (x) ); \
}
#define bli_zdscaljs( a, x ) \
{ \
	(x)          = bli_zdimulcn_r( (a), (x) ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scscaljs( a, x ) \
{ \
	bli_creal(x) = bli_scimulcn_r( (a), (x) ); \
	bli_cimag(x) = bli_scimulcn_i( (a), (x) ); \
}
#define bli_dcscaljs( a, x ) \
{ \
	bli_creal(x) = bli_dcimulcn_r( (a), (x) ); \
	bli_cimag(x) = bli_dcimulcn_i( (a), (x) ); \
}
#define bli_ccscaljs( a, x ) \
{ \
	float  tempr = bli_ccimulcn_r( (a), (x) ); \
	float  tempi = bli_ccimulcn_i( (a), (x) ); \
	bli_creal(x) = tempr; \
	bli_cimag(x) = tempi; \
}
#define bli_zcscaljs( a, x ) \
{ \
	float  tempr = bli_zcimulcn_r( (a), (x) ); \
	float  tempi = bli_zcimulcn_i( (a), (x) ); \
	bli_creal(x) = tempr; \
	bli_cimag(x) = tempi; \
}


#define bli_szscaljs( a, x ) \
{ \
	bli_zreal(x) = bli_szimulcn_r( (a), (x) ); \
	bli_zimag(x) = bli_szimulcn_i( (a), (x) ); \
}
#define bli_dzscaljs( a, x ) \
{ \
	bli_zreal(x) = bli_dzimulcn_r( (a), (x) ); \
	bli_zimag(x) = bli_dzimulcn_i( (a), (x) ); \
}
#define bli_czscaljs( a, x ) \
{ \
	double tempr = bli_czimulcn_r( (a), (x) ); \
	double tempi = bli_czimulcn_i( (a), (x) ); \
	bli_zreal(x) = tempr; \
	bli_zimag(x) = tempi; \
}
#define bli_zzscaljs( a, x ) \
{ \
	double tempr = bli_zzimulcn_r( (a), (x) ); \
	double tempi = bli_zzimulcn_i( (a), (x) ); \
	bli_zreal(x) = tempr; \
	bli_zimag(x) = tempi; \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scscaljs( a, x )  { (x) *= (a); }
#define bli_dcscaljs( a, x )  { (x) *= (a); }
#define bli_ccscaljs( a, x )  { (x) *= conjf(a); }
#define bli_zcscaljs( a, x )  { (x) *= conj(a); }

#define bli_szscaljs( a, x )  { (x) *= (a); }
#define bli_dzscaljs( a, x )  { (x) *= (a); }
#define bli_czscaljs( a, x )  { (x) *= conjf(a); }
#define bli_zzscaljs( a, x )  { (x) *= conj(a); }


#endif // BLIS_ENABLE_C99_COMPLEX



#define bli_sscaljs( a, x )  bli_ssscaljs( a, x )
#define bli_dscaljs( a, x )  bli_ddscaljs( a, x )
#define bli_cscaljs( a, x )  bli_ccscaljs( a, x )
#define bli_zscaljs( a, x )  bli_zzscaljs( a, x )


#endif
