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

#ifndef BLIS_SCALS_H
#define BLIS_SCALS_H

// scals

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.


#define bli_ssscals( a, x ) \
{ \
	(x)          = bli_ssimulnn_r( (a), (x) ); \
}
#define bli_dsscals( a, x ) \
{ \
	(x)          = bli_dsimulnn_r( (a), (x) ); \
}
#define bli_csscals( a, x ) \
{ \
	(x)          = bli_csimulnn_r( (a), (x) ); \
}
#define bli_zsscals( a, x ) \
{ \
	(x)          = bli_zsimulnn_r( (a), (x) ); \
}


#define bli_sdscals( a, x ) \
{ \
	(x)          = bli_sdimulnn_r( (a), (x) ); \
}
#define bli_ddscals( a, x ) \
{ \
	(x)          = bli_ddimulnn_r( (a), (x) ); \
}
#define bli_cdscals( a, x ) \
{ \
	(x)          = bli_cdimulnn_r( (a), (x) ); \
}
#define bli_zdscals( a, x ) \
{ \
	(x)          = bli_zdimulnn_r( (a), (x) ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scscals( a, x ) \
{ \
	bli_creal(x) = bli_scimulnn_r( (a), (x) ); \
	bli_cimag(x) = bli_scimulnn_i( (a), (x) ); \
}
#define bli_dcscals( a, x ) \
{ \
	bli_creal(x) = bli_dcimulnn_r( (a), (x) ); \
	bli_cimag(x) = bli_dcimulnn_i( (a), (x) ); \
}
#define bli_ccscals( a, x ) \
{ \
	float  tempr = bli_ccimulnn_r( (a), (x) ); \
	float  tempi = bli_ccimulnn_i( (a), (x) ); \
	bli_creal(x) = tempr; \
	bli_cimag(x) = tempi; \
}
#define bli_zcscals( a, x ) \
{ \
	float  tempr = bli_zcimulnn_r( (a), (x) ); \
	float  tempi = bli_zcimulnn_i( (a), (x) ); \
	bli_creal(x) = tempr; \
	bli_cimag(x) = tempi; \
}


#define bli_szscals( a, x ) \
{ \
	bli_zreal(x) = bli_szimulnn_r( (a), (x) ); \
	bli_zimag(x) = bli_szimulnn_i( (a), (x) ); \
}
#define bli_dzscals( a, x ) \
{ \
	bli_zreal(x) = bli_dzimulnn_r( (a), (x) ); \
	bli_zimag(x) = bli_dzimulnn_i( (a), (x) ); \
}
#define bli_czscals( a, x ) \
{ \
	double tempr = bli_czimulnn_r( (a), (x) ); \
	double tempi = bli_czimulnn_i( (a), (x) ); \
	bli_zreal(x) = tempr; \
	bli_zimag(x) = tempi; \
}
#define bli_zzscals( a, x ) \
{ \
	double tempr = bli_zzimulnn_r( (a), (x) ); \
	double tempi = bli_zzimulnn_i( (a), (x) ); \
	bli_zreal(x) = tempr; \
	bli_zimag(x) = tempi; \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scscals( a, x )  { (x) *= (a); }
#define bli_dcscals( a, x )  { (x) *= (a); }
#define bli_ccscals( a, x )  { (x) *= (a); }
#define bli_zcscals( a, x )  { (x) *= (a); }

#define bli_szscals( a, x )  { (x) *= (a); }
#define bli_dzscals( a, x )  { (x) *= (a); }
#define bli_czscals( a, x )  { (x) *= (a); }
#define bli_zzscals( a, x )  { (x) *= (a); }


#endif // BLIS_ENABLE_C99_COMPLEX



#define bli_sscals( a, x )  bli_ssscals( a, x )
#define bli_dscals( a, x )  bli_ddscals( a, x )
#define bli_cscals( a, x )  bli_ccscals( a, x )
#define bli_zscals( a, x )  bli_zzscals( a, x )


#endif
