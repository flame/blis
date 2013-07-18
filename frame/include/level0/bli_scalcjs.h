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

#ifndef BLIS_SCALCJS_H
#define BLIS_SCALCJS_H

// scalcjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.


#define bli_ssscalcjs( conj, a, x ) \
{ \
	(x)          = bli_ssimulnn_r( (a), (x) ); \
}
#define bli_dsscalcjs( conj, a, x ) \
{ \
	(x)          = bli_dsimulnn_r( (a), (x) ); \
}
#define bli_csscalcjs( conj, a, x ) \
{ \
	(x)          = bli_csimulnn_r( (a), (x) ); \
}
#define bli_zsscalcjs( conj, a, x ) \
{ \
	(x)          = bli_zsimulnn_r( (a), (x) ); \
}


#define bli_sdscalcjs( conj, a, x ) \
{ \
	(x)          = bli_sdimulnn_r( (a), (x) ); \
}
#define bli_ddscalcjs( conj, a, x ) \
{ \
	(x)          = bli_ddimulnn_r( (a), (x) ); \
}
#define bli_cdscalcjs( conj, a, x ) \
{ \
	(x)          = bli_cdimulnn_r( (a), (x) ); \
}
#define bli_zdscalcjs( conj, a, x ) \
{ \
	(x)          = bli_zdimulnn_r( (a), (x) ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scscalcjs( conj, a, x ) \
{ \
	bli_creal(x) = bli_scimulnn_r( (a), (x) ); \
	bli_cimag(x) = bli_scimulnn_i( (a), (x) ); \
}
#define bli_dcscalcjs( conj, a, x ) \
{ \
	bli_creal(x) = bli_dcimulnn_r( (a), (x) ); \
	bli_cimag(x) = bli_dcimulnn_i( (a), (x) ); \
}
#define bli_ccscalcjs( conj, a, x ) \
{ \
	float  tempr, tempi; \
	if ( bli_is_conj( conj ) ) { tempr = bli_ccimulcn_r( (a), (x) );   \
	                             tempi = bli_ccimulcn_i( (a), (x) ); } \
	else                       { tempr = bli_ccimulnn_r( (a), (x) );   \
	                             tempi = bli_ccimulnn_i( (a), (x) ); } \
	bli_creal(x) = tempr; \
	bli_cimag(x) = tempi; \
}
#define bli_zcscalcjs( conj, a, x ) \
{ \
	float  tempr, tempi; \
	if ( bli_is_conj( conj ) ) { tempr = bli_zcimulcn_r( (a), (x) );   \
	                             tempi = bli_zcimulcn_i( (a), (x) ); } \
	else                       { tempr = bli_zcimulnn_r( (a), (x) );   \
	                             tempi = bli_zcimulnn_i( (a), (x) ); } \
	bli_creal(x) = tempr; \
	bli_cimag(x) = tempi; \
}


#define bli_szscalcjs( conj, a, x ) \
{ \
	bli_zreal(x) = bli_szimulnn_r( (a), (x) ); \
	bli_zimag(x) = bli_szimulnn_i( (a), (x) ); \
}
#define bli_dzscalcjs( conj, a, x ) \
{ \
	bli_zreal(x) = bli_dzimulnn_r( (a), (x) ); \
	bli_zimag(x) = bli_dzimulnn_i( (a), (x) ); \
}
#define bli_czscalcjs( conj, a, x ) \
{ \
	double tempr, tempi; \
	if ( bli_is_conj( conj ) ) { tempr = bli_czimulcn_r( (a), (x) );   \
	                             tempi = bli_czimulcn_i( (a), (x) ); } \
	else                       { tempr = bli_czimulnn_r( (a), (x) );   \
	                             tempi = bli_czimulnn_i( (a), (x) ); } \
	bli_zreal(x) = tempr; \
	bli_zimag(x) = tempi; \
}
#define bli_zzscalcjs( conj, a, x ) \
{ \
	double tempr, tempi; \
	if ( bli_is_conj( conj ) ) { tempr = bli_zzimulcn_r( (a), (x) );   \
	                             tempi = bli_zzimulcn_i( (a), (x) ); } \
	else                       { tempr = bli_zzimulnn_r( (a), (x) );   \
	                             tempi = bli_zzimulnn_i( (a), (x) ); } \
	bli_zreal(x) = tempr; \
	bli_zimag(x) = tempi; \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scscalcjs( conj, a, x )  { (x) *= (a); }
#define bli_dcscalcjs( conj, a, x )  { (x) *= (a); }
#define bli_ccscalcjs( conj, a, x )  { (x) *= ( bli_is_conj( conj ) ? conjf(a) : (a) ); }
#define bli_zcscalcjs( conj, a, x )  { (x) *= ( bli_is_conj( conj ) ? conj(a)  : (a) ); }

#define bli_szscalcjs( conj, a, x )  { (x) *= (a); }
#define bli_dzscalcjs( conj, a, x )  { (x) *= (a); }
#define bli_czscalcjs( conj, a, x )  { (x) *= ( bli_is_conj( conj ) ? conjf(a) : (a) ); }
#define bli_zzscalcjs( conj, a, x )  { (x) *= ( bli_is_conj( conj ) ? conj(a)  : (a) ); }


#endif // BLIS_ENABLE_C99_COMPLEX



#define bli_sscalcjs( conj, a, x )  bli_ssscalcjs( conj, a, x )
#define bli_dscalcjs( conj, a, x )  bli_ddscalcjs( conj, a, x )
#define bli_cscalcjs( conj, a, x )  bli_ccscalcjs( conj, a, x )
#define bli_zscalcjs( conj, a, x )  bli_zzscalcjs( conj, a, x )


#endif
