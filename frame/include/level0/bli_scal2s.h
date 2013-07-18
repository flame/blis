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

#ifndef BLIS_SCAL2S_H
#define BLIS_SCAL2S_H

// scal2s

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.


// -- (axy) = (?ss) ------------------------------------------------------------

#define bli_sssscal2s( a, x, y ) \
{ \
	(y)          = bli_ssimulnn_r( (a), (x) ); \
}
#define bli_dssscal2s( a, x, y ) \
{ \
	(y)          = bli_dsimulnn_r( (a), (x) ); \
}
#define bli_cssscal2s( a, x, y ) \
{ \
	(y)          = bli_csimulnn_r( (a), (x) ); \
}
#define bli_zssscal2s( a, x, y ) \
{ \
	(y)          = bli_zsimulnn_r( (a), (x) ); \
}

// -- (axy) = (?ds) ------------------------------------------------------------

#define bli_sdsscal2s( a, x, y ) \
{ \
	(y)          = bli_sdimulnn_r( (a), (x) ); \
}
#define bli_ddsscal2s( a, x, y ) \
{ \
	(y)          = bli_ddimulnn_r( (a), (x) ); \
}
#define bli_cdsscal2s( a, x, y ) \
{ \
	(y)          = bli_cdimulnn_r( (a), (x) ); \
}
#define bli_zdsscal2s( a, x, y ) \
{ \
	(y)          = bli_zdimulnn_r( (a), (x) ); \
}

// -- (axy) = (?cs) ------------------------------------------------------------

#define bli_scsscal2s( a, x, y ) \
{ \
	(y)          = bli_scimulnn_r( (a), (x) ); \
}
#define bli_dcsscal2s( a, x, y ) \
{ \
	(y)          = bli_dcimulnn_r( (a), (x) ); \
}
#define bli_ccsscal2s( a, x, y ) \
{ \
	(y)          = bli_ccimulnn_r( (a), (x) ); \
}
#define bli_zcsscal2s( a, x, y ) \
{ \
	(y)          = bli_zcimulnn_r( (a), (x) ); \
}

// -- (axy) = (?zs) ------------------------------------------------------------

#define bli_szsscal2s( a, x, y ) \
{ \
	(y)          = bli_szimulnn_r( (a), (x) ); \
}
#define bli_dzsscal2s( a, x, y ) \
{ \
	(y)          = bli_dzimulnn_r( (a), (x) ); \
}
#define bli_czsscal2s( a, x, y ) \
{ \
	(y)          = bli_czimulnn_r( (a), (x) ); \
}
#define bli_zzsscal2s( a, x, y ) \
{ \
	(y)          = bli_zzimulnn_r( (a), (x) ); \
}


// -- (axy) = (?sd) ------------------------------------------------------------

#define bli_ssdscal2s( a, x, y ) \
{ \
	(y)          = bli_ssimulnn_r( (a), (x) ); \
}
#define bli_dsdscal2s( a, x, y ) \
{ \
	(y)          = bli_dsimulnn_r( (a), (x) ); \
}
#define bli_csdscal2s( a, x, y ) \
{ \
	(y)          = bli_csimulnn_r( (a), (x) ); \
}
#define bli_zsdscal2s( a, x, y ) \
{ \
	(y)          = bli_zsimulnn_r( (a), (x) ); \
}

// -- (axy) = (?dd) ------------------------------------------------------------

#define bli_sddscal2s( a, x, y ) \
{ \
	(y)          = bli_sdimulnn_r( (a), (x) ); \
}
#define bli_dddscal2s( a, x, y ) \
{ \
	(y)          = bli_ddimulnn_r( (a), (x) ); \
}
#define bli_cddscal2s( a, x, y ) \
{ \
	(y)          = bli_cdimulnn_r( (a), (x) ); \
}
#define bli_zddscal2s( a, x, y ) \
{ \
	(y)          = bli_zdimulnn_r( (a), (x) ); \
}

// -- (axy) = (?cd) ------------------------------------------------------------

#define bli_scdscal2s( a, x, y ) \
{ \
	(y)          = bli_scimulnn_r( (a), (x) ); \
}
#define bli_dcdscal2s( a, x, y ) \
{ \
	(y)          = bli_dcimulnn_r( (a), (x) ); \
}
#define bli_ccdscal2s( a, x, y ) \
{ \
	(y)          = bli_ccimulnn_r( (a), (x) ); \
}
#define bli_zcdscal2s( a, x, y ) \
{ \
	(y)          = bli_zcimulnn_r( (a), (x) ); \
}

// -- (axy) = (?zd) ------------------------------------------------------------

#define bli_szdscal2s( a, x, y ) \
{ \
	(y)          = bli_szimulnn_r( (a), (x) ); \
}
#define bli_dzdscal2s( a, x, y ) \
{ \
	(y)          = bli_dzimulnn_r( (a), (x) ); \
}
#define bli_czdscal2s( a, x, y ) \
{ \
	(y)          = bli_czimulnn_r( (a), (x) ); \
}
#define bli_zzdscal2s( a, x, y ) \
{ \
	(y)          = bli_zzimulnn_r( (a), (x) ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


// -- (axy) = (?sc) ------------------------------------------------------------

#define bli_sscscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_ssimulnn_r( (a), (x) ); \
}
#define bli_dscscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_dsimulnn_r( (a), (x) ); \
}
#define bli_cscscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_csimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_csimulnn_i( (a), (x) ); \
}
#define bli_zscscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_zsimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_zsimulnn_i( (a), (x) ); \
}

// -- (axy) = (?dc) ------------------------------------------------------------

#define bli_sdcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_sdimulnn_r( (a), (x) ); \
}
#define bli_ddcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_ddimulnn_r( (a), (x) ); \
}
#define bli_cdcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_cdimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_cdimulnn_i( (a), (x) ); \
}
#define bli_zdcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_zdimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_zdimulnn_i( (a), (x) ); \
}

// -- (axy) = (?cc) ------------------------------------------------------------

#define bli_sccscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_scimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_scimulnn_i( (a), (x) ); \
}
#define bli_dccscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_dcimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_dcimulnn_i( (a), (x) ); \
}
#define bli_cccscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_ccimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_ccimulnn_i( (a), (x) ); \
}
#define bli_zccscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_zcimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_zcimulnn_i( (a), (x) ); \
}

// -- (axy) = (?zc) ------------------------------------------------------------

#define bli_szcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_szimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_szimulnn_i( (a), (x) ); \
}
#define bli_dzcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_dzimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_dzimulnn_i( (a), (x) ); \
}
#define bli_czcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_czimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_czimulnn_i( (a), (x) ); \
}
#define bli_zzcscal2s( a, x, y ) \
{ \
	bli_creal(y) = bli_zzimulnn_r( (a), (x) ); \
	bli_cimag(y) = bli_zzimulnn_i( (a), (x) ); \
}


// -- (axy) = (?sz) ------------------------------------------------------------

#define bli_sszscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_ssimulnn_r( (a), (x) ); \
}
#define bli_dszscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_dsimulnn_r( (a), (x) ); \
}
#define bli_cszscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_csimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_csimulnn_i( (a), (x) ); \
}
#define bli_zszscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_zsimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_zsimulnn_i( (a), (x) ); \
}

// -- (axy) = (?dz) ------------------------------------------------------------

#define bli_sdzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_sdimulnn_r( (a), (x) ); \
}
#define bli_ddzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_ddimulnn_r( (a), (x) ); \
}
#define bli_cdzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_cdimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_cdimulnn_i( (a), (x) ); \
}
#define bli_zdzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_zdimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_zdimulnn_i( (a), (x) ); \
}

// -- (axy) = (?cz) ------------------------------------------------------------

#define bli_sczscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_scimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_scimulnn_i( (a), (x) ); \
}
#define bli_dczscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_dcimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_dcimulnn_i( (a), (x) ); \
}
#define bli_cczscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_ccimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_ccimulnn_i( (a), (x) ); \
}
#define bli_zczscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_zcimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_zcimulnn_i( (a), (x) ); \
}

// -- (axy) = (?zz) ------------------------------------------------------------

#define bli_szzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_szimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_szimulnn_i( (a), (x) ); \
}
#define bli_dzzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_dzimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_dzimulnn_i( (a), (x) ); \
}
#define bli_czzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_czimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_czimulnn_i( (a), (x) ); \
}
#define bli_zzzscal2s( a, x, y ) \
{ \
	bli_zreal(y) = bli_zzimulnn_r( (a), (x) ); \
	bli_zimag(y) = bli_zzimulnn_i( (a), (x) ); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_sscscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_dscscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_cscscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zscscal2s( a, x, y )  { (y) = (a) * (x); }

#define bli_sdcscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_ddcscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_cdcscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zdcscal2s( a, x, y )  { (y) = (a) * (x); }

#define bli_sccscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_dccscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_cccscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zccscal2s( a, x, y )  { (y) = (a) * (x); }

#define bli_szcscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_dzcscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_czcscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zzcscal2s( a, x, y )  { (y) = (a) * (x); }


#define bli_sszscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_dszscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_cszscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zszscal2s( a, x, y )  { (y) = (a) * (x); }

#define bli_sdzscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_ddzscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_cdzscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zdzscal2s( a, x, y )  { (y) = (a) * (x); }

#define bli_sczscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_dczscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_cczscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zczscal2s( a, x, y )  { (y) = (a) * (x); }

#define bli_szzscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_dzzscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_czzscal2s( a, x, y )  { (y) = (a) * (x); }
#define bli_zzzscal2s( a, x, y )  { (y) = (a) * (x); }


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_sscal2s( a, x, y )  bli_sssscal2s( a, x, y )
#define bli_dscal2s( a, x, y )  bli_dddscal2s( a, x, y )
#define bli_cscal2s( a, x, y )  bli_cccscal2s( a, x, y )
#define bli_zscal2s( a, x, y )  bli_zzzscal2s( a, x, y )


#endif
