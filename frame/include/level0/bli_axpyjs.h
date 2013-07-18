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

#ifndef BLIS_AXPYJS_H
#define BLIS_AXPYJS_H

// axpyjs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.


// -- (axy) = (?ss) ------------------------------------------------------------

#define bli_sssaxpyjs( a, x, y ) \
{ \
	(y)          += bli_ssimulnc_r( (a), (x) ); \
}
#define bli_dssaxpyjs( a, x, y ) \
{ \
	(y)          += bli_dsimulnc_r( (a), (x) ); \
}
#define bli_cssaxpyjs( a, x, y ) \
{ \
	(y)          += bli_csimulnc_r( (a), (x) ); \
}
#define bli_zssaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zsimulnc_r( (a), (x) ); \
}

// -- (axy) = (?ds) ------------------------------------------------------------

#define bli_sdsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_sdimulnc_r( (a), (x) ); \
}
#define bli_ddsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_ddimulnc_r( (a), (x) ); \
}
#define bli_cdsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_cdimulnc_r( (a), (x) ); \
}
#define bli_zdsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zdimulnc_r( (a), (x) ); \
}

// -- (axy) = (?cs) ------------------------------------------------------------

#define bli_scsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_scimulnc_r( (a), (x) ); \
}
#define bli_dcsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_dcimulnc_r( (a), (x) ); \
}
#define bli_ccsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_ccimulnc_r( (a), (x) ); \
}
#define bli_zcsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zcimulnc_r( (a), (x) ); \
}

// -- (axy) = (?zs) ------------------------------------------------------------

#define bli_szsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_szimulnc_r( (a), (x) ); \
}
#define bli_dzsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_dzimulnc_r( (a), (x) ); \
}
#define bli_czsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_czimulnc_r( (a), (x) ); \
}
#define bli_zzsaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zzimulnc_r( (a), (x) ); \
}


// -- (axy) = (?sd) ------------------------------------------------------------

#define bli_ssdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_ssimulnc_r( (a), (x) ); \
}
#define bli_dsdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_dsimulnc_r( (a), (x) ); \
}
#define bli_csdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_csimulnc_r( (a), (x) ); \
}
#define bli_zsdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zsimulnc_r( (a), (x) ); \
}

// -- (axy) = (?dd) ------------------------------------------------------------

#define bli_sddaxpyjs( a, x, y ) \
{ \
	(y)          += bli_sdimulnc_r( (a), (x) ); \
}
#define bli_dddaxpyjs( a, x, y ) \
{ \
	(y)          += bli_ddimulnc_r( (a), (x) ); \
}
#define bli_cddaxpyjs( a, x, y ) \
{ \
	(y)          += bli_cdimulnc_r( (a), (x) ); \
}
#define bli_zddaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zdimulnc_r( (a), (x) ); \
}

// -- (axy) = (?cd) ------------------------------------------------------------

#define bli_scdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_scimulnc_r( (a), (x) ); \
}
#define bli_dcdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_dcimulnc_r( (a), (x) ); \
}
#define bli_ccdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_ccimulnc_r( (a), (x) ); \
}
#define bli_zcdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zcimulnc_r( (a), (x) ); \
}

// -- (axy) = (?zd) ------------------------------------------------------------

#define bli_szdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_szimulnc_r( (a), (x) ); \
}
#define bli_dzdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_dzimulnc_r( (a), (x) ); \
}
#define bli_czdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_czimulnc_r( (a), (x) ); \
}
#define bli_zzdaxpyjs( a, x, y ) \
{ \
	(y)          += bli_zzimulnc_r( (a), (x) ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


// -- (axy) = (?sc) ------------------------------------------------------------

#define bli_sscaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_ssimulnc_r( (a), (x) ); \
}
#define bli_dscaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_dsimulnc_r( (a), (x) ); \
}
#define bli_cscaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_csimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_csimulnc_i( (a), (x) ); \
}
#define bli_zscaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_zsimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_zsimulnc_i( (a), (x) ); \
}

// -- (axy) = (?dc) ------------------------------------------------------------

#define bli_sdcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_sdimulnc_r( (a), (x) ); \
}
#define bli_ddcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_ddimulnc_r( (a), (x) ); \
}
#define bli_cdcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_cdimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_cdimulnc_i( (a), (x) ); \
}
#define bli_zdcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_zdimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_zdimulnc_i( (a), (x) ); \
}

// -- (axy) = (?cc) ------------------------------------------------------------

#define bli_sccaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_scimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_scimulnc_i( (a), (x) ); \
}
#define bli_dccaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_dcimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_dcimulnc_i( (a), (x) ); \
}
#define bli_cccaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_ccimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_ccimulnc_i( (a), (x) ); \
}
#define bli_zccaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_zcimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_zcimulnc_i( (a), (x) ); \
}

// -- (axy) = (?zc) ------------------------------------------------------------

#define bli_szcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_szimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_szimulnc_i( (a), (x) ); \
}
#define bli_dzcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_dzimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_dzimulnc_i( (a), (x) ); \
}
#define bli_czcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_czimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_czimulnc_i( (a), (x) ); \
}
#define bli_zzcaxpyjs( a, x, y ) \
{ \
	bli_creal(y) += bli_zzimulnc_r( (a), (x) ); \
	bli_cimag(y) += bli_zzimulnc_i( (a), (x) ); \
}


// -- (axy) = (?sz) ------------------------------------------------------------

#define bli_sszaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_ssimulnc_r( (a), (x) ); \
}
#define bli_dszaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_dsimulnc_r( (a), (x) ); \
}
#define bli_cszaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_csimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_csimulnc_i( (a), (x) ); \
}
#define bli_zszaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_zsimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_zsimulnc_i( (a), (x) ); \
}

// -- (axy) = (?dz) ------------------------------------------------------------

#define bli_sdzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_sdimulnc_r( (a), (x) ); \
}
#define bli_ddzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_ddimulnc_r( (a), (x) ); \
}
#define bli_cdzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_cdimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_cdimulnc_i( (a), (x) ); \
}
#define bli_zdzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_zdimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_zdimulnc_i( (a), (x) ); \
}

// -- (axy) = (?cz) ------------------------------------------------------------

#define bli_sczaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_scimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_scimulnc_i( (a), (x) ); \
}
#define bli_dczaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_dcimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_dcimulnc_i( (a), (x) ); \
}
#define bli_cczaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_ccimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_ccimulnc_i( (a), (x) ); \
}
#define bli_zczaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_zcimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_zcimulnc_i( (a), (x) ); \
}

// -- (axy) = (?zz) ------------------------------------------------------------

#define bli_szzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_szimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_szimulnc_i( (a), (x) ); \
}
#define bli_dzzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_dzimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_dzimulnc_i( (a), (x) ); \
}
#define bli_czzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_czimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_czimulnc_i( (a), (x) ); \
}
#define bli_zzzaxpyjs( a, x, y ) \
{ \
	bli_zreal(y) += bli_zzimulnc_r( (a), (x) ); \
	bli_zimag(y) += bli_zzimulnc_i( (a), (x) ); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_sscaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_dscaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_cscaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_zscaxpyjs( a, x, y )  { (y) += (a) * (x); }

#define bli_sdcaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_ddcaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_cdcaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_zdcaxpyjs( a, x, y )  { (y) += (a) * (x); }

#define bli_sccaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }
#define bli_dccaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }
#define bli_cccaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }
#define bli_zccaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }

#define bli_szcaxpyjs( a, x, y )  { (y) += (a) * conj(x); }
#define bli_dzcaxpyjs( a, x, y )  { (y) += (a) * conj(x); }
#define bli_czcaxpyjs( a, x, y )  { (y) += (a) * conj(x); }
#define bli_zzcaxpyjs( a, x, y )  { (y) += (a) * conj(x); }


#define bli_sszaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_dszaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_cszaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_zszaxpyjs( a, x, y )  { (y) += (a) * (x); }

#define bli_sdzaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_ddzaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_cdzaxpyjs( a, x, y )  { (y) += (a) * (x); }
#define bli_zdzaxpyjs( a, x, y )  { (y) += (a) * (x); }

#define bli_sczaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }
#define bli_dczaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }
#define bli_cczaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }
#define bli_zczaxpyjs( a, x, y )  { (y) += (a) * conjf(x); }

#define bli_szzaxpyjs( a, x, y )  { (y) += (a) * conj(x); }
#define bli_dzzaxpyjs( a, x, y )  { (y) += (a) * conj(x); }
#define bli_czzaxpyjs( a, x, y )  { (y) += (a) * conj(x); }
#define bli_zzzaxpyjs( a, x, y )  { (y) += (a) * conj(x); }


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_saxpyjs( a, x, y )  bli_sssaxpyjs( a, x, y )
#define bli_daxpyjs( a, x, y )  bli_dddaxpyjs( a, x, y )
#define bli_caxpyjs( a, x, y )  bli_cccaxpyjs( a, x, y )
#define bli_zaxpyjs( a, x, y )  bli_zzzaxpyjs( a, x, y )


#endif
