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

#ifndef BLIS_XPBYS_H
#define BLIS_XPBYS_H

// xpbys

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of b.
// - The third char encodes the type of y.


// -- (xby) = (?ss) ------------------------------------------------------------

#define bli_sssxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_sreal(x) + ( float  ) bli_ssimulnn_r( (b), (y) ); \
}
#define bli_dssxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_dreal(x) + ( float  ) bli_ssimulnn_r( (b), (y) ); \
}
#define bli_cssxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_creal(x) + ( float  ) bli_ssimulnn_r( (b), (y) ); \
}
#define bli_zssxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_zreal(x) + ( float  ) bli_ssimulnn_r( (b), (y) ); \
}

// -- (xby) = (?ds) ------------------------------------------------------------

#define bli_sdsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_sreal(x) + ( float  ) bli_dsimulnn_r( (b), (y) ); \
}
#define bli_ddsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_dreal(x) + ( float  ) bli_dsimulnn_r( (b), (y) ); \
}
#define bli_cdsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_creal(x) + ( float  ) bli_dsimulnn_r( (b), (y) ); \
}
#define bli_zdsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_zreal(x) + ( float  ) bli_dsimulnn_r( (b), (y) ); \
}

// -- (xby) = (?cs) ------------------------------------------------------------

#define bli_scsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_sreal(x) + ( float  ) bli_csimulnn_r( (b), (y) ); \
}
#define bli_dcsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_dreal(x) + ( float  ) bli_csimulnn_r( (b), (y) ); \
}
#define bli_ccsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_creal(x) + ( float  ) bli_csimulnn_r( (b), (y) ); \
}
#define bli_zcsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_zreal(x) + ( float  ) bli_csimulnn_r( (b), (y) ); \
}

// -- (xby) = (?zs) ------------------------------------------------------------

#define bli_szsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_sreal(x) + ( float  ) bli_zsimulnn_r( (b), (y) ); \
}
#define bli_dzsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_dreal(x) + ( float  ) bli_zsimulnn_r( (b), (y) ); \
}
#define bli_czsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_creal(x) + ( float  ) bli_zsimulnn_r( (b), (y) ); \
}
#define bli_zzsxpbys( x, b, y ) \
{ \
	(y)          = ( float  ) bli_zreal(x) + ( float  ) bli_zsimulnn_r( (b), (y) ); \
}

// -- (xby) = (?sd) ------------------------------------------------------------

#define bli_ssdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_sreal(x) + ( double ) bli_sdimulnn_r( (b), (y) ); \
}
#define bli_dsdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_dreal(x) + ( double ) bli_sdimulnn_r( (b), (y) ); \
}
#define bli_csdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_creal(x) + ( double ) bli_sdimulnn_r( (b), (y) ); \
}
#define bli_zsdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_zreal(x) + ( double ) bli_sdimulnn_r( (b), (y) ); \
}

// -- (xby) = (?dd) ------------------------------------------------------------

#define bli_sddxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_sreal(x) + ( double ) bli_ddimulnn_r( (b), (y) ); \
}
#define bli_dddxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_dreal(x) + ( double ) bli_ddimulnn_r( (b), (y) ); \
}
#define bli_cddxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_creal(x) + ( double ) bli_ddimulnn_r( (b), (y) ); \
}
#define bli_zddxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_zreal(x) + ( double ) bli_ddimulnn_r( (b), (y) ); \
}

// -- (xby) = (?cd) ------------------------------------------------------------

#define bli_scdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_sreal(x) + ( double ) bli_cdimulnn_r( (b), (y) ); \
}
#define bli_dcdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_dreal(x) + ( double ) bli_cdimulnn_r( (b), (y) ); \
}
#define bli_ccdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_creal(x) + ( double ) bli_cdimulnn_r( (b), (y) ); \
}
#define bli_zcdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_zreal(x) + ( double ) bli_cdimulnn_r( (b), (y) ); \
}

// -- (xby) = (?zd) ------------------------------------------------------------

#define bli_szdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_sreal(x) + ( double ) bli_zdimulnn_r( (b), (y) ); \
}
#define bli_dzdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_dreal(x) + ( double ) bli_zdimulnn_r( (b), (y) ); \
}
#define bli_czdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_creal(x) + ( double ) bli_zdimulnn_r( (b), (y) ); \
}
#define bli_zzdxpbys( x, b, y ) \
{ \
	(y)          = ( double ) bli_zreal(x) + ( double ) bli_zdimulnn_r( (b), (y) ); \
}

// -- (xby) = (?sc) ------------------------------------------------------------

#define bli_sscxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_sreal(x) + ( float  ) bli_scimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_scimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_dscxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_dreal(x) + ( float  ) bli_scimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_scimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_cscxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_creal(x) + ( float  ) bli_scimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_cimag(x) + ( float  ) bli_scimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_zscxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_zreal(x) + ( float  ) bli_scimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_zimag(x) + ( float  ) bli_scimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}

// -- (xby) = (?dc) ------------------------------------------------------------

#define bli_sdcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_sreal(x) + ( float  ) bli_dcimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_dcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_ddcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_dreal(x) + ( float  ) bli_dcimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_dcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_cdcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_creal(x) + ( float  ) bli_dcimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_cimag(x) + ( float  ) bli_dcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_zdcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_zreal(x) + ( float  ) bli_dcimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_zimag(x) + ( float  ) bli_dcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}

// -- (xby) = (?cc) ------------------------------------------------------------

#define bli_sccxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_sreal(x) + ( float  ) bli_ccimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_ccimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_dccxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_dreal(x) + ( float  ) bli_ccimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_ccimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_cccxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_creal(x) + ( float  ) bli_ccimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_cimag(x) + ( float  ) bli_ccimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_zccxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_zreal(x) + ( float  ) bli_ccimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_zimag(x) + ( float  ) bli_ccimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}

// -- (xby) = (?zc) ------------------------------------------------------------

#define bli_szcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_sreal(x) + ( float  ) bli_zcimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_zcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_dzcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_dreal(x) + ( float  ) bli_zcimulnn_r( (b), (y) ); \
	float  tempi =                           ( float  ) bli_zcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_czcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_creal(x) + ( float  ) bli_zcimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_cimag(x) + ( float  ) bli_zcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}
#define bli_zzcxpbys( x, b, y ) \
{ \
	float  tempr = ( float  ) bli_zreal(x) + ( float  ) bli_zcimulnn_r( (b), (y) ); \
	float  tempi = ( float  ) bli_zimag(x) + ( float  ) bli_zcimulnn_i( (b), (y) ); \
	bli_creal(y) = tempr; \
	bli_cimag(y) = tempi; \
}

// -- (xby) = (?sz) ------------------------------------------------------------

#define bli_sszxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_sreal(x) + ( double ) bli_szimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_szimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_dszxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_dreal(x) + ( double ) bli_szimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_szimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_cszxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_creal(x) + ( double ) bli_szimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_cimag(x) + ( double ) bli_szimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_zszxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_zreal(x) + ( double ) bli_szimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_zimag(x) + ( double ) bli_szimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}

// -- (xby) = (?dz) ------------------------------------------------------------

#define bli_sdzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_sreal(x) + ( double ) bli_dzimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_dzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_ddzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_dreal(x) + ( double ) bli_dzimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_dzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_cdzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_creal(x) + ( double ) bli_dzimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_cimag(x) + ( double ) bli_dzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_zdzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_zreal(x) + ( double ) bli_dzimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_zimag(x) + ( double ) bli_dzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}

// -- (xby) = (?cz) ------------------------------------------------------------

#define bli_sczxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_sreal(x) + ( double ) bli_czimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_czimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_dczxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_dreal(x) + ( double ) bli_czimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_czimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_cczxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_creal(x) + ( double ) bli_czimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_cimag(x) + ( double ) bli_czimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_zczxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_zreal(x) + ( double ) bli_czimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_zimag(x) + ( double ) bli_czimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}

// -- (xby) = (?zz) ------------------------------------------------------------

#define bli_szzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_sreal(x) + ( double ) bli_zzimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_zzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_dzzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_dreal(x) + ( double ) bli_zzimulnn_r( (b), (y) ); \
	double tempi =                           ( double ) bli_zzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_czzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_creal(x) + ( double ) bli_zzimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_cimag(x) + ( double ) bli_zzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}
#define bli_zzzxpbys( x, b, y ) \
{ \
	double tempr = ( double ) bli_zreal(x) + ( double ) bli_zzimulnn_r( (b), (y) ); \
	double tempi = ( double ) bli_zimag(x) + ( double ) bli_zzimulnn_i( (b), (y) ); \
	bli_zreal(y) = tempr; \
	bli_zimag(y) = tempi; \
}



#define bli_sxpbys( x, b, y )  bli_sssxpbys( x, b, y )
#define bli_dxpbys( x, b, y )  bli_dddxpbys( x, b, y )
#define bli_cxpbys( x, b, y )  bli_cccxpbys( x, b, y )
#define bli_zxpbys( x, b, y )  bli_zzzxpbys( x, b, y )


#endif
