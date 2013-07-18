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

#ifndef BLIS_IMULCN_H
#define BLIS_IMULCN_H

// imulcn_r, imulcn_i

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of b.
// - a is used in conjugated form.


#define bli_ssimulcn_r( a, b )  bli_ssimulnc_r( b, a )
#define bli_dsimulcn_r( a, b )  bli_sdimulnc_r( b, a )
#define bli_csimulcn_r( a, b )  bli_scimulnc_r( b, a )
#define bli_zsimulcn_r( a, b )  bli_szimulnc_r( b, a )

#define bli_sdimulcn_r( a, b )  bli_dsimulnc_r( b, a )
#define bli_ddimulcn_r( a, b )  bli_ddimulnc_r( b, a )
#define bli_cdimulcn_r( a, b )  bli_dcimulnc_r( b, a )
#define bli_zdimulcn_r( a, b )  bli_dzimulnc_r( b, a )

#define bli_scimulcn_r( a, b )  bli_csimulnc_r( b, a )
#define bli_dcimulcn_r( a, b )  bli_cdimulnc_r( b, a )
#define bli_ccimulcn_r( a, b )  bli_ccimulnc_r( b, a )
#define bli_zcimulcn_r( a, b )  bli_czimulnc_r( b, a )

#define bli_szimulcn_r( a, b )  bli_zsimulnc_r( b, a )
#define bli_dzimulcn_r( a, b )  bli_zdimulnc_r( b, a )
#define bli_czimulcn_r( a, b )  bli_zcimulnc_r( b, a )
#define bli_zzimulcn_r( a, b )  bli_zzimulnc_r( b, a )


#define bli_ssimulcn_i( a, b )  bli_ssimulnc_i( b, a )
#define bli_dsimulcn_i( a, b )  bli_sdimulnc_i( b, a )
#define bli_csimulcn_i( a, b )  bli_scimulnc_i( b, a )
#define bli_zsimulcn_i( a, b )  bli_szimulnc_i( b, a )

#define bli_sdimulcn_i( a, b )  bli_dsimulnc_i( b, a )
#define bli_ddimulcn_i( a, b )  bli_ddimulnc_i( b, a )
#define bli_cdimulcn_i( a, b )  bli_dcimulnc_i( b, a )
#define bli_zdimulcn_i( a, b )  bli_dzimulnc_i( b, a )

#define bli_scimulcn_i( a, b )  bli_csimulnc_i( b, a )
#define bli_dcimulcn_i( a, b )  bli_cdimulnc_i( b, a )
#define bli_ccimulcn_i( a, b )  bli_ccimulnc_i( b, a )
#define bli_zcimulcn_i( a, b )  bli_czimulnc_i( b, a )

#define bli_szimulcn_i( a, b )  bli_zsimulnc_i( b, a )
#define bli_dzimulcn_i( a, b )  bli_zdimulnc_i( b, a )
#define bli_czimulcn_i( a, b )  bli_zcimulnc_i( b, a )
#define bli_zzimulcn_i( a, b )  bli_zzimulnc_i( b, a )



#define bli_simulcn_r( a, b )  bli_ssimulcn_r( a, b )
#define bli_dimulcn_r( a, b )  bli_ddimulcn_r( a, b )
#define bli_cimulcn_r( a, b )  bli_ccimulcn_r( a, b )
#define bli_zimulcn_r( a, b )  bli_zzimulcn_r( a, b )

#define bli_simulcn_i( a, b )  bli_ssimulcn_i( a, b )
#define bli_dimulcn_i( a, b )  bli_ddimulcn_i( a, b )
#define bli_cimulcn_i( a, b )  bli_ccimulcn_i( a, b )
#define bli_zimulcn_i( a, b )  bli_zzimulcn_i( a, b )


#endif
