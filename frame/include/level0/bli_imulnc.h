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

#ifndef BLIS_IMULNC_H
#define BLIS_IMULNC_H

// imulnc_r, imulnc_i

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of b.
// - b is used in conjugated form.


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_ssimulnc_r( a, b )  ( ( float  ) bli_sreal(a) * ( float  ) bli_sreal(b) )
#define bli_ssimulnc_i( a, b )  (                                              0.0F )

#define bli_dsimulnc_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_sreal(b) )
#define bli_dsimulnc_i( a, b )  (                                              0.0F )

#define bli_csimulnc_r( a, b )  ( ( float  ) bli_creal(a) * ( float  ) bli_sreal(b) )
#define bli_csimulnc_i( a, b )  ( ( float  ) bli_cimag(a) * ( float  ) bli_sreal(b) )

#define bli_zsimulnc_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_sreal(b) )
#define bli_zsimulnc_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_sreal(b) )


#define bli_sdimulnc_r( a, b )  ( ( double ) bli_sreal(a) * ( double ) bli_sreal(b) )
#define bli_sdimulnc_i( a, b )  (                                              0.0  )

#define bli_ddimulnc_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_sreal(b) )
#define bli_ddimulnc_i( a, b )  (                                              0.0  )

#define bli_cdimulnc_r( a, b )  ( ( double ) bli_creal(a) * ( double ) bli_sreal(b) )
#define bli_cdimulnc_i( a, b )  ( ( double ) bli_cimag(a) * ( double ) bli_sreal(b) )

#define bli_zdimulnc_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_sreal(b) )
#define bli_zdimulnc_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_sreal(b) )



#define bli_scimulnc_r( a, b )  ( ( float  ) bli_sreal(a) * ( float  ) bli_creal(b) )
#define bli_scimulnc_i( a, b )  ( ( float  ) bli_sreal(a) * ( float  )-bli_cimag(b) )

#define bli_dcimulnc_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_creal(b) )
#define bli_dcimulnc_i( a, b )  ( ( double ) bli_dreal(a) * ( double )-bli_cimag(b) )

#define bli_ccimulnc_r( a, b )  ( ( float  ) bli_creal(a) * ( float  ) bli_creal(b) - \
                                  ( float  ) bli_cimag(a) * ( float  )-bli_cimag(b) )
#define bli_ccimulnc_i( a, b )  ( ( float  ) bli_cimag(a) * ( float  ) bli_creal(b) + \
                                  ( float  ) bli_creal(a) * ( float  )-bli_cimag(b) )

#define bli_zcimulnc_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_creal(b) - \
                                  ( double ) bli_zimag(a) * ( double )-bli_cimag(b) )
#define bli_zcimulnc_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_creal(b) + \
                                  ( double ) bli_zreal(a) * ( double )-bli_cimag(b) )



#define bli_szimulnc_r( a, b )  ( ( double ) bli_sreal(a) * ( double ) bli_zreal(b) )
#define bli_szimulnc_i( a, b )  ( ( double ) bli_sreal(a) * ( double )-bli_zimag(b) )

#define bli_dzimulnc_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_zreal(b) )
#define bli_dzimulnc_i( a, b )  ( ( double ) bli_dreal(a) * ( double )-bli_zimag(b) )

#define bli_czimulnc_r( a, b )  ( ( double ) bli_creal(a) * ( double ) bli_zreal(b) - \
                                  ( double ) bli_cimag(a) * ( double )-bli_zimag(b) )
#define bli_czimulnc_i( a, b )  ( ( double ) bli_cimag(a) * ( double ) bli_zreal(b) + \
                                  ( double ) bli_creal(a) * ( double )-bli_zimag(b) )

#define bli_zzimulnc_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_zreal(b) - \
                                  ( double ) bli_zimag(a) * ( double )-bli_zimag(b) )
#define bli_zzimulnc_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_zreal(b) + \
                                  ( double ) bli_zreal(a) * ( double )-bli_zimag(b) )


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_ssimulnc_r( a, b )  ( bli_sreal( ( float    )(a) * ( float    )(b) ) )
#define bli_ssimulnc_i( a, b )  (                                          0.0F  )

#define bli_dsimulnc_r( a, b )  ( bli_dreal( ( double   )(a) * ( double   )(b) ) )
#define bli_dsimulnc_i( a, b )  (                                          0.0   )

#define bli_csimulnc_r( a, b )  ( bli_creal( ( scomplex )(a) * ( float    )(b) ) )
#define bli_csimulnc_i( a, b )  ( bli_cimag( ( scomplex )(a) * ( float    )(b) ) )

#define bli_zsimulnc_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( double   )(b) ) )
#define bli_zsimulnc_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( double   )(b) ) )


#define bli_sdimulnc_r( a, b )  ( bli_dreal( ( double   )(a) * ( double   )(b) ) )
#define bli_sdimulnc_i( a, b )  (                                          0.0   )

#define bli_ddimulnc_r( a, b )  ( bli_dreal( ( double   )(a) * ( double   )(b) ) )
#define bli_ddimulnc_i( a, b )  (                                          0.0   )

#define bli_cdimulnc_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( double   )(b) ) )
#define bli_cdimulnc_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( double   )(b) ) )

#define bli_zdimulnc_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( double   )(b) ) )
#define bli_zdimulnc_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( double   )(b) ) )


#define bli_scimulnc_r( a, b )  ( bli_creal( ( float    )(a) * ( scomplex )conjf(b) ) )
#define bli_scimulnc_i( a, b )  ( bli_cimag( ( float    )(a) * ( scomplex )conjf(b) ) )

#define bli_dcimulnc_r( a, b )  ( bli_zreal( ( double   )(a) * ( dcomplex )conjf(b) ) )
#define bli_dcimulnc_i( a, b )  ( bli_zimag( ( double   )(a) * ( dcomplex )conjf(b) ) )

#define bli_ccimulnc_r( a, b )  ( bli_creal( ( scomplex )(a) * ( scomplex )conjf(b) ) )
#define bli_ccimulnc_i( a, b )  ( bli_cimag( ( scomplex )(a) * ( scomplex )conjf(b) ) )

#define bli_zcimulnc_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( dcomplex )conjf(b) ) )
#define bli_zcimulnc_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( dcomplex )conjf(b) ) )


#define bli_szimulnc_r( a, b )  ( bli_zreal( ( double   )(a) * ( dcomplex )conj(b) ) )
#define bli_szimulnc_i( a, b )  ( bli_zimag( ( double   )(a) * ( dcomplex )conj(b) ) )

#define bli_dzimulnc_r( a, b )  ( bli_zreal( ( double   )(a) * ( dcomplex )conj(b) ) )
#define bli_dzimulnc_i( a, b )  ( bli_zimag( ( double   )(a) * ( dcomplex )conj(b) ) )

#define bli_czimulnc_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( dcomplex )conj(b) ) )
#define bli_czimulnc_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( dcomplex )conj(b) ) )

#define bli_zzimulnc_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( dcomplex )conj(b) ) )
#define bli_zzimulnc_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( dcomplex )conj(b) ) )


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_simulnc_r( a, b )  bli_ssimulnc_r( a, b )
#define bli_simulnc_i( a, b )  bli_ssimulnc_i( a, b )

#define bli_dimulnc_r( a, b )  bli_ddimulnc_r( a, b )
#define bli_dimulnc_i( a, b )  bli_ddimulnc_i( a, b )

#define bli_cimulnc_r( a, b )  bli_ccimulnc_r( a, b )
#define bli_cimulnc_i( a, b )  bli_ccimulnc_i( a, b )

#define bli_zimulnc_r( a, b )  bli_zzimulnc_r( a, b )
#define bli_zimulnc_i( a, b )  bli_zzimulnc_i( a, b )


#endif
