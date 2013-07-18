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

#ifndef BLIS_IMULNN_H
#define BLIS_IMULNN_H

// imulnn_r, imulnn_i

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - Neither operand is used in conjugated form.


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_ssimulnn_r( a, b )  ( ( float  ) bli_sreal(a) * ( float  ) bli_sreal(b) )
#define bli_ssimulnn_i( a, b )  (                                              0.0F )

#define bli_dsimulnn_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_sreal(b) )
#define bli_dsimulnn_i( a, b )  (                                              0.0F )

#define bli_csimulnn_r( a, b )  ( ( float  ) bli_creal(a) * ( float  ) bli_sreal(b) )
#define bli_csimulnn_i( a, b )  ( ( float  ) bli_cimag(a) * ( float  ) bli_sreal(b) )

#define bli_zsimulnn_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_sreal(b) )
#define bli_zsimulnn_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_sreal(b) )


#define bli_sdimulnn_r( a, b )  ( ( double ) bli_sreal(a) * ( double ) bli_sreal(b) )
#define bli_sdimulnn_i( a, b )  (                                              0.0  )

#define bli_ddimulnn_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_sreal(b) )
#define bli_ddimulnn_i( a, b )  (                                              0.0  )

#define bli_cdimulnn_r( a, b )  ( ( double ) bli_creal(a) * ( double ) bli_sreal(b) )
#define bli_cdimulnn_i( a, b )  ( ( double ) bli_cimag(a) * ( double ) bli_sreal(b) )

#define bli_zdimulnn_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_sreal(b) )
#define bli_zdimulnn_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_sreal(b) )



#define bli_scimulnn_r( a, b )  ( ( float  ) bli_sreal(a) * ( float  ) bli_creal(b) )
#define bli_scimulnn_i( a, b )  ( ( float  ) bli_sreal(a) * ( float  ) bli_cimag(b) )

#define bli_dcimulnn_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_creal(b) )
#define bli_dcimulnn_i( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_cimag(b) )

#define bli_ccimulnn_r( a, b )  ( ( float  ) bli_creal(a) * ( float  ) bli_creal(b) - \
                                  ( float  ) bli_cimag(a) * ( float  ) bli_cimag(b) )
#define bli_ccimulnn_i( a, b )  ( ( float  ) bli_cimag(a) * ( float  ) bli_creal(b) + \
                                  ( float  ) bli_creal(a) * ( float  ) bli_cimag(b) )

#define bli_zcimulnn_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_creal(b) - \
                                  ( double ) bli_zimag(a) * ( double ) bli_cimag(b) )
#define bli_zcimulnn_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_creal(b) + \
                                  ( double ) bli_zreal(a) * ( double ) bli_cimag(b) )



#define bli_szimulnn_r( a, b )  ( ( double ) bli_sreal(a) * ( double ) bli_zreal(b) )
#define bli_szimulnn_i( a, b )  ( ( double ) bli_sreal(a) * ( double ) bli_zimag(b) )

#define bli_dzimulnn_r( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_zreal(b) )
#define bli_dzimulnn_i( a, b )  ( ( double ) bli_dreal(a) * ( double ) bli_zimag(b) )

#define bli_czimulnn_r( a, b )  ( ( double ) bli_creal(a) * ( double ) bli_zreal(b) - \
                                  ( double ) bli_cimag(a) * ( double ) bli_zimag(b) )
#define bli_czimulnn_i( a, b )  ( ( double ) bli_cimag(a) * ( double ) bli_zreal(b) + \
                                  ( double ) bli_creal(a) * ( double ) bli_zimag(b) )

#define bli_zzimulnn_r( a, b )  ( ( double ) bli_zreal(a) * ( double ) bli_zreal(b) - \
                                  ( double ) bli_zimag(a) * ( double ) bli_zimag(b) )
#define bli_zzimulnn_i( a, b )  ( ( double ) bli_zimag(a) * ( double ) bli_zreal(b) + \
                                  ( double ) bli_zreal(a) * ( double ) bli_zimag(b) )


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_ssimulnn_r( a, b )  ( bli_sreal( ( float    )(a) * ( float    )(b) ) )
#define bli_ssimulnn_i( a, b )  (                                          0.0F  )

#define bli_dsimulnn_r( a, b )  ( bli_dreal( ( double   )(a) * ( double   )(b) ) )
#define bli_dsimulnn_i( a, b )  (                                          0.0   )

#define bli_csimulnn_r( a, b )  ( bli_creal( ( scomplex )(a) * ( float    )(b) ) )
#define bli_csimulnn_i( a, b )  ( bli_cimag( ( scomplex )(a) * ( float    )(b) ) )

#define bli_zsimulnn_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( double   )(b) ) )
#define bli_zsimulnn_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( double   )(b) ) )


#define bli_sdimulnn_r( a, b )  ( bli_dreal( ( double   )(a) * ( double   )(b) ) )
#define bli_sdimulnn_i( a, b )  (                                          0.0   )

#define bli_ddimulnn_r( a, b )  ( bli_dreal( ( double   )(a) * ( double   )(b) ) )
#define bli_ddimulnn_i( a, b )  (                                          0.0   )

#define bli_cdimulnn_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( double   )(b) ) )
#define bli_cdimulnn_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( double   )(b) ) )

#define bli_zdimulnn_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( double   )(b) ) )
#define bli_zdimulnn_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( double   )(b) ) )


#define bli_scimulnn_r( a, b )  ( bli_creal( ( float    )(a) * ( scomplex )(b) ) )
#define bli_scimulnn_i( a, b )  ( bli_cimag( ( float    )(a) * ( scomplex )(b) ) )

#define bli_dcimulnn_r( a, b )  ( bli_zreal( ( double   )(a) * ( dcomplex )(b) ) )
#define bli_dcimulnn_i( a, b )  ( bli_zimag( ( double   )(a) * ( dcomplex )(b) ) )

#define bli_ccimulnn_r( a, b )  ( bli_creal( ( scomplex )(a) * ( scomplex )(b) ) )
#define bli_ccimulnn_i( a, b )  ( bli_cimag( ( scomplex )(a) * ( scomplex )(b) ) )

#define bli_zcimulnn_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( dcomplex )(b) ) )
#define bli_zcimulnn_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( dcomplex )(b) ) )


#define bli_szimulnn_r( a, b )  ( bli_zreal( ( double   )(a) * ( dcomplex )(b) ) )
#define bli_szimulnn_i( a, b )  ( bli_zimag( ( double   )(a) * ( dcomplex )(b) ) )

#define bli_dzimulnn_r( a, b )  ( bli_zreal( ( double   )(a) * ( dcomplex )(b) ) )
#define bli_dzimulnn_i( a, b )  ( bli_zimag( ( double   )(a) * ( dcomplex )(b) ) )

#define bli_czimulnn_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( dcomplex )(b) ) )
#define bli_czimulnn_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( dcomplex )(b) ) )

#define bli_zzimulnn_r( a, b )  ( bli_zreal( ( dcomplex )(a) * ( dcomplex )(b) ) )
#define bli_zzimulnn_i( a, b )  ( bli_zimag( ( dcomplex )(a) * ( dcomplex )(b) ) )


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_simulnn_r( a, b )  bli_ssimulnn_r( a, b )
#define bli_simulnn_i( a, b )  bli_ssimulnn_i( a, b )

#define bli_dimulnn_r( a, b )  bli_ddimulnn_r( a, b )
#define bli_dimulnn_i( a, b )  bli_ddimulnn_i( a, b )

#define bli_cimulnn_r( a, b )  bli_ccimulnn_r( a, b )
#define bli_cimulnn_i( a, b )  bli_ccimulnn_i( a, b )

#define bli_zimulnn_r( a, b )  bli_zzimulnn_r( a, b )
#define bli_zimulnn_i( a, b )  bli_zzimulnn_i( a, b )


#endif
