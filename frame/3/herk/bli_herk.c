/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#include "blis.h"

//
// Define object-based interface.
//
void bli_herk( obj_t*  alpha,
               obj_t*  a,
               obj_t*  beta,
               obj_t*  c )
{
	num_t dt = bli_obj_datatype( *c );

	if      ( bli_3mh_is_enabled_dt( dt ) ) bli_herk3mh_entry( alpha, a, beta, c );
	else if ( bli_3m1_is_enabled_dt( dt ) )  bli_herk3m1_entry( alpha, a, beta, c );
	else if ( bli_4mh_is_enabled_dt( dt ) ) bli_herk4mh_entry( alpha, a, beta, c );
	else if ( bli_4m1_is_enabled_dt( dt ) )  bli_herk4m1_entry( alpha, a, beta, c );
	else                                    bli_herk_entry( alpha, a, beta, c );
}


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          uplo_t    uploc, \
                          trans_t   transa, \
                          dim_t     m, \
                          dim_t     k, \
                          ctype_r*  alpha, \
                          ctype*    a, inc_t rs_a, inc_t cs_a, \
                          ctype_r*  beta, \
                          ctype*    c, inc_t rs_c, inc_t cs_c  \
                        ) \
{ \
	const num_t dt_r = PASTEMAC(chr,type); \
	const num_t dt   = PASTEMAC(ch,type); \
\
	obj_t       alphao, ao, betao, co; \
\
	dim_t       m_a, n_a; \
\
	bli_set_dims_with_trans( transa, m, k, m_a, n_a ); \
\
	bli_obj_create_1x1_with_attached_buffer( dt_r, alpha, &alphao ); \
	bli_obj_create_1x1_with_attached_buffer( dt_r, beta,  &betao  ); \
\
	bli_obj_create_with_attached_buffer( dt, m_a, n_a, a, rs_a, cs_a, &ao ); \
	bli_obj_create_with_attached_buffer( dt, m,   m,   c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploc, co ); \
	bli_obj_set_conjtrans( transa, ao ); \
\
	bli_obj_set_struc( BLIS_HERMITIAN, co ); \
\
	PASTEMAC0(opname)( &alphao, \
	                   &ao, \
	                   &betao, \
	                   &co ); \
}

INSERT_GENTFUNCR_BASIC( herk, herk )


//
// Define BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTFUNC2R
#define GENTFUNC2R( ctype_a, ctype_c, ctype_ar, cha, chc, chAr, opname, varname ) \
\
void PASTEMAC2(cha,chc,opname)( \
                                uplo_t    uploc, \
                                trans_t   transa, \
                                dim_t     m, \
                                dim_t     k, \
                                ctype_ar* alpha, \
                                ctype_a*  a, inc_t rs_a, inc_t cs_a, \
                                ctype_c*  beta, \
                                ctype_c*  c, inc_t rs_c, inc_t cs_c  \
                              ) \
{ \
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

INSERT_GENTFUNC2R_BASIC( herk, herk )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2R_MIX_D( herk, herk )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2R_MIX_P( herk, herk )
#endif

