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

extern func_t* gemmtrsm3m_l_ukrs;
extern func_t* gemmtrsm3m_u_ukrs;
extern func_t* gemmtrsm4m_l_ukrs;
extern func_t* gemmtrsm4m_u_ukrs;
extern func_t* gemmtrsm_l_ukrs;
extern func_t* gemmtrsm_u_ukrs;

extern func_t* trsm3m_l_ukrs;
extern func_t* trsm3m_u_ukrs;
extern func_t* trsm4m_l_ukrs;
extern func_t* trsm4m_u_ukrs;
extern func_t* trsm_l_ukrs;
extern func_t* trsm_u_ukrs;

func_t* bli_gemmtrsm_query_ukrs( uplo_t uplo, num_t dt )
{
	if      ( bli_3m_is_enabled_dt( dt ) )
		return ( bli_is_lower( uplo ) ? gemmtrsm3m_l_ukrs
		                              : gemmtrsm3m_u_ukrs );
	else if ( bli_4m_is_enabled_dt( dt ) )
		return ( bli_is_lower( uplo ) ? gemmtrsm4m_l_ukrs
		                              : gemmtrsm4m_u_ukrs );
	else
		return ( bli_is_lower( uplo ) ? gemmtrsm_l_ukrs
		                              : gemmtrsm_u_ukrs );
}

func_t* bli_trsm_query_ukrs( uplo_t uplo, num_t dt )
{
	if      ( bli_3m_is_enabled_dt( dt ) )
		return ( bli_is_lower( uplo ) ? trsm3m_l_ukrs
		                              : trsm3m_u_ukrs );
	else if ( bli_4m_is_enabled_dt( dt ) )
		return ( bli_is_lower( uplo ) ? trsm4m_l_ukrs
		                              : trsm4m_u_ukrs );
	else
		return ( bli_is_lower( uplo ) ? trsm_l_ukrs
		                              : trsm_u_ukrs );
}

char* bli_trsm_query_impl_string( num_t dt )
{
	if      ( bli_3m_is_enabled_dt( dt ) ) return bli_3m_get_string();
	else if ( bli_4m_is_enabled_dt( dt ) ) return bli_4m_get_string();
	else                                   return bli_native_get_string();
}

kimpl_t bli_gemmtrsm_l_ukernel_impl_type( num_t dt )
{
	func_t* ukrs = bli_gemmtrsm_query_ukrs( BLIS_LOWER, dt );
	void*   p    = bli_func_obj_query( dt, ukrs );

	if      ( p == BLIS_SGEMMTRSM_L_UKERNEL_REF ||
	          p == BLIS_DGEMMTRSM_L_UKERNEL_REF ||
	          p == BLIS_CGEMMTRSM_L_UKERNEL_REF ||
	          p == BLIS_ZGEMMTRSM_L_UKERNEL_REF
	        ) return BLIS_REFERENCE_UKERNEL;
	else if (
	          p == BLIS_CGEMMTRSM3M_L_UKERNEL_REF ||
	          p == BLIS_ZGEMMTRSM3M_L_UKERNEL_REF
	        ) return BLIS_VIRTUAL3M_UKERNEL;
	else if (
	          p == BLIS_CGEMMTRSM4M_L_UKERNEL_REF ||
	          p == BLIS_ZGEMMTRSM4M_L_UKERNEL_REF
	        ) return BLIS_VIRTUAL4M_UKERNEL;
	else 
	          return BLIS_OPTIMIZED_UKERNEL;
}

kimpl_t bli_gemmtrsm_u_ukernel_impl_type( num_t dt )
{
	func_t* ukrs = bli_gemmtrsm_query_ukrs( BLIS_UPPER, dt );
	void*   p    = bli_func_obj_query( dt, ukrs );

	if      ( p == BLIS_SGEMMTRSM_U_UKERNEL_REF ||
	          p == BLIS_DGEMMTRSM_U_UKERNEL_REF ||
	          p == BLIS_CGEMMTRSM_U_UKERNEL_REF ||
	          p == BLIS_ZGEMMTRSM_U_UKERNEL_REF
	        ) return BLIS_REFERENCE_UKERNEL;
	else if (
	          p == BLIS_CGEMMTRSM3M_U_UKERNEL_REF ||
	          p == BLIS_ZGEMMTRSM3M_U_UKERNEL_REF
	        ) return BLIS_VIRTUAL3M_UKERNEL;
	else if (
	          p == BLIS_CGEMMTRSM4M_U_UKERNEL_REF ||
	          p == BLIS_ZGEMMTRSM4M_U_UKERNEL_REF
	        ) return BLIS_VIRTUAL4M_UKERNEL;
	else 
	          return BLIS_OPTIMIZED_UKERNEL;
}

kimpl_t bli_trsm_l_ukernel_impl_type( num_t dt )
{
	func_t* ukrs = bli_trsm_query_ukrs( BLIS_LOWER, dt );
	void*   p    = bli_func_obj_query( dt, ukrs );

	if      ( p == BLIS_STRSM_L_UKERNEL_REF ||
	          p == BLIS_DTRSM_L_UKERNEL_REF ||
	          p == BLIS_CTRSM_L_UKERNEL_REF ||
	          p == BLIS_ZTRSM_L_UKERNEL_REF
	        ) return BLIS_REFERENCE_UKERNEL;
	else if (
	          p == BLIS_CTRSM3M_L_UKERNEL_REF ||
	          p == BLIS_ZTRSM3M_L_UKERNEL_REF
	        ) return BLIS_VIRTUAL3M_UKERNEL;
	else if (
	          p == BLIS_CTRSM4M_L_UKERNEL_REF ||
	          p == BLIS_ZTRSM4M_L_UKERNEL_REF
	        ) return BLIS_VIRTUAL4M_UKERNEL;
	else 
	          return BLIS_OPTIMIZED_UKERNEL;
}

kimpl_t bli_trsm_u_ukernel_impl_type( num_t dt )
{
	func_t* ukrs = bli_trsm_query_ukrs( BLIS_UPPER, dt );
	void*   p    = bli_func_obj_query( dt, ukrs );

	if      ( p == BLIS_STRSM_U_UKERNEL_REF ||
	          p == BLIS_DTRSM_U_UKERNEL_REF ||
	          p == BLIS_CTRSM_U_UKERNEL_REF ||
	          p == BLIS_ZTRSM_U_UKERNEL_REF
	        ) return BLIS_REFERENCE_UKERNEL;
	else if (
	          p == BLIS_CTRSM3M_U_UKERNEL_REF ||
	          p == BLIS_ZTRSM3M_U_UKERNEL_REF
	        ) return BLIS_VIRTUAL3M_UKERNEL;
	else if (
	          p == BLIS_CTRSM4M_U_UKERNEL_REF ||
	          p == BLIS_ZTRSM4M_U_UKERNEL_REF
	        ) return BLIS_VIRTUAL4M_UKERNEL;
	else 
	          return BLIS_OPTIMIZED_UKERNEL;
}

