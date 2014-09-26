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

extern func_t* gemm3mh_ukrs;
extern func_t* gemm3m_ukrs;
extern func_t* gemm4mh_ukrs;
extern func_t* gemm4m_ukrs;
extern func_t* gemm_ukrs;

func_t* bli_gemm_query_ukrs( num_t dt )
{
	if      ( bli_3mh_is_enabled_dt( dt ) ) return gemm3mh_ukrs;
	else if ( bli_3m_is_enabled_dt( dt ) )  return gemm3m_ukrs;
	else if ( bli_4mh_is_enabled_dt( dt ) ) return gemm4mh_ukrs;
	else if ( bli_4m_is_enabled_dt( dt ) )  return gemm4m_ukrs;
	else                                    return gemm_ukrs;
}

char* bli_gemm_query_impl_string( num_t dt )
{
	if      ( bli_3mh_is_enabled_dt( dt ) ) return bli_3mh_get_string();
	else if ( bli_3m_is_enabled_dt( dt ) )  return bli_3m_get_string();
	else if ( bli_4mh_is_enabled_dt( dt ) ) return bli_4mh_get_string();
	else if ( bli_4m_is_enabled_dt( dt ) )  return bli_4m_get_string();
	else                                    return bli_native_get_string();
}

kimpl_t bli_gemm_ukernel_impl_type( num_t dt )
{
	func_t* ukrs = bli_gemm_query_ukrs( dt );
	void*   p    = bli_func_obj_query( dt, ukrs );

	if      ( p == BLIS_SGEMM_UKERNEL_REF ||
	          p == BLIS_DGEMM_UKERNEL_REF ||
	          p == BLIS_CGEMM_UKERNEL_REF ||
	          p == BLIS_ZGEMM_UKERNEL_REF
	        ) return BLIS_REFERENCE_UKERNEL;
	else if (
	          p == BLIS_CGEMM3MH_UKERNEL_REF ||
	          p == BLIS_ZGEMM3MH_UKERNEL_REF ||
	          p == BLIS_CGEMM3M_UKERNEL_REF  ||
	          p == BLIS_ZGEMM3M_UKERNEL_REF
	        ) return BLIS_VIRTUAL3M_UKERNEL;
	else if (
	          p == BLIS_CGEMM4MH_UKERNEL_REF ||
	          p == BLIS_ZGEMM4MH_UKERNEL_REF ||
	          p == BLIS_CGEMM4M_UKERNEL_REF  ||
	          p == BLIS_ZGEMM4M_UKERNEL_REF
	        ) return BLIS_VIRTUAL4M_UKERNEL;
	else 
	          return BLIS_OPTIMIZED_UKERNEL;
}

