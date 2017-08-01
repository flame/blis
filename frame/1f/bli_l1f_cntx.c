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
// Define context initialization functions.
//

#undef  GENFRONT
#define GENFRONT( opname, kertype, depname ) \
\
void PASTEMAC(opname,_cntx_init)( num_t dt, cntx_t* cntx ) \
{ \
	bli_cntx_create( cntx ); \
\
	/* Initialize the context with kernel dependencies. */ \
	PASTEMAC(depname,_cntx_init)( dt, cntx ); \
\
	/* Initialize the context with the kernel associated with the current
	   operation. */ \
	bli_gks_cntx_set_l1f_ker( kertype, cntx ); \
} \
void PASTEMAC(opname,_cntx_finalize)( cntx_t* cntx ) \
{ \
	bli_cntx_free( cntx ); \
}

GENFRONT( axpy2v, BLIS_AXPY2V_KER, axpyv )


#undef  GENFRONT
#define GENFRONT( opname, kertype, depname1, depname2 ) \
\
void PASTEMAC(opname,_cntx_init)( num_t dt, cntx_t* cntx ) \
{ \
	bli_cntx_create( cntx ); \
\
	/* Initialize the context with kernel dependencies. */ \
	PASTEMAC(depname1,_cntx_init)( dt, cntx ); \
	PASTEMAC(depname2,_cntx_init)( dt, cntx ); \
\
	/* Initialize the context with the kernel associated with the current
	   operation. */ \
	bli_gks_cntx_set_l1f_ker( kertype, cntx ); \
} \
void PASTEMAC(opname,_cntx_finalize)( cntx_t* cntx ) \
{ \
	bli_cntx_free( cntx ); \
}

GENFRONT( dotaxpyv, BLIS_DOTAXPYV_KER, dotxv, axpyv )


#undef  GENFRONT
#define GENFRONT( opname, kertype, depname ) \
\
void PASTEMAC(opname,_cntx_init)( num_t dt, cntx_t* cntx ) \
{ \
	bli_cntx_create( cntx ); \
\
	/* Initialize the context with kernel dependencies. */ \
	PASTEMAC(depname,_cntx_init)( dt, cntx ); \
\
	/* Initialize the context with the kernel associated with the current
	   operation. */ \
	bli_gks_cntx_set_l1f_ker( kertype, cntx ); \
\
	/* Initialize the context with the current architecture's level-1f
	   fusing blocksizes. */ \
	bli_gks_cntx_set_blkszs( BLIS_NAT, 1, \
	                         BLIS_AF, BLIS_AF, /* axpyf fusing factor */ \
	                         cntx ); \
} \
void PASTEMAC(opname,_cntx_finalize)( cntx_t* cntx ) \
{ \
	bli_cntx_free( cntx ); \
}

GENFRONT( axpyf, BLIS_AXPYF_KER, axpyv )


#undef  GENFRONT
#define GENFRONT( opname, kertype, depname1, depname2 ) \
\
void PASTEMAC(opname,_cntx_init)( num_t dt, cntx_t* cntx ) \
{ \
	bli_cntx_create( cntx ); \
\
	/* Initialize the context with kernel dependencies. */ \
	PASTEMAC(depname1,_cntx_init)( dt, cntx ); \
	PASTEMAC(depname2,_cntx_init)( dt, cntx ); \
\
	/* Initialize the context with the kernel associated with the current
	   operation. */ \
	bli_gks_cntx_set_l1f_ker( kertype, cntx ); \
\
	/* Initialize the context with the current architecture's level-1f
	   fusing blocksizes. */ \
	bli_gks_cntx_set_blkszs( BLIS_NAT, 2, \
	                         BLIS_DF, BLIS_DF, /* dotxf fusing factor */ \
	                         BLIS_XF, BLIS_XF, /* dotxaxpyf fusing factor */ \
	                         cntx ); \
} \
void PASTEMAC(opname,_cntx_finalize)( cntx_t* cntx ) \
{ \
	bli_cntx_free( cntx ); \
}

GENFRONT( dotxf,     BLIS_DOTXF_KER,     dotv,  dotxv )
GENFRONT( dotxaxpyf, BLIS_DOTXAXPYF_KER, dotxf, axpyf )

