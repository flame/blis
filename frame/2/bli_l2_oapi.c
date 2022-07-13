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
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

// Guard the function definitions so that they are only compiled when
// #included from files that define the object API macros.
#ifdef BLIS_ENABLE_OAPI

//
// Define object-based interfaces.
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* x, \
       const obj_t* beta, \
       const obj_t* y  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t     dt        = bli_obj_dt( a ); \
\
	trans_t   transa    = bli_obj_conjtrans_status( a ); \
	conj_t    conjx     = bli_obj_conj_status( x ); \
	dim_t     m         = bli_obj_length( a ); \
	dim_t     n         = bli_obj_width( a ); \
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	inc_t     rs_a      = bli_obj_row_stride( a ); \
	inc_t     cs_a      = bli_obj_col_stride( a ); \
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
	void*     buf_y     = bli_obj_buffer_at_off( y ); \
	inc_t     incy      = bli_obj_vector_inc( y ); \
\
	void*     buf_alpha; \
	void*     buf_beta; \
\
	obj_t     alpha_local; \
	obj_t     beta_local; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, x, beta, y ); \
\
	/* Create local copy-casts of scalars (and apply internal conjugation
	   as needed). */ \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      alpha, &alpha_local ); \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      beta, &beta_local ); \
	buf_alpha = bli_obj_buffer_for_1x1( dt, &alpha_local ); \
	buf_beta  = bli_obj_buffer_for_1x1( dt, &beta_local ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  transa, \
	  conjx, \
	  m, \
	  n, \
	  buf_alpha, \
	  buf_a, rs_a, cs_a, \
	  buf_x, incx, \
	  buf_beta, \
	  buf_y, incy, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( gemv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* x, \
       const obj_t* y, \
       const obj_t* a  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t     dt        = bli_obj_dt( a ); \
\
	conj_t    conjx     = bli_obj_conj_status( x ); \
	conj_t    conjy     = bli_obj_conj_status( y ); \
	dim_t     m         = bli_obj_length( a ); \
	dim_t     n         = bli_obj_width( a ); \
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
	void*     buf_y     = bli_obj_buffer_at_off( y ); \
	inc_t     incy      = bli_obj_vector_inc( y ); \
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	inc_t     rs_a      = bli_obj_row_stride( a ); \
	inc_t     cs_a      = bli_obj_col_stride( a ); \
\
	void*     buf_alpha; \
\
	obj_t     alpha_local; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, x, y, a ); \
\
	/* Create local copy-casts of scalars (and apply internal conjugation
	   as needed). */ \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      alpha, &alpha_local ); \
	buf_alpha = bli_obj_buffer_for_1x1( dt, &alpha_local ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  conjx, \
	  conjy, \
	  m, \
	  n, \
	  buf_alpha, \
	  buf_x, incx, \
	  buf_y, incy, \
	  buf_a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( ger )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* x, \
       const obj_t* beta, \
       const obj_t* y  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t     dt        = bli_obj_dt( a ); \
\
	uplo_t    uploa     = bli_obj_uplo( a ); \
	conj_t    conja     = bli_obj_conj_status( a ); \
	conj_t    conjx     = bli_obj_conj_status( x ); \
	dim_t     m         = bli_obj_length( a ); \
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	inc_t     rs_a      = bli_obj_row_stride( a ); \
	inc_t     cs_a      = bli_obj_col_stride( a ); \
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
	void*     buf_y     = bli_obj_buffer_at_off( y ); \
	inc_t     incy      = bli_obj_vector_inc( y ); \
\
	void*     buf_alpha; \
	void*     buf_beta; \
\
	obj_t     alpha_local; \
	obj_t     beta_local; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, x, beta, y ); \
\
	/* Create local copy-casts of scalars (and apply internal conjugation
	   as needed). */ \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      alpha, &alpha_local ); \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      beta, &beta_local ); \
	buf_alpha = bli_obj_buffer_for_1x1( dt, &alpha_local ); \
	buf_beta  = bli_obj_buffer_for_1x1( dt, &beta_local ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  uploa, \
	  conja, \
	  conjx, \
	  m, \
	  buf_alpha, \
	  buf_a, rs_a, cs_a, \
	  buf_x, incx, \
	  buf_beta, \
	  buf_y, incy, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( hemv )
GENFRONT( symv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* x, \
       const obj_t* a  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t     dt        = bli_obj_dt( a ); \
\
	uplo_t    uploa     = bli_obj_uplo( a ); \
	conj_t    conjx     = bli_obj_conj_status( x ); \
	dim_t     m         = bli_obj_length( a ); \
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	inc_t     rs_a      = bli_obj_row_stride( a ); \
	inc_t     cs_a      = bli_obj_col_stride( a ); \
\
	void*     buf_alpha; \
\
	obj_t     alpha_local; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, x, a ); \
\
	/* Create local copy-casts of scalars (and apply internal conjugation
	   as needed). */ \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      alpha, &alpha_local ); \
	buf_alpha = bli_obj_buffer_for_1x1( dt, &alpha_local ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  uploa, \
	  conjx, \
	  m, \
	  buf_alpha, \
	  buf_x, incx, \
	  buf_a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( her )
GENFRONT( syr )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* x, \
       const obj_t* y, \
       const obj_t* a  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t     dt        = bli_obj_dt( a ); \
\
	uplo_t    uploa     = bli_obj_uplo( a ); \
	conj_t    conjx     = bli_obj_conj_status( x ); \
	conj_t    conjy     = bli_obj_conj_status( y ); \
	dim_t     m         = bli_obj_length( a ); \
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
	void*     buf_y     = bli_obj_buffer_at_off( y ); \
	inc_t     incy      = bli_obj_vector_inc( y ); \
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	inc_t     rs_a      = bli_obj_row_stride( a ); \
	inc_t     cs_a      = bli_obj_col_stride( a ); \
\
	void*     buf_alpha; \
\
	obj_t     alpha_local; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, x, y, a ); \
\
	/* Create local copy-casts of scalars (and apply internal conjugation
	   as needed). */ \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      alpha, &alpha_local ); \
	buf_alpha = bli_obj_buffer_for_1x1( dt, &alpha_local ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  uploa, \
	  conjx, \
	  conjy, \
	  m, \
	  buf_alpha, \
	  buf_x, incx, \
	  buf_y, incy, \
	  buf_a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( her2 )
GENFRONT( syr2 )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* alpha, \
       const obj_t* a, \
       const obj_t* x  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t     dt        = bli_obj_dt( a ); \
\
	uplo_t    uploa     = bli_obj_uplo( a ); \
	trans_t   transa    = bli_obj_conjtrans_status( a ); \
	diag_t    diaga     = bli_obj_diag( a ); \
	dim_t     m         = bli_obj_length( a ); \
	void*     buf_a     = bli_obj_buffer_at_off( a ); \
	inc_t     rs_a      = bli_obj_row_stride( a ); \
	inc_t     cs_a      = bli_obj_col_stride( a ); \
	void*     buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t     incx      = bli_obj_vector_inc( x ); \
\
	void*     buf_alpha; \
\
	obj_t     alpha_local; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, x ); \
\
	/* Create local copy-casts of scalars (and apply internal conjugation
	   as needed). */ \
	bli_obj_scalar_init_detached_copy_of( dt, BLIS_NO_CONJUGATE, \
	                                      alpha, &alpha_local ); \
	buf_alpha = bli_obj_buffer_for_1x1( dt, &alpha_local ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  uploa, \
	  transa, \
	  diaga, \
	  m, \
	  buf_alpha, \
	  buf_a, rs_a, cs_a, \
	  buf_x, incx, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( trmv )
GENFRONT( trsv )


#endif

