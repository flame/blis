/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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

#include "blis.h"

//
// Define object-based interfaces.
//

void bli_gemm_ex
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	/* If the rntm is non-NULL, it may indicate that we should forgo sup
	   handling altogether. */
	bool enable_sup = TRUE;
	if ( rntm != NULL ) enable_sup = bli_rntm_l3_sup( rntm );

	if ( enable_sup )
	{
		/* Execute the small/unpacked oapi handler. If it finds that the problem
		   does not fall within the thresholds that define "small", or for some
		   other reason decides not to use the small/unpacked implementation,
		   the function returns with BLIS_FAILURE, which causes execution to
		   proceed towards the conventional implementation. */
		err_t result = bli_gemmsup( alpha, a, b, beta, c, cntx, rntm );
		if ( result == BLIS_SUCCESS )
		{
			return;
		}
	}

	/* Only proceed with an induced method if each of the operands have a
	   complex storage datatype. NOTE: Allowing precisions to vary while
	   using 1m, which is what we do here, is unique to gemm; other level-3
	   operations use 1m only if all storage datatypes are equal (and they
	   ignore the computation precision). If any operands are real, skip the
	   induced method chooser function and proceed directly with native
	   execution. */
	if ( bli_obj_is_complex( c ) &&
	     bli_obj_is_complex( a ) &&
	     bli_obj_is_complex( b ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_gemmind( alpha, a, b, beta, c, cntx, rntm );
	}
	else
	{
		bli_gemmnat( alpha, a, b, beta, c, cntx, rntm );
	}
}

void bli_gemm
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_gemm_ex( alpha, a, b, beta, c, NULL, NULL );
}

void bli_gemmt_ex
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	/* If the rntm is non-NULL, it may indicate that we should forgo sup
	   handling altogether. */
	/*
	bool enable_sup = TRUE;
	if ( rntm != NULL ) enable_sup = bli_rntm_l3_sup( rntm );
	*/

	/* NOTE: The sup handling for gemmt is disabled here because gemmtsup
	   is not yet fully implemented. */
	/*
	if ( enable_sup )
	{
	*/
		/* Execute the small/unpacked oapi handler. If it finds that the problem
		   does not fall within the thresholds that define "small", or for some
		   other reason decides not to use the small/unpacked implementation,
		   the function returns with BLIS_FAILURE, which causes execution to
		   proceed towards the conventional implementation. */
	/*
		err_t result = PASTEMAC(opname,sup)( alpha, a, b, beta, c, cntx, rntm );
		if ( result == BLIS_SUCCESS )
		{
			return;
		}
	}
	*/

	/* Only proceed with an induced method if all operands have the same
	   (complex) datatype. If any datatypes differ, skip the induced method
	   chooser function and proceed directly with native execution, which is
	   where mixed datatype support will be implemented (if at all). */
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		bli_gemmtind( alpha, a, b, beta, c, cntx, rntm );
	}
	else
	{
		bli_gemmtnat( alpha, a, b, beta, c, cntx, rntm );
	}
}

void bli_gemmt
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
    bli_gemmt_ex( alpha, a, b, beta, c, NULL, NULL );
}

#undef GENTFUNC
#define GENTFUNC(opname,ind) \
void PASTEMAC(opname,ind) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm \
     ) \
{ \
    bli_init_once(); \
\
    obj_t ah; \
    obj_t bh; \
    obj_t alphah; \
\
	/* Check parameters. */ \
	if ( bli_error_checking_is_enabled() ) \
		bli_her2k_check( alpha, a, b, beta, c, cntx ); \
\
	bli_obj_alias_to( alpha, &alphah ); \
	bli_obj_toggle_conj( &alphah ); \
\
	bli_obj_alias_to( a, &ah ); \
	bli_obj_induce_trans( &ah ); \
	bli_obj_toggle_conj( &ah ); \
\
	bli_obj_alias_to( b, &bh ); \
	bli_obj_induce_trans( &bh ); \
	bli_obj_toggle_conj( &bh ); \
\
	/* Invoke gemmt twice, using beta only the first time. */ \
\
    PASTEMAC(gemmt,ind)(   alpha, a, &bh,      beta, c, cntx, rntm ); \
    PASTEMAC(gemmt,ind)( &alphah, b, &ah, &BLIS_ONE, c, cntx, rntm ); \
\
	/* The Hermitian rank-2k product was computed as A*B'+B*A', even for \
	 * the diagonal elements. Mathematically, the imaginary components of \
	 * diagonal elements of a Hermitian rank-2k product should always be \
	 * zero. However, in practice, they sometimes accumulate meaningless \
	 * non-zero values. To prevent this, we explicitly set those values \
	 * to zero before returning. */ \
 \
    bli_setid( &BLIS_ZERO, c ); \
}

GENTFUNC(her2k,_ex);
GENTFUNC(her2k,3mh);
GENTFUNC(her2k,3m1);
GENTFUNC(her2k,4mh);
GENTFUNC(her2k,4m1);
GENTFUNC(her2k,1m);
GENTFUNC(her2k,nat);
GENTFUNC(her2k,ind);

void bli_her2k
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_her2k_ex( alpha, a, b, beta, c, NULL, NULL );
}

#undef GENTFUNC
#define GENTFUNC(opname,ind) \
void PASTEMAC(opname,ind) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm \
     ) \
{ \
    bli_init_once(); \
\
    obj_t at; \
    obj_t bt; \
\
	/* Check parameters. */ \
	if ( bli_error_checking_is_enabled() ) \
		bli_syr2k_check( alpha, a, b, beta, c, cntx ); \
\
	bli_obj_alias_to( b, &bt ); \
	bli_obj_induce_trans( &bt ); \
	bli_obj_alias_to( a, &at ); \
	bli_obj_induce_trans( &at ); \
\
	/* Invoke gemmt twice, using beta only the first time. */ \
\
    PASTEMAC(gemmt,ind)( alpha, a, &bt,      beta, c, cntx, rntm ); \
    PASTEMAC(gemmt,ind)( alpha, b, &at, &BLIS_ONE, c, cntx, rntm ); \
}

GENTFUNC(syr2k,_ex);
GENTFUNC(syr2k,3mh);
GENTFUNC(syr2k,3m1);
GENTFUNC(syr2k,4mh);
GENTFUNC(syr2k,4m1);
GENTFUNC(syr2k,1m);
GENTFUNC(syr2k,nat);
GENTFUNC(syr2k,ind);

void bli_syr2k
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_syr2k_ex( alpha, a, b, beta, c, NULL, NULL );
}

void bli_hemm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	/* Only proceed with an induced method if all operands have the same
	   (complex) datatype. If any datatypes differ, skip the induced method
	   chooser function and proceed directly with native execution, which is
	   where mixed datatype support will be implemented (if at all). */
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_hemmind( side, alpha, a, b, beta, c, cntx, rntm );
	}
	else
	{
		bli_hemmnat( side, alpha, a, b, beta, c, cntx, rntm );
	}
}

void bli_hemm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_hemm_ex( side, alpha, a, b, beta, c, NULL, NULL );
}

void bli_symm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	/* Only proceed with an induced method if all operands have the same
	   (complex) datatype. If any datatypes differ, skip the induced method
	   chooser function and proceed directly with native execution, which is
	   where mixed datatype support will be implemented (if at all). */
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_symmind( side, alpha, a, b, beta, c, cntx, rntm );
	}
	else
	{
		bli_symmnat( side, alpha, a, b, beta, c, cntx, rntm );
	}
}

void bli_symm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_symm_ex( side, alpha, a, b, beta, c, NULL, NULL );
}

void bli_trmm3_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	/* Only proceed with an induced method if all operands have the same
	   (complex) datatype. If any datatypes differ, skip the induced method
	   chooser function and proceed directly with native execution, which is
	   where mixed datatype support will be implemented (if at all). */
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_trmm3ind( side, alpha, a, b, beta, c, cntx, rntm );
	}
	else
	{
		bli_trmm3nat( side, alpha, a, b, beta, c, cntx, rntm );
	}
}

void bli_trmm3
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_trmm3_ex( side, alpha, a, b, beta, c, NULL, NULL );
}

#undef GENTFUNC
#define GENTFUNC(opname,ind) \
void PASTEMAC(opname,ind) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm \
     ) \
{ \
    bli_init_once(); \
\
    obj_t ah; \
\
	/* Check parameters. */ \
	if ( bli_error_checking_is_enabled() ) \
		bli_herk_check( alpha, a, beta, c, cntx ); \
\
	bli_obj_alias_to( a, &ah ); \
	bli_obj_induce_trans( &ah ); \
    bli_obj_toggle_conj( &ah ); \
\
    PASTEMAC(gemmt,ind)( alpha, a, &ah, beta, c, cntx, rntm ); \
\
	/* The Hermitian rank-k product was computed as A*A', even for the \
	 * diagonal elements. Mathematically, the imaginary components of \
	 * diagonal elements of a Hermitian rank-k product should always be \
	 * zero. However, in practice, they sometimes accumulate meaningless \
	 * non-zero values. To prevent this, we explicitly set those values \
	 * to zero before returning. */ \
\
	bli_setid( &BLIS_ZERO, c ); \
}

GENTFUNC(herk,_ex);
GENTFUNC(herk,3mh);
GENTFUNC(herk,3m1);
GENTFUNC(herk,4mh);
GENTFUNC(herk,4m1);
GENTFUNC(herk,1m);
GENTFUNC(herk,nat);
GENTFUNC(herk,ind);

void bli_herk
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_herk_ex( alpha, a, beta, c, NULL, NULL );
}

#undef GENTFUNC
#define GENTFUNC(opname,ind) \
void PASTEMAC(opname,ind) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm \
     ) \
{ \
    bli_init_once(); \
\
    obj_t at; \
\
	/* Check parameters. */ \
	if ( bli_error_checking_is_enabled() ) \
		bli_syrk_check( alpha, a, beta, c, cntx ); \
\
	bli_obj_alias_to( a, &at ); \
	bli_obj_induce_trans( &at ); \
\
    PASTEMAC(gemmt,ind)( alpha, a, &at, beta, c, cntx, rntm ); \
}

GENTFUNC(syrk,_ex);
GENTFUNC(syrk,3mh);
GENTFUNC(syrk,3m1);
GENTFUNC(syrk,4mh);
GENTFUNC(syrk,4m1);
GENTFUNC(syrk,1m);
GENTFUNC(syrk,nat);
GENTFUNC(syrk,ind);

void bli_syrk
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  beta,
       obj_t*  c
     )
{
	bli_syrk_ex( alpha, a, beta, c, NULL, NULL );
}

void bli_trmm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	/* Only proceed with an induced method if all operands have the same
	   (complex) datatype. If any datatypes differ, skip the induced method
	   chooser function and proceed directly with native execution, which is
	   where mixed datatype support will be implemented (if at all). */
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_is_complex( b ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_trmmind( side, alpha, a, b, cntx, rntm );
	}
	else
	{
		bli_trmmnat( side, alpha, a, b, cntx, rntm );
	}
}

void bli_trmm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b
     )
{
	bli_trmm_ex( side, alpha, a, b, NULL, NULL );
}

void bli_trsm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	/* Only proceed with an induced method if all operands have the same
	   (complex) datatype. If any datatypes differ, skip the induced method
	   chooser function and proceed directly with native execution, which is
	   where mixed datatype support will be implemented (if at all). */
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_is_complex( b ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_trsmind( side, alpha, a, b, cntx, rntm );
	}
	else
	{
		bli_trsmnat( side, alpha, a, b, cntx, rntm );
	}
}

void bli_trsm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b
     )
{
	bli_trsm_ex( side, alpha, a, b, NULL, NULL );
}
