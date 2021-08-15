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
		/* FIXME: BLIS does not yet support induced methods for gemmt. Thus,
		   we call the native implementation code path for now. */
		/*PASTEMAC(opname,ind)( alpha, a, b, beta, c, cntx, rntm );*/
		bli_gemmtnat( alpha, a, b, beta, c, cntx, rntm );
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

void bli_her2k_ex
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
		bli_her2kind( alpha, a, b, beta, c, cntx, rntm );
	}
	else
	{
		bli_her2knat( alpha, a, b, beta, c, cntx, rntm );
	}
}

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

void bli_syr2k_ex
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
		bli_syr2kind( alpha, a, b, beta, c, cntx, rntm );
	}
	else
	{
		bli_syr2knat( alpha, a, b, beta, c, cntx, rntm );
	}
}

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

void bli_herk_ex
     (
       obj_t*  alpha,
       obj_t*  a,
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
	     bli_obj_is_complex( c ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_herkind( alpha, a, beta, c, cntx, rntm );
	}
	else
	{
		bli_herknat( alpha, a, beta, c, cntx, rntm );
	}
}

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

void bli_syrk_ex
     (
       obj_t*  alpha,
       obj_t*  a,
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
	     bli_obj_is_complex( c ) )
	{
		/* Invoke the operation's "ind" function--its induced method front-end.
		   For complex problems, it calls the highest priority induced method
		   that is available (ie: implemented and enabled), and if none are
		   enabled, it calls native execution. (For real problems, it calls
		   the operation's native execution interface.) */
		bli_syrkind( alpha, a, beta, c, cntx, rntm );
	}
	else
	{
		bli_syrknat( alpha, a, beta, c, cntx, rntm );
	}
}

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
