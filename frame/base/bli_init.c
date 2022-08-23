/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

// -----------------------------------------------------------------------------

err_t bli_init( void )
{
	BLIS_INIT_ONCE();

	return BLIS_SUCCESS;
}

err_t bli_finalize( void )
{
	BLIS_FINALIZE_ONCE();

	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

err_t bli_init_auto( void )
{
	// NOTE: Most callers of this function (e.g. the BLAS compatibility layer)
	// will ignore the return value of this function since those functions can't
	// return error codes.
	BLIS_INIT_ONCE();

	return BLIS_SUCCESS;
}

err_t bli_finalize_auto( void )
{
	// The _auto() functions are used when initializing the BLAS compatibility
	// layer. It would not make much sense to automatically initialize and
	// finalize for every BLAS routine call; therefore, we remain initialized
	// unless and until the application explicitly calls bli_finalize().
	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

static bli_pthread_switch_t lib_state = BLIS_PTHREAD_SWITCH_INIT;

err_t bli_init_once( void )
{
	// We can typecast from the return value of bli_pthread_switch_on()
	// (which is of type 'int') directly to 'err_t' since they share the same
	// basic semantics: 0 indicates success while all other values represent
	// some kind of error.
	return ( err_t )bli_pthread_switch_on( &lib_state, bli_init_apis );
}

err_t bli_finalize_once( void )
{
	// We can typecast from the return value of bli_pthread_switch_off()
	// (which is of type 'int') directly to 'err_t' since they share the same
	// basic semantics: 0 indicates success while all other values represent
	// some kind of error.
	return ( err_t )bli_pthread_switch_off( &lib_state, bli_finalize_apis );
}

// -----------------------------------------------------------------------------

int bli_init_apis( void )
{
	err_t r_val = BLIS_SUCCESS;

	// NOTE: Each of the sub-APIs should either (a) fully initialize into a good
	// state (ie: a state in which a subsequent call to the corresponding
	// _finalize() function would fully de-allocate whatever was allocated and
	// thereby avoid a memory leak), or (b) not initialize at all.

	// NOTE: The bli_check_return_if_failure() macro will return r_val when
	// the variable indicates a value indicating failure. Since r_val is
	// declared as of type 'err_t' and the function returns a value of type
	// 'int', an implicit typecast will occur if/when the macro detects failure.

	r_val = bli_gks_init();    bli_check_return_if_failure( r_val );
	r_val = bli_ind_init();    bli_check_return_if_failure( r_val );
	r_val = bli_thread_init(); bli_check_return_if_failure( r_val );
	r_val = bli_pack_init();   bli_check_return_if_failure( r_val );
	r_val = bli_pba_init();    bli_check_return_if_failure( r_val );
	r_val = bli_sba_init();    bli_check_return_if_failure( r_val );

	return ( int )BLIS_SUCCESS;
}

int bli_finalize_apis( void )
{
	err_t r_val = BLIS_SUCCESS;

	// Finalize various sub-APIs.
	r_val = bli_sba_finalize();    bli_check_return_if_failure( r_val );
	r_val = bli_pba_finalize();    bli_check_return_if_failure( r_val );
	r_val = bli_pack_finalize();   bli_check_return_if_failure( r_val );
	r_val = bli_thread_finalize(); bli_check_return_if_failure( r_val );
	r_val = bli_ind_finalize();    bli_check_return_if_failure( r_val );
	r_val = bli_gks_finalize();    bli_check_return_if_failure( r_val );

	return ( int )BLIS_SUCCESS;
}

#if 0
void bli_finalize_apis_fast( void )
{
	// Finalize all APIs but skip the error checking.
	bli_sba_finalize();
	bli_pba_finalize();
	bli_pack_finalize();
	bli_thread_finalize();
	bli_ind_finalize();
	bli_gks_finalize();
}
#endif

