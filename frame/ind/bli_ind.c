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

static bool_t bli_ind_is_init = FALSE;

static char* bli_ind_impl_str[BLIS_NUM_IND_METHODS] =
{
/* 3mh  */ "3mh",
/* 3m2  */ "3m3",
/* 3m2  */ "3m2",
/* 3m1  */ "3m1",
/* 4mh  */ "4mh",
/* 4m1b */ "4m1b",
/* 4m1a */ "4m1a",
/* nat  */ "native",
};

// -----------------------------------------------------------------------------

void bli_ind_init( void )
{
	// If the API is already initialized, return early.
	if ( bli_ind_is_initialized() ) return;

#ifdef BLIS_ENABLE_INDUCED_SCOMPLEX
	bli_ind_enable_dt( BLIS_4M1A, BLIS_SCOMPLEX );
#endif
#ifdef BLIS_ENABLE_INDUCED_DCOMPLEX
	bli_ind_enable_dt( BLIS_4M1A, BLIS_DCOMPLEX );
#endif

	// Mark API as initialized.
	bli_ind_is_init = TRUE;
}

void bli_ind_finalize( void )
{
	// Mark API as uninitialized.
	bli_ind_is_init = FALSE;
}

bool_t bli_ind_is_initialized( void )
{
	return bli_ind_is_init;
}

// -----------------------------------------------------------------------------

void bli_ind_enable( ind_t method )
{
	bli_ind_enable_dt( method, BLIS_SCOMPLEX );
	bli_ind_enable_dt( method, BLIS_DCOMPLEX );
}

void bli_ind_disable( ind_t method )
{
	bli_ind_disable_dt( method, BLIS_SCOMPLEX );
	bli_ind_disable_dt( method, BLIS_DCOMPLEX );
}

void bli_ind_disable_all( void )
{
	bli_ind_disable_all_dt( BLIS_SCOMPLEX );
	bli_ind_disable_all_dt( BLIS_DCOMPLEX );
}

// -----------------------------------------------------------------------------

void bli_ind_enable_dt( ind_t method, num_t dt )
{
	if ( !bli_is_complex( dt ) ) return;

	bli_l3_ind_set_enable_dt( method, dt, TRUE );
}

void bli_ind_disable_dt( ind_t method, num_t dt )
{
	if ( !bli_is_complex( dt ) ) return;

	bli_l3_ind_set_enable_dt( method, dt, FALSE );
}

void bli_ind_disable_all_dt( num_t dt )
{
	ind_t im;

	for ( im = 0; im < BLIS_NUM_IND_METHODS; ++im )
	{
		// Never disable native execution.
		if ( im != BLIS_NAT )
			bli_ind_disable_dt( im, dt );
	}
}

// -----------------------------------------------------------------------------

void bli_ind_oper_enable_only( opid_t oper, ind_t method, num_t dt )
{
	if ( !bli_is_complex( dt ) ) return;

	if ( bli_opid_is_level3( oper ) )
	{
		bli_l3_ind_oper_enable_only( oper, method, dt );
	}
	else
	{
		// Other operations are not implemented, so requests to enable
		// them for any given induced method are currently no-ops.
		;
	}
}

// -----------------------------------------------------------------------------

bool_t bli_ind_oper_is_impl( opid_t oper, ind_t method )
{
	bool_t is_impl = FALSE;

	if ( bli_opid_is_level3( oper ) )
	{
		// Look up whether its func_t pointer in the table is NULL.
		is_impl = ( bli_l3_ind_oper_get_func( oper, method ) != NULL );
	}
	else
	{
		// All other operations should be reported as not implemented,
		// unless the requested check was for BLIS_NAT, in which case
		// all operations are implemented.
	    if ( method == BLIS_NAT ) is_impl = TRUE;
	    else                      is_impl = FALSE;
	}

	return is_impl;
}

bool_t bli_ind_oper_has_avail( opid_t oper, num_t dt )
{
	ind_t method = bli_ind_oper_find_avail( oper, dt );

	if ( method == BLIS_NAT ) return FALSE;
	else                      return TRUE;
}

void* bli_ind_oper_get_avail( opid_t oper, num_t dt )
{
	ind_t method = bli_ind_oper_find_avail( oper, dt );
	void* func_p;

	if ( bli_opid_is_level3( oper ) )
	{
		func_p = bli_l3_ind_oper_get_func( oper, method );
	}
	else
	{
		// Currently, any operation that is not level-3 does not
		// have induced method implementations. (This should actually	
		// assign the pointer to be the native front-end, but for
		// now there are no calls to bli_ind_oper_get_avail() in the
		// context of level-2 operations.
		func_p = NULL;
	}

	return func_p;
}

ind_t bli_ind_oper_find_avail( opid_t oper, num_t dt )
{
	ind_t method;

	if ( bli_opid_is_level3( oper ) )
	{
		method = bli_l3_ind_oper_find_avail( oper, dt );
	}
	else
	{
		// Currently, any operation that is not level-3 is guaranteed
		// to be native.
		method = BLIS_NAT;
	}

	return method;
}

char* bli_ind_oper_get_avail_impl_string( opid_t oper, num_t dt )
{
	ind_t method = bli_ind_oper_find_avail( oper, dt );

	return bli_ind_get_impl_string( method );
}

// -----------------------------------------------------------------------------

char* bli_ind_get_impl_string( ind_t method )
{
	return bli_ind_impl_str[ method ];
}

num_t bli_ind_map_cdt_to_index( num_t dt )
{
	// A non-complex datatype should never be passed in.
	if ( !bli_is_complex( dt ) ) bli_abort();

	// Map the complex datatype to a zero-based index.
	if         ( bli_is_scomplex( dt ) )    return 0;
	else /* if ( bli_is_dcomplex( dt ) ) */ return 1;
}

