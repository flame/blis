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

#include "blis2.h"

// -- Global variables --

static bool_t bl2_initialized = FALSE;

obj_t BLIS_TWO;
obj_t BLIS_ONE;
obj_t BLIS_ONE_HALF;
obj_t BLIS_ZERO;
obj_t BLIS_MINUS_ONE_HALF;
obj_t BLIS_MINUS_ONE;
obj_t BLIS_MINUS_TWO;



void bl2_init( void )
{
	bl2_initialized = TRUE;

	bl2_init_const();

	bl2_cntl_init();

	bl2_error_msgs_init();
}

void bl2_finalize( void )
{
	bl2_initialized = FALSE;

	bl2_finalize_const();

	bl2_cntl_finalize();

	// Don't need to do anything to finalize error messages.
}

void bl2_init_const( void )
{
	bl2_obj_create_const(  2.0, &BLIS_TWO );
	bl2_obj_create_const(  1.0, &BLIS_ONE );
	bl2_obj_create_const(  0.5, &BLIS_ONE_HALF );
	bl2_obj_create_const(  0.0, &BLIS_ZERO );
	bl2_obj_create_const( -0.5, &BLIS_MINUS_ONE_HALF );
	bl2_obj_create_const( -1.0, &BLIS_MINUS_ONE );
	bl2_obj_create_const( -2.0, &BLIS_MINUS_TWO );
}

void bl2_finalize_const( void )
{
	bl2_obj_free( &BLIS_TWO );
	bl2_obj_free( &BLIS_ONE );
	bl2_obj_free( &BLIS_ONE_HALF );
	bl2_obj_free( &BLIS_ZERO );
	bl2_obj_free( &BLIS_MINUS_ONE_HALF );
	bl2_obj_free( &BLIS_MINUS_ONE );
	bl2_obj_free( &BLIS_MINUS_TWO );
}

void bl2_init_safe( err_t* init_result )
{
	if ( bl2_initialized )
	{
		*init_result = BLIS_FAILURE;
	}
	else
	{
		bl2_init();
		*init_result = BLIS_SUCCESS;
	}
}

void bl2_finalize_safe( err_t init_result )
{
#ifdef BLIS_ENABLE_STAY_AUTO_INITIALIZED

	// If BLIS was configured to stay initialized after being automatically
	// initialized, we honor the configuration request and do nothing.
	// BLIS will remain initialized unless and until the user explicitly
	// calls bl2_finalize().

#else
	// Only finalize if the corresponding bl2_init_safe() actually
	// resulted in BLIS being initialized; if it did nothing, we
	// similarly do nothing here.
	if ( init_result == BLIS_SUCCESS )
		bl2_finalize();
#endif
}

