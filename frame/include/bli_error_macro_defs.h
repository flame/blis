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

#ifndef BLIS_ERROR_MACRO_DEFS_H
#define BLIS_ERROR_MACRO_DEFS_H

// Used to insert filenames and line numbers into error-checking code.
#define bli_check_error_code( code ) \
        bli_check_error_code_helper( code, __FILE__, __LINE__ )


// TODO: Consider renaming this macro to one of:
// - bli_error_handle()
// - bli_error_handle_code()
// Also, consider replacing instances of
//   if ( bli_is_failure( r_val ) ) return r_val;
// to a macro named something like:
// - bli_check_return_failure( r_val );
// Also, consider adding some of logic from bli_check_error_code_helper() to
// 'else' branch of bli_check_return_error_code() so that we can intercept
// and handle undefined error codes?

#define bli_check_return_error_code( code ) \
{ \
	if ( bli_is_failure( code ) ) \
	{ \
		if ( bli_error_mode_is_return() ) \
		{ \
			return code; \
		} \
		else /* if ( bli_error_mode_is_abort() ) */ \
		{ \
			bli_print_msg( bli_error_string_for_code( code ), \
						   __FILE__, __LINE__ ); \
			bli_abort(); \
		} \
	} \
}

#define bli_check_threads_return_if_failure( e_val_p, thread ) \
{ \
	/* Broadcast the address of the master thread's copy of e_val. */ \
	err_t* e_val_t0_p = bli_thread_broadcast( thread, e_val_p ); \
\
	/* If the local error checking resulted in failure, save it to the master
	   thread's e_val. Note this includes master overwriting its own e_val. */ \
	if ( bli_is_failure( *(e_val_p) ) ) *e_val_t0_p = *(e_val_p); \
\
	/* Wait for all theads to execute the previous code. */ \
	bli_thread_barrier( thread ); \
\
	/* If any thread reported failure, everyone returns. All threads
	   return their local error code. */ \
	if ( bli_is_failure( *e_val_t0_p ) ) return *e_val_p; \
}

#define bli_check_thread0_return_if_failure( e_val_p, thread ) \
{ \
	/* Broadcast the address of the master thread's copy of e_val. */ \
	err_t* e_val_t0_p = bli_thread_broadcast( thread, e_val_p ); \
\
	/* If the master thread reported failure, everyone returns. All threads
	   return their local error code. */ \
	if ( bli_is_failure( *e_val_t0_p ) ) return *e_val_p; \
}

#define bli_check_return_if_failure( error_code ) \
{ \
	if ( bli_is_failure( error_code ) ) return error_code; \
}

#define bli_check_callthen_return_if_failure( func, error_code ) \
{ \
	/* Note that the 'func' token will be a function call, including its
	   parenthesized parameter list (even if it is empty). */ \
	if ( bli_is_failure( error_code ) ) { func; return error_code; } \
}

#define bli_check_return_other_if_failure( error_code, other_val ) \
{ \
	if ( bli_is_failure( error_code ) ) return other_val; \
}

#define BLIS_INIT_ONCE() \
{ \
	err_t r_val = bli_init_once(); \
	bli_check_return_if_failure( r_val ); \
}

#define BLIS_FINALIZE_ONCE() \
{ \
	err_t r_val = bli_finalize_once(); \
	bli_check_return_if_failure( r_val ); \
}

#endif

