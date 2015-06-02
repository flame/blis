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

// -- Global variables --

static bool_t bli_initialized = FALSE;

obj_t BLIS_TWO;
obj_t BLIS_ONE;
obj_t BLIS_ONE_HALF;
obj_t BLIS_ZERO;
obj_t BLIS_MINUS_ONE_HALF;
obj_t BLIS_MINUS_ONE;
obj_t BLIS_MINUS_TWO;

packm_thrinfo_t BLIS_PACKM_SINGLE_THREADED;
gemm_thrinfo_t BLIS_GEMM_SINGLE_THREADED;
herk_thrinfo_t BLIS_HERK_SINGLE_THREADED;
thread_comm_t BLIS_SINGLE_COMM;

#ifdef BLIS_ENABLE_PTHREADS
pthread_mutex_t initialize_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

err_t bli_init( void )
{
	err_t r_val = BLIS_FAILURE;

	// If bli_initialized is TRUE, then we know without a doubt that
	// BLIS is presently initialized, and thus we can return early.
	if ( bli_initialized == TRUE ) return r_val;

	// NOTE: if bli_initialized is FALSE, we cannot be certain that BLIS
	// is ready to be initialized; it may be the case that a thread is
	// inside the critical section below and is already in the process
	// of initializing BLIS, but has not yet finished and updated
	// bli_initialized accordingly. This boolean asymmetry is important!

	// We enclose the bodies of bli_init() and bli_finalize() in a
	// critical section (both with the same name) so that they can be
	// safely called from multiple external (application) threads.
	// Note that while the conditional test for early return may reside
	// outside the critical section (as it should, for efficiency
	// reasons), the conditional test below MUST be within the critical
	// section to prevent a race condition of the type described above.

	// BEGIN CRITICAL SECTION
#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (init)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &initialize_mutex );
#endif
	{

	// Proceed with initialization only if BLIS is presently uninitialized.
	// Since we bli_init() and bli_finalize() use the same named critical
	// section, we can be sure that no other thread is either (a) updating
	// bli_initialized, or (b) testing bli_initialized within the critical
	// section (for the purposes of deciding whether to perform the
	// necessary initialization subtasks).
	if ( bli_initialized == FALSE )
	{
		bli_init_const();

		bli_cntl_init();

		bli_error_msgs_init();

		bli_mem_init();

		bli_ind_init();

		bli_setup_communicator( &BLIS_SINGLE_COMM, 1 );
		bli_setup_packm_single_threaded_info( &BLIS_PACKM_SINGLE_THREADED );
		bli_setup_gemm_single_threaded_info( &BLIS_GEMM_SINGLE_THREADED );
		bli_setup_herk_single_threaded_info( &BLIS_HERK_SINGLE_THREADED );

		// After initialization is complete, mark BLIS as initialized.
		bli_initialized = TRUE;

		// Only the thread that actually performs the initialization will
		// return "success".
		r_val = BLIS_SUCCESS;
	}

	// END CRITICAL SECTION
	}
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &initialize_mutex );
#endif

	return r_val;
}

err_t bli_finalize( void )
{
	err_t r_val = BLIS_FAILURE;

	// If bli_initialized is FALSE, then we know without a doubt that
	// BLIS is presently uninitialized, and thus we can return early.
	if ( bli_initialized == FALSE ) return r_val;

	// NOTE: if bli_initialized is TRUE, we cannot be certain that BLIS
	// is ready to be finalized; it may be the case that a thread is
	// inside the critical section below and is already in the process
	// of finalizing BLIS, but has not yet finished and updated
	// bli_initialized accordingly. This boolean asymmetry is important!

	// We enclose the bodies of bli_init() and bli_finalize() in a
	// critical section (both with the same name) so that they can be
	// safely called from multiple external (application) threads.
	// Note that while the conditional test for early return may reside
	// outside the critical section (as it should, for efficiency
	// reasons), the conditional test below MUST be within the critical
	// section to prevent a race condition of the type described above.

	// BEGIN CRITICAL SECTION
#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (init)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &initialize_mutex );
#endif
	{

	// Proceed with finalization only if BLIS is presently initialized.
	// Since we bli_init() and bli_finalize() use the same named critical
	// section, we can be sure that no other thread is either (a) updating
	// bli_initialized, or (b) testing bli_initialized within the critical
	// section (for the purposes of deciding whether to perform the
	// necessary finalization subtasks).
	if ( bli_initialized == TRUE )
	{
		bli_finalize_const();

		bli_cntl_finalize();

		// Don't need to do anything to finalize error messages.

		bli_mem_finalize();

		// After finalization is complete, mark BLIS as uninitialized.
		bli_initialized = FALSE;

		// Only the thread that actually performs the finalization will
		// return "success".
		r_val = BLIS_SUCCESS;
	}

	// END CRITICAL SECTION
	}
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &initialize_mutex );
#endif

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_destroy( &initialize_mutex );
#endif

	return r_val;
}

void bli_init_const( void )
{
	bli_obj_create_const(  2.0, &BLIS_TWO );
	bli_obj_create_const(  1.0, &BLIS_ONE );
	bli_obj_create_const(  0.5, &BLIS_ONE_HALF );
	bli_obj_create_const(  0.0, &BLIS_ZERO );
	bli_obj_create_const( -0.5, &BLIS_MINUS_ONE_HALF );
	bli_obj_create_const( -1.0, &BLIS_MINUS_ONE );
	bli_obj_create_const( -2.0, &BLIS_MINUS_TWO );
}

void bli_finalize_const( void )
{
	bli_obj_free( &BLIS_TWO );
	bli_obj_free( &BLIS_ONE );
	bli_obj_free( &BLIS_ONE_HALF );
	bli_obj_free( &BLIS_ZERO );
	bli_obj_free( &BLIS_MINUS_ONE_HALF );
	bli_obj_free( &BLIS_MINUS_ONE );
	bli_obj_free( &BLIS_MINUS_TWO );
}

void bli_init_auto( err_t* init_result )
{
	*init_result = bli_init();
}

void bli_finalize_auto( err_t init_result )
{
#ifdef BLIS_ENABLE_STAY_AUTO_INITIALIZED

	// If BLIS was configured to stay initialized after being automatically
	// initialized, we honor the configuration request and do nothing.
	// BLIS will remain initialized unless and until the user explicitly
	// calls bli_finalize().

#else

	// If BLIS was NOT configured to stay initialized after being automatically
	// initialized, we call bli_finalize() only if the corresponding call to
	// bli_init_auto() actually resulted in BLIS being initialized (indicated
	// by it returning BLIS_SUCCESS); if it did nothing, we similarly do
	// nothing here.
	if ( init_result == BLIS_SUCCESS )
		bli_finalize();

#endif
}

