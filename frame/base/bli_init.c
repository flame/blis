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

#ifdef BLIS_ENABLE_PTHREADS
static pthread_mutex_t initialize_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static bool_t bli_is_init = FALSE;


err_t bli_init( void )
{
	err_t r_val = BLIS_FAILURE;

	// If bli_is_init is TRUE, then we know without a doubt that
	// BLIS is presently initialized, and thus we can return early.
	if ( bli_is_init == TRUE ) return r_val;

	// NOTE: if bli_is_init is FALSE, we cannot be certain that BLIS
	// is ready to be initialized; it may be the case that a thread is
	// inside the critical section below and is already in the process
	// of initializing BLIS, but has not yet finished and updated
	// bli_is_init accordingly. This boolean asymmetry is important!

	// We enclose the bodies of bli_init() and bli_finalize() in a
	// critical section (both with the same name) so that they can be
	// safely called from multiple external (application) threads.
	// Note that while the conditional test for early return may reside
	// outside the critical section (as it should, for efficiency
	// reasons), the conditional test below MUST be within the critical
	// section to prevent a race condition of the type described above.

#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (init)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &initialize_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{

		// Proceed with initialization only if BLIS is presently uninitialized.
		// Since we bli_init() and bli_finalize() use the same named critical
		// section, we can be sure that no other thread is either (a) updating
		// bli_is_init, or (b) testing bli_is_init within the critical section
		// (for the purposes of deciding whether to perform the necessary
		// initialization subtasks).
		if ( bli_is_init == FALSE )
		{
			// Initialize various sub-APIs.
			bli_const_init();
			bli_error_init();
			bli_memsys_init();
			bli_ind_init();
			bli_thread_init();

			// After initialization is complete, mark BLIS as initialized.
			bli_is_init = TRUE;

			// Only the thread that actually performs the initialization will
			// return "success".
			r_val = BLIS_SUCCESS;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &initialize_mutex );
#endif

	return r_val;
}

err_t bli_finalize( void )
{
	err_t r_val = BLIS_FAILURE;

	// If bli_is_init is FALSE, then we know without a doubt that
	// BLIS is presently uninitialized, and thus we can return early.
	if ( bli_is_init == FALSE ) return r_val;

	// NOTE: if bli_is_init is TRUE, we cannot be certain that BLIS
	// is ready to be finalized; it may be the case that a thread is
	// inside the critical section below and is already in the process
	// of finalizing BLIS, but has not yet finished and updated
	// bli_is_init accordingly. This boolean asymmetry is important!

	// We enclose the bodies of bli_init() and bli_finalize() in a
	// critical section (both with the same name) so that they can be
	// safely called from multiple external (application) threads.
	// Note that while the conditional test for early return may reside
	// outside the critical section (as it should, for efficiency
	// reasons), the conditional test below MUST be within the critical
	// section to prevent a race condition of the type described above.

#ifdef BLIS_ENABLE_OPENMP
	_Pragma( "omp critical (init)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_lock( &initialize_mutex );
#endif

	// BEGIN CRITICAL SECTION
	{

		// Proceed with finalization only if BLIS is presently initialized.
		// Since we bli_init() and bli_finalize() use the same named critical
		// section, we can be sure that no other thread is either (a) updating
		// bli_is_init, or (b) testing bli_is_init within the critical section
		// (for the purposes of deciding whether to perform the necessary
		// finalization subtasks).
		if ( bli_is_init == TRUE )
		{
			// Finalize various sub-APIs.
			bli_const_finalize();
			bli_error_finalize();
			bli_memsys_finalize();
			bli_ind_finalize();
			bli_thread_finalize();

			// After finalization is complete, mark BLIS as uninitialized.
			bli_is_init = FALSE;

			// Only the thread that actually performs the finalization will
			// return "success".
			r_val = BLIS_SUCCESS;
		}
	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_unlock( &initialize_mutex );
#endif

#ifdef BLIS_ENABLE_PTHREADS
	pthread_mutex_destroy( &initialize_mutex );
#endif

	return r_val;
}

bool_t bli_is_initialized( void )
{
	return bli_is_init;
}

// -----------------------------------------------------------------------------

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

