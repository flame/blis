/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022 Tactical Computing Laboratories, LLC

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

#ifdef BLIS_ENABLE_HPX

#include <hpx/local/execution.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>

// A data structure to assist in passing operands to additional threads.
typedef struct thread_data
{
	      dim_t         tid;
	      thrcomm_t*    gl_comm;
	      thread_func_t func;
	const void*         params;
} thread_data_t;

// Entry point for additional threads
static void* bli_hpx_thread_entry( void* data_void )
{
	const thread_data_t* data     = static_cast<thread_data_t*>(data_void);

	const dim_t          tid      = data->tid;
	      thrcomm_t*     gl_comm  = data->gl_comm;
	      thread_func_t  func     = data->func;
	const void*          params   = data->params;

	// Call the thread entry point, passing the global communicator, the
	// thread id, and the params struct as arguments.
	func( gl_comm, tid, params );

	return NULL;
}

#ifdef __cplusplus
extern "C" {
#endif

void bli_thread_launch_hpx( dim_t n_threads, thread_func_t func, const void* params )
{
	err_t r_val;

	const timpl_t ti = BLIS_HPX;

	// Allocate a global communicator for the root thrinfo_t structures.
	pool_t*    gl_comm_pool = NULL;
	thrcomm_t* gl_comm      = bli_thrcomm_create( ti, gl_comm_pool, n_threads );

	// Allocate an array of pthread objects and auxiliary data structs to pass
	// to the thread entry functions.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif
	thread_data_t* datas    = static_cast<thread_data_t*>(bli_malloc_intl( sizeof( thread_data_t ) * n_threads, &r_val ));

	// NOTE: We must iterate backwards so that the chief thread (thread id 0)
	// can spawn all other threads before proceeding with its own computation.
	auto irange = hpx::util::detail::make_counting_shape(n_threads);

        hpx::for_each(hpx::execution::par, hpx::util::begin(irange), hpx::util::end(irange),
	[&datas, &gl_comm, &func, &params](const dim_t tid) {
		// Set up thread data for additional threads (beyond thread 0).
		datas[tid].tid      = tid;
		datas[tid].gl_comm  = gl_comm;
		datas[tid].func     = func;
		datas[tid].params   = params;

		bli_hpx_thread_entry(&datas[0]);
	});

	// Free the global communicator, because the root thrinfo_t node
	// never frees its communicator.
	bli_thrcomm_free( gl_comm_pool, gl_comm );

	// Free the array of pthread objects and auxiliary data structs.
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif
	bli_free_intl( datas );
}

#ifdef __cplusplus
} // end extern "C"
#endif

#endif
