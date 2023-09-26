/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

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

#include <hpx/execution.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>

extern "C"
{

void bli_thread_launch_hpx
     (
             dim_t         n_threads,
             thread_func_t func,
       const void*         params
     )
{
	const timpl_t ti = BLIS_HPX;

	// Allocate a global communicator for the root thrinfo_t structures.
	pool_t*    gl_comm_pool = nullptr;
	thrcomm_t* gl_comm      = bli_thrcomm_create( ti, gl_comm_pool, n_threads );
	hpx::threads::run_as_hpx_thread([&]()
	{
		hpx::execution::experimental::num_cores num_cores_(n_threads);
		hpx::execution::static_chunk_size chunk_size_(1);
		hpx::experimental::for_loop(
		hpx::execution::par.with(num_cores_).with(chunk_size_), 0, n_threads,
		[&gl_comm, &func, &params](const dim_t tid)
		{
			func( gl_comm, tid, params );
		});
	});

	// Free the global communicator, because the root thrinfo_t node
	// never frees its communicator.
	bli_thrcomm_free( gl_comm_pool, gl_comm );
}

void bli_thread_initialize_hpx( int argc, char** argv )
{
	hpx::start( nullptr, argc, argv );
}

int bli_thread_finalize_hpx()
{
	hpx::post([]() { hpx::finalize(); });
	return hpx::stop();
}

} // extern "C"

#endif
