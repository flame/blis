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

extern "C" {

#ifdef BLIS_USE_HPX_BARRIER

// Define the pthread_barrier_t implementations of the init, cleanup, and
// barrier functions.

void bli_thrcomm_init_hpx( dim_t n_threads, thrcomm_t* comm )
{
	if ( comm == nullptr ) return;
	comm->barrier = new hpx:barrier<>();
}

void bli_thrcomm_cleanup_hpx( thrcomm_t* comm )
{
	if ( comm == nullptr ) return;
	delete comm->barrier;
}

void bli_thrcomm_barrier( dim_t t_id, thrcomm_t* comm )
{
	comm->barrier->arrive_and_wait();
}

#else

// Define the non-hpx::barrier implementations of the init, cleanup,
// and barrier functions. These are the default unless the hpx::barrier
// versions are requested at compile-time.

void bli_thrcomm_init_hpx( dim_t n_threads, thrcomm_t* comm )
{
	if ( comm == nullptr ) return;
	comm->sent_object = nullptr;
	comm->n_threads = n_threads;
	comm->barrier_sense = 0;
	comm->barrier_threads_arrived = 0;
}

void bli_thrcomm_cleanup_hpx( thrcomm_t* comm )
{
}

void bli_thrcomm_barrier_hpx( dim_t t_id, thrcomm_t* comm )
{
	bli_thrcomm_barrier_atomic( t_id, comm );
}

} // extern "C"

#endif

#endif

