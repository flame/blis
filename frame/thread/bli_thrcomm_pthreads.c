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

#ifdef BLIS_ENABLE_PTHREADS

thrcomm_t* bli_thrcomm_create( rntm_t* rntm, dim_t n_threads )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_thrcomm_create(): " );
	#endif

	thrcomm_t* comm = bli_sba_acquire( rntm, sizeof(thrcomm_t) );

	bli_thrcomm_init( n_threads, comm );

	return comm;
}

void bli_thrcomm_free( rntm_t* rntm, thrcomm_t* comm )
{
	if ( comm == NULL ) return;

	bli_thrcomm_cleanup( comm );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_thrcomm_free(): " );
	#endif

	bli_sba_release( rntm, comm );
}

#ifdef BLIS_USE_PTHREAD_BARRIER

void bli_thrcomm_init( dim_t n_threads, thrcomm_t* comm )
{
	if ( comm == NULL ) return;
	comm->sent_object = NULL;
	comm->n_threads = n_threads;
	bli_pthread_barrier_init( &comm->barrier, NULL, n_threads );
}

void bli_thrcomm_cleanup( thrcomm_t* comm )
{
	if ( comm == NULL ) return;
	bli_pthread_barrier_destroy( &comm->barrier );
}

void bli_thrcomm_barrier( dim_t t_id, thrcomm_t* comm )
{
	bli_pthread_barrier_wait( &comm->barrier );
}

#else

void bli_thrcomm_init( dim_t n_threads, thrcomm_t* comm )
{
	if ( comm == NULL ) return;
	comm->sent_object = NULL;
	comm->n_threads = n_threads;
	comm->barrier_sense = 0;
	comm->barrier_threads_arrived = 0;
}

void bli_thrcomm_cleanup( thrcomm_t* comm )
{
}

void bli_thrcomm_barrier( dim_t t_id, thrcomm_t* comm )
{
#if 0
	if ( comm == NULL || comm->n_threads == 1 ) return;
	bool  my_sense = comm->sense;
	dim_t my_threads_arrived;

	my_threads_arrived = __sync_add_and_fetch(&(comm->threads_arrived), 1);

	if ( my_threads_arrived == comm->n_threads )
	{
		comm->threads_arrived = 0;
		comm->sense = !comm->sense;
	}
	else
	{
		volatile bool* listener = &comm->sense;
		while( *listener == my_sense ) {}
	}
#endif
	bli_thrcomm_barrier_atomic( t_id, comm );
}

#endif

#endif

