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

#ifndef BLIS_ENABLE_MULTITHREADING

void bli_l3_thread_decorator
     (
       dim_t    n_threads,
       l3_int_t func,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       obj_t*   beta,
       obj_t*   c,
       void*    cntx,
       void*    cntl,
       void**   thread
     )
{
	func( alpha, a, b, beta, c, cntx, cntl, thread[0] );
}


//Constructors and destructors for constructors
thrcomm_t* bli_thrcomm_create( dim_t n_threads )
{
	thrcomm_t* comm = bli_malloc_intl( sizeof( thrcomm_t ) );
	bli_thrcomm_init( comm, n_threads );
	return comm;
}

void bli_thrcomm_free( thrcomm_t* communicator )
{
	if ( communicator == NULL ) return;
	bli_thrcomm_cleanup( communicator );
	bli_free_intl( communicator );
}

void bli_thrcomm_init( thrcomm_t* communicator, dim_t n_threads )
{
	if ( communicator == NULL ) return;

	communicator->sent_object             = NULL;
	communicator->n_threads               = n_threads;
	communicator->barrier_sense           = 0;
	communicator->barrier_threads_arrived = 0;
}

void bli_thrcomm_cleanup( thrcomm_t* communicator )
{
	if ( communicator == NULL ) return;
}

void bli_thrcomm_barrier( thrcomm_t* communicator, dim_t t_id )
{
	return;
}

#endif

