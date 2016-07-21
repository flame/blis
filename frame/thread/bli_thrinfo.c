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

thrinfo_t* bli_thrinfo_create
     (
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       thrcomm_t* icomm,
       dim_t      icomm_id,
       dim_t      n_way,
       dim_t      work_id, 
       thrinfo_t* opackm,
       thrinfo_t* ipackm,
       thrinfo_t* sub_self
     )
{
    thrinfo_t* thread = bli_malloc_intl( sizeof( thrinfo_t ) );

    bli_thrinfo_init
	(
	  thread,
	  ocomm, ocomm_id,
	  icomm, icomm_id,
	  n_way, work_id, 
	  opackm,
	  ipackm,
	  sub_self
	);

    return thread;
}

void bli_thrinfo_init
     (
       thrinfo_t* thread,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       thrcomm_t* icomm,
       dim_t      icomm_id,
       dim_t      n_way,
       dim_t      work_id, 
       thrinfo_t* opackm,
       thrinfo_t* ipackm,
       thrinfo_t* sub_self
     )
{
	thread->ocomm    = ocomm;
	thread->ocomm_id = ocomm_id;
	thread->icomm    = icomm;
	thread->icomm_id = icomm_id;
	thread->n_way    = n_way;
	thread->work_id  = work_id;

	thread->opackm   = opackm;
	thread->ipackm   = ipackm;
	thread->sub_self = sub_self;
}

void bli_thrinfo_init_single
     (
       thrinfo_t* thread
     )
{
	bli_thrinfo_init
	(
	  thread,
	  &BLIS_SINGLE_COMM, 0,
	  &BLIS_SINGLE_COMM, 0,
	  1,
	  0,
	  &BLIS_PACKM_SINGLE_THREADED,
	  &BLIS_PACKM_SINGLE_THREADED,
	  thread
	);
}

#if 0
void bli_thrinfo_free
     (
       thrinfo_t* thread
     )
{
	if ( thread == NULL ||
	     thread == &BLIS_GEMM_SINGLE_THREADED ||
	     thread == &BLIS_HERK_SINGLE_THREADED ||
	     thread == &BLIS_PACKM_SINGLE_THREADED
	   ) return;

	// Free Communicators
	if ( bli_thread_am_ochief( thread ) )
		bli_thrcomm_free( thread->ocomm );
	if ( bli_thrinfo_sub_self( thread ) == NULL && bli_thread_am_ichief( thread ) )
		bli_thrcomm_free( thread->icomm );

	// Free thrinfo chidren
	bli_packm_thrinfo_free( thread->opackm );
	bli_packm_thrinfo_free( thread->ipackm );
	bli_l3_thrinfo_free( thread->sub_self );
	bli_free_intl( thread );
	
	return; 
}
#endif

