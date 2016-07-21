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

#ifndef BLIS_THRINFO_H
#define BLIS_THRINFO_H

// Thread info structure definition
struct thrinfo_s
{
	// The thread communicator for the other threads sharing the same work
	// at this level.
	thrcomm_t*         ocomm;

	// Our thread id within the ocomm thread communicator.
	dim_t              ocomm_id;

	// The thread communicator for the other threads sharing the same work
	// at this level.
	thrcomm_t*         icomm;

	// Our thread id within the icomm thread communicator.
	dim_t              icomm_id;

	// The number of distinct threads used to parallelize the loop.
	dim_t              n_way;

	// What we're working on.
	dim_t              work_id;

	struct thrinfo_s*  opackm;
	struct thrinfo_s*  ipackm;
	struct thrinfo_s*  sub_self;
};
typedef struct thrinfo_s thrinfo_t;


#define bli_thread_num_threads( t )     ( t->ocomm->n_threads )

#define bli_thread_n_way( t )           ( t->n_way )
#define bli_thread_work_id( t )         ( t->work_id )
#define bli_thread_am_ochief( t )       ( t->ocomm_id == 0 )
#define bli_thread_am_ichief( t )       ( t->icomm_id == 0 )

#define bli_thread_obroadcast( t, ptr ) bli_thrcomm_bcast( t->ocomm, t->ocomm_id, ptr )
#define bli_thread_ibroadcast( t, ptr ) bli_thrcomm_bcast( t->icomm, t->icomm_id, ptr )
#define bli_thread_obarrier( t )        bli_thrcomm_barrier( t->ocomm, t->ocomm_id )
#define bli_thread_ibarrier( t )        bli_thrcomm_barrier( t->icomm, t->icomm_id )

//
// Generic accessor macros for all thrinfo_t objects.
//

#define bli_thrinfo_sub_opackm( t )     ( t->opackm )
#define bli_thrinfo_sub_ipackm( t )     ( t->ipackm )
#define bli_thrinfo_sub_self( t )       ( t->sub_self )

//
// Prototypes for level-3 thrinfo functions not specific to any operation.
//

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
     );

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
     );

void bli_thrinfo_init_single
     (
       thrinfo_t* thread
     );

void bli_thrinfo_free
     (
       thrinfo_t* thread
     );

#endif
