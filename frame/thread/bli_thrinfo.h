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

	// The number of distinct threads used to parallelize the loop.
	dim_t              n_way;

	// What we're working on.
	dim_t              work_id;

	// When freeing, should the communicators in this node be freed? Usually,
	// this is field is true, but when nodes are created that share the same
	// communicators as other nodes (such as with packm nodes), this is set
	// to false.
	bool_t             free_comm;

	struct thrinfo_s*  sub_node;
};
typedef struct thrinfo_s thrinfo_t;

//
// thrinfo_t macros
// NOTE: The naming of these should be made consistent at some point.
// (ie: bli_thrinfo_ vs. bli_thread_)
//

// thrinfo_t query (field only)

#define bli_thread_num_threads( t )        ( (t)->ocomm->n_threads )

#define bli_thread_n_way( t )              ( (t)->n_way )
#define bli_thread_work_id( t )            ( (t)->work_id )
#define bli_thread_ocomm_id( t )           ( (t)->ocomm_id )

#define bli_thrinfo_ocomm( t )             ( (t)->ocomm )
#define bli_thrinfo_needs_free_comm( t )   ( (t)->free_comm )

#define bli_thrinfo_sub_node( t )          ( (t)->sub_node )

// thrinfo_t query (complex)

#define bli_thread_am_ochief( t )          ( (t)->ocomm_id == 0 )

// thrinfo_t modification

#define bli_thrinfo_set_sub_node( _sub_node, thread ) \
{ \
	(thread)->sub_node = _sub_node; \
}

// other thrinfo_t-related macros

#define bli_thread_obroadcast( t, p ) bli_thrcomm_bcast( (t)->ocomm, \
                                                         (t)->ocomm_id, p )
#define bli_thread_obarrier( t )      bli_thrcomm_barrier( (t)->ocomm, \
                                                           (t)->ocomm_id )


//
// Prototypes for level-3 thrinfo functions not specific to any operation.
//

thrinfo_t* bli_thrinfo_create
     (
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id, 
       bool_t     free_comm,
       thrinfo_t* sub_node
     );

void bli_thrinfo_init
     (
       thrinfo_t* thread,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id, 
       bool_t     free_comm,
       thrinfo_t* sub_node
     );

void bli_thrinfo_init_single
     (
       thrinfo_t* thread
     );

// -----------------------------------------------------------------------------

thrinfo_t* bli_thrinfo_create_for_cntl
     (
       cntx_t*    cntx,
       cntl_t*    cntl_par,
       cntl_t*    cntl_chl,
       thrinfo_t* thread_par
     );

void bli_thrinfo_grow
     (
       cntx_t*    cntx,
       cntl_t*    cntl,
       thrinfo_t* thread
     );

thrinfo_t* bli_thrinfo_rgrow
     (
       cntx_t*    cntx,
       cntl_t*    cntl_par,
       cntl_t*    cntl_cur,
       thrinfo_t* thread_par
     );

#endif
