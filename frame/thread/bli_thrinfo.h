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

	// The bszid_t to help identify the node. This is mostly only useful when
	// debugging or tracing the allocation and release of thrinfo_t nodes.
	bszid_t            bszid;

	struct thrinfo_s*  sub_prenode;
	struct thrinfo_s*  sub_node;
};
typedef struct thrinfo_s thrinfo_t;

//
// thrinfo_t functions
// NOTE: The naming of these should be made consistent at some point.
// (ie: bli_thrinfo_ vs. bli_thread_)
//

// thrinfo_t query (field only)

static dim_t bli_thread_num_threads( thrinfo_t* t )
{
	return (t->ocomm)->n_threads;
}

static dim_t bli_thread_ocomm_id( thrinfo_t* t )
{
	return t->ocomm_id;
}

static dim_t bli_thread_n_way( thrinfo_t* t )
{
	return t->n_way;
}

static dim_t bli_thread_work_id( thrinfo_t* t )
{
	return t->work_id;
}

static thrcomm_t* bli_thrinfo_ocomm( thrinfo_t* t )
{
	return t->ocomm;
}

static bool_t bli_thrinfo_needs_free_comm( thrinfo_t* t )
{
	return t->free_comm;
}

static dim_t bli_thread_bszid( thrinfo_t* t )
{
	return t->bszid;
}

static thrinfo_t* bli_thrinfo_sub_node( thrinfo_t* t )
{
	return t->sub_node;
}

static thrinfo_t* bli_thrinfo_sub_prenode( thrinfo_t* t )
{
	return t->sub_prenode;
}

// thrinfo_t query (complex)

static bool_t bli_thread_am_ochief( thrinfo_t* t )
{
	return t->ocomm_id == 0;
}

// thrinfo_t modification

static void bli_thrinfo_set_ocomm( thrcomm_t* ocomm, thrinfo_t* t )
{
	t->ocomm = ocomm;
}

static void bli_thrinfo_set_ocomm_id( dim_t ocomm_id, thrinfo_t* t )
{
	t->ocomm_id = ocomm_id;
}

static void bli_thrinfo_set_n_way( dim_t n_way, thrinfo_t* t )
{
	t->n_way = n_way;
}

static void bli_thrinfo_set_work_id( dim_t work_id, thrinfo_t* t )
{
	t->work_id = work_id;
}

static void bli_thrinfo_set_free_comm( bool_t free_comm, thrinfo_t* t )
{
	t->free_comm = free_comm;
}

static void bli_thrinfo_set_bszid( bszid_t bszid, thrinfo_t* t )
{
	t->bszid = bszid;
}

static void bli_thrinfo_set_sub_node( thrinfo_t* sub_node, thrinfo_t* t )
{
	t->sub_node = sub_node;
}

static void bli_thrinfo_set_sub_prenode( thrinfo_t* sub_prenode, thrinfo_t* t )
{
	t->sub_prenode = sub_prenode;
}

// other thrinfo_t-related functions

static void* bli_thread_broadcast( thrinfo_t* t, void* p )
{
	return bli_thrcomm_bcast( t->ocomm_id, p, t->ocomm );
}

static void bli_thread_barrier( thrinfo_t* t )
{
	bli_thrcomm_barrier( t->ocomm_id, t->ocomm );
}


//
// Prototypes for level-3 thrinfo functions not specific to any operation.
//

thrinfo_t* bli_thrinfo_create
     (
       rntm_t*    rntm,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id, 
       bool_t     free_comm,
       bszid_t    bszid,
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
       bszid_t    bszid,
       thrinfo_t* sub_node
     );

void bli_thrinfo_init_single
     (
       thrinfo_t* thread
     );

void bli_thrinfo_free
     (
       rntm_t*    rntm,
       thrinfo_t* thread
     );

// -----------------------------------------------------------------------------

void bli_thrinfo_grow
     (
       rntm_t*    rntm,
       cntl_t*    cntl,
       thrinfo_t* thread
     );

thrinfo_t* bli_thrinfo_rgrow
     (
       rntm_t*    rntm,
       cntl_t*    cntl_par,
       cntl_t*    cntl_cur,
       thrinfo_t* thread_par
     );

thrinfo_t* bli_thrinfo_create_for_cntl
     (
       rntm_t*    rntm,
       cntl_t*    cntl_par,
       cntl_t*    cntl_chl,
       thrinfo_t* thread_par
     );

thrinfo_t* bli_thrinfo_rgrow_prenode
     (
       rntm_t*    rntm,
       cntl_t*    cntl_par,
       cntl_t*    cntl_cur,
       thrinfo_t* thread_par
     );

thrinfo_t* bli_thrinfo_create_for_cntl_prenode
     (
       rntm_t*    rntm,
       cntl_t*    cntl_par,
       cntl_t*    cntl_chl,
       thrinfo_t* thread_par
     );

// -----------------------------------------------------------------------------

#if 0
void bli_thrinfo_grow_tree
     (
       rntm_t*    rntm,
       cntl_t*    cntl,
       thrinfo_t* thread
     );

void bli_thrinfo_grow_tree_ic
     (
       rntm_t*    rntm,
       cntl_t*    cntl,
       thrinfo_t* thread
     );
#endif

#endif
