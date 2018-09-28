/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP

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

#ifndef BLIS_RNTM_H
#define BLIS_RNTM_H


// Runtime object type (defined in bli_type_defs.h)

/*
typedef struct rntm_s
{
	dim_t     num_threads;
	dim_t*    thrloop;
} rntm_t;
*/

//
// -- rntm_t query -------------------------------------------------------------
//

static dim_t bli_rntm_num_threads( rntm_t* rntm )
{
	return rntm->num_threads;
}

static dim_t bli_rntm_ways_for( bszid_t bszid, rntm_t* rntm )
{
	return rntm->thrloop[ bszid ];
}

static dim_t bli_rntm_jc_ways( rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_NC, rntm );
}
static dim_t bli_rntm_pc_ways( rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_KC, rntm );
}
static dim_t bli_rntm_ic_ways( rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_MC, rntm );
}
static dim_t bli_rntm_jr_ways( rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_NR, rntm );
}
static dim_t bli_rntm_ir_ways( rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_MR, rntm );
}
static dim_t bli_rntm_pr_ways( rntm_t* rntm )
{
	return bli_rntm_ways_for( BLIS_KR, rntm );
}

//
// -- rntm_t modification (internal use only) ----------------------------------
//

static void bli_rntm_set_num_threads_only( dim_t nt, rntm_t* rntm )
{
	rntm->num_threads = nt;
}

static void bli_rntm_set_ways_for_only( bszid_t loop, dim_t n_ways, rntm_t* rntm )
{
	rntm->thrloop[ loop ] = n_ways;
}

static void bli_rntm_set_jc_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_NC, ways, rntm );
}
static void bli_rntm_set_pc_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_KC, ways, rntm );
}
static void bli_rntm_set_ic_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_MC, ways, rntm );
}
static void bli_rntm_set_jr_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_NR, ways, rntm );
}
static void bli_rntm_set_ir_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_MR, ways, rntm );
}
static void bli_rntm_set_pr_ways_only( dim_t ways, rntm_t* rntm )
{
	bli_rntm_set_ways_for_only( BLIS_KR, ways, rntm );
}

static void bli_rntm_set_ways_only( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir, rntm_t* rntm )
{
	// Record the number of ways of parallelism per loop.
	bli_rntm_set_jc_ways_only( jc, rntm );
	bli_rntm_set_pc_ways_only( pc, rntm );
	bli_rntm_set_ic_ways_only( ic, rntm );
	bli_rntm_set_jr_ways_only( jr, rntm );
	bli_rntm_set_ir_ways_only( ir, rntm );
	bli_rntm_set_pr_ways_only(  1, rntm );
}

static void bli_rntm_clear_num_threads_only( rntm_t* rntm )
{
	bli_rntm_set_num_threads_only( -1, rntm );
}
static void bli_rntm_clear_ways_only( rntm_t* rntm )
{
	bli_rntm_set_ways_only( -1, -1, -1, -1, -1, rntm );
}

//
// -- rntm_t modification (public API) -----------------------------------------
//

static void bli_rntm_set_num_threads( dim_t nt, rntm_t* rntm )
{
	// Record the total number of threads to use.
	bli_rntm_set_num_threads_only( nt, rntm );

	// Set the individual ways of parallelism to default states.
	bli_rntm_clear_ways_only( rntm );
}

static void bli_rntm_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir, rntm_t* rntm )
{
	// Record the number of ways of parallelism per loop.
	bli_rntm_set_jc_ways_only( jc, rntm );
	bli_rntm_set_pc_ways_only( pc, rntm );
	bli_rntm_set_ic_ways_only( ic, rntm );
	bli_rntm_set_jr_ways_only( jr, rntm );
	bli_rntm_set_ir_ways_only( ir, rntm );
	bli_rntm_set_pr_ways_only(  1, rntm );

	// Set the num_threads field to a default state.
	bli_rntm_clear_num_threads_only( rntm );
}

//
// -- rntm_t initialization ----------------------------------------------------
//

// NOTE: Initialization is not necessary as long the user calls at least ONE
// of the public "set" accessors, each of which guarantees that the rntm_t
// will be in a good state upon return.

#define BLIS_RNTM_INITIALIZER { .num_threads = -1, \
                                .thrloop = { -1, -1, -1, -1, -1, -1 } } \

static void bli_rntm_init( rntm_t* rntm )
{
	bli_rntm_clear_num_threads_only( rntm );
	bli_rntm_clear_ways_only( rntm );
}

// -----------------------------------------------------------------------------

// Function prototypes

void bli_rntm_set_ways_for_op
     (
       opid_t  l3_op,
       side_t  side,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     );

void bli_rntm_set_ways_from_rntm
     (
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     );

void bli_rntm_print
     (
       rntm_t* rntm
     );

#endif

