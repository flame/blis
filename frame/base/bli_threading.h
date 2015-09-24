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

#ifndef BLIS_THREADING_H
#define BLIS_THREADING_H

// Perform a sanity check to make sure the user doesn't try to enable
// both OpenMP and pthreads.
#if defined (BLIS_ENABLE_OPENMP) && defined (BLIS_ENABLE_PTHREADS)
  #error "BLIS_ENABLE_OPENMP and BLIS_ENABLE_PTHREADS may not be simultaneously defined."
#endif

// Here, we define BLIS_ENABLE_MULTITHREADING if either OpenMP
// or pthreads are enabled. This macro is useful in situations when
// we want to detect use of either OpenMP or pthreads (as opposed
// to neither being used).
#if defined (BLIS_ENABLE_OPENMP) || defined (BLIS_ENABLE_PTHREADS)
  #define BLIS_ENABLE_MULTITHREADING
#endif

// Define thread_comm_t for situations when multithreading is disabled.
#ifndef BLIS_ENABLE_MULTITHREADING 

//thread communicators may be implementation dependent
#ifdef BLIS_TREE_BARRIER
    struct barrier_s
    {   
        int arity;
        int count;
        struct barrier_s* dad;
        int signal;
    };  
    typedef struct barrier_s barrier_t;

    struct thread_comm_s
    {   
        void*   sent_object;
        dim_t   n_threads;
        barrier_t** barriers;
    }; 
#else
    struct thread_comm_s
    {
        void*   sent_object;
        dim_t   n_threads;

        bool_t  barrier_sense;
        dim_t   barrier_threads_arrived;
    };
#endif
typedef struct thread_comm_s thread_comm_t;

#endif

// Include OpenMP or pthreads definitions.
#include "bli_threading_omp.h"
#include "bli_threading_pthreads.h"

void    bli_thread_init( void );
void    bli_thread_finalize( void );
bool_t  bli_thread_is_initialized( void );

// Thread Communicator Interface Definitions
void    bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads );
void    bli_cleanup_communicator( thread_comm_t* communicator );
thread_comm_t*    bli_create_communicator( dim_t n_threads );
void    bli_free_communicator( thread_comm_t* communicator );
void*   bli_broadcast_structure( thread_comm_t* communicator, dim_t inside_id, void* to_send );
void    bli_barrier( thread_comm_t* communicator, dim_t thread_id );

struct thrinfo_s
{
    // The thread communicator for the other threads sharing the same work
	// at this level.
    thread_comm_t*      ocomm;

    // Our thread id within the ocomm thread communicator.
    dim_t               ocomm_id;

    // The thread communicator for the other threads sharing the same work
	// at this level.
    thread_comm_t*      icomm;

	// Our thread id within the icomm thread communicator.
    dim_t               icomm_id;

	// The number of distinct threads used to parallelize the loop.
    dim_t               n_way;

	// What we're working on.
    dim_t               work_id;
};
typedef struct thrinfo_s thrinfo_t;

// Thread Info Interface Definitions
#define thread_ocomm( thread )          (thread->ocomm)
#define thread_icomm( thread )          (thread->icomm)

#define thread_id( thread )             (thread->ocomm_id)
#define thread_num_threads( thread )    (thread->ocomm->n_threads)

#define thread_work_id( thread )        (thread->work_id)
#define thread_n_way( thread )          (thread->n_way)
#define thread_am_ochief( thread )      (thread->ocomm_id == 0)
#define thread_am_ichief( thread )      (thread->icomm_id == 0)

#define thread_obroadcast( thread, ptr )       bli_broadcast_structure( thread->ocomm, thread->ocomm_id, ptr )
#define thread_ibroadcast( thread, ptr )       bli_broadcast_structure( thread->icomm, thread->icomm_id, ptr )
#define thread_obarrier( thread )              bli_barrier( thread->ocomm, thread->ocomm_id )
#define thread_ibarrier( thread )              bli_barrier( thread->icomm, thread->icomm_id )

void bli_get_range( void* thr, dim_t n, dim_t bf, bool_t handle_edge_low,
                    dim_t* start, dim_t* end );
siz_t bli_get_range_l2r( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );
siz_t bli_get_range_r2l( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );
siz_t bli_get_range_t2b( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );
siz_t bli_get_range_b2t( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );

dim_t bli_get_range_width_l( doff_t diagoff_j,
                             dim_t  m,
                             dim_t  n_j,
                             dim_t  j,
                             dim_t  n_way,
                             dim_t  bf,
                             dim_t  bf_left,
                             double area_per_thr,
                             bool_t handle_edge_low );
siz_t bli_find_area_trap_l( dim_t m, dim_t n, doff_t diagoff );
siz_t bli_get_range_weighted( void*  thr,
                              doff_t diagoff,
                              uplo_t uplo,
                              dim_t  m,
                              dim_t  n,
                              dim_t  bf,
                              bool_t handle_edge_low,
                              dim_t* j_start_thr,
                              dim_t* j_end_thr );

siz_t bli_get_range_weighted_l2r( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );
siz_t bli_get_range_weighted_r2l( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );
siz_t bli_get_range_weighted_t2b( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );
siz_t bli_get_range_weighted_b2t( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end );

thrinfo_t* bli_create_thread_info( thread_comm_t* ocomm, dim_t ocomm_id, 
                                   thread_comm_t* icomm, dim_t icomm_id,
                                   dim_t n_way, dim_t work_id );
void bli_setup_thread_info( thrinfo_t* thread, thread_comm_t* ocomm, dim_t ocomm_id, 
                            thread_comm_t* icomm, dim_t icomm_id,
                            dim_t n_way, dim_t work_id );
dim_t bli_read_nway_from_env( char* env );

//void bli_setup_single_threaded_info( thrinfo_t* thr, thread_comm_t* comm );
//thrinfo_t* bli_create_thread_info( dim_t* n_threads_each_level, dim_t n_levels );


//TODO: These nneed to be included after the thread info and thread comm definitions
// But this doesn't seem like the best place to put these includes.
// Note that the bli_packm_threading.h must be included before the others!
#include "bli_packm_threading.h"
#include "bli_gemm_threading.h"
#include "bli_herk_threading.h"
#include "bli_trmm_threading.h"
#include "bli_trsm_threading.h"

typedef void (*level3_int_t) ( obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c, void* cntl, void* thread );
void bli_level3_thread_decorator( dim_t num_threads, 
                                  level3_int_t func, 
                                  obj_t* alpha, 
                                  obj_t* a,  
                                  obj_t* b,  
                                  obj_t* beta, 
                                  obj_t* c,  
                                  void* cntl, 
                                  void** thread );


dim_t bli_gcd( dim_t x, dim_t y );
dim_t bli_lcm( dim_t x, dim_t y );

#endif
