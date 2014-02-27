/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
#ifndef BLIS_THREADING_H
#define BLIS_THREADING_H

typedef omp_lock_t lock_t;

struct thread_comm_s
{
    void*   sent_object;
    dim_t   n_threads;

    bool_t  barrier_sense;
    lock_t  barrier_lock;
    dim_t   barrier_threads_arrived;
};
typedef struct thread_comm_s thread_comm_t;

struct thread_comm_tree_s
{
    struct thread_comm_tree_s* parent;
    thread_comm_t* comm;
};
typedef struct thread_comm_tree_s thread_comm_tree_t;


void    bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads );
thread_comm_t*    bli_create_communicator( dim_t n_threads );

void*   bli_broadcast_structure( thread_comm_t* communicator, dim_t inside_id, void* to_send );

void    bli_barrier( thread_comm_t* communicator, dim_t thread_id );
void    bli_set_lock( lock_t* lock );
void    bli_unset_lock( lock_t* lock );
void    bli_init_lock( lock_t* lock );
void    bli_destroy_lock( lock_t* lock );

/*
 * Each thrinfo_t is a linked list.
 * It represents a path through a thread communicator hierarchy.
 * There is a 1:1 correspondence between leaf nodes and thrinfo_t 
 *
 * When we hit a loop, we advance the linked list towards the bottom of the hierarchy
 */
struct thrinfo_s
{
    thread_comm_t*      ocomm;       //The thread communicator for the other threads sharing the same work at this level
    dim_t               ocomm_id;    //Our thread id within that thread comm

    struct thrinfo_s*   caucus;      //my thread info for the caucus I am a part of 
    dim_t               n_caucuses; //Number of distinct caucuses used to parallelize the loop
    dim_t               caucus_id;  //Which caucus we are part of
};
typedef struct thrinfo_s thrinfo_t;

#define thread_comm( thread )           thread->ocomm
#define thread_caucus_comm( thread )     (thread->caucus->ocomm)

#define thread_id( thread )             thread->ocomm_id
#define thread_num_threads( thread )    thread->ocomm->n_threads
#define thread_sub_caucus( thread )      thread->caucus
#define thread_caucus_id( thread )       thread->caucus_id
#define thread_num_caucuses( thread )    thread->n_caucuses
#define thread_am_chief( thread )        (thread->ocomm_id == 0)
#define thread_am_caucus_chief( thread ) (thread->caucus->ocomm_id == 0) 

#define thread_broadcast( thread, ptr )             bli_broadcast_structure( thread->ocomm, thread->ocomm_id, ptr )
#define thread_caucus_broadcast( thread, ptr )      bli_broadcast_structure( thread->caucus->ocomm, thread->caucus->ocomm_id, ptr )
#define thread_barrier( thread )                    bli_barrier( thread->ocomm, thread->ocomm_id )
#define thread_caucus_barrier( thread )             bli_barrier( thread->caucus->ocomm, thread->caucus->ocomm_id )

thrinfo_t* bli_create_thread_info( dim_t* n_threads_each_level, dim_t n_levels );
void bli_get_range( thrinfo_t* thread, dim_t size, dim_t block_factor, dim_t* start, dim_t* end );
void bli_setup_single_threaded_info( thrinfo_t* thr, thread_comm_t* comm );

#endif
