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

#ifdef BLIS_ENABLE_OPENMP

//Constructors and destructors for constructors
thread_comm_t* bli_create_communicator( dim_t n_threads )
{
    thread_comm_t* comm = (thread_comm_t*) bli_malloc( sizeof(thread_comm_t) );
    bli_setup_communicator( comm, n_threads );
    return comm;
}

void bli_free_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    bli_cleanup_communicator( communicator );
    bli_free( communicator );
}

void bli_level3_thread_decorator( dim_t n_threads, 
                                  level3_int_t func, 
                                  obj_t* alpha, 
                                  obj_t* a, 
                                  obj_t* b, 
                                  obj_t* beta, 
                                  obj_t* c, 
                                  void* cntl, 
                                  void** thread )
{
    _Pragma( "omp parallel num_threads(n_threads)" )
    {
        dim_t omp_id = omp_get_thread_num();

        func( alpha,
                  a,
                  b,
                  beta,
                  c,
                  cntl,
                  thread[omp_id] );
    }
}

#ifndef BLIS_TREE_BARRIER
//'Normal' barrier for openmp
//barrier routine taken from art of multicore programming
void bli_barrier( thread_comm_t* communicator, dim_t t_id )
{
    if(communicator == NULL || communicator->n_threads == 1)
        return;
    bool_t my_sense = communicator->barrier_sense;
    dim_t my_threads_arrived;

    _Pragma("omp atomic capture")
        my_threads_arrived = ++(communicator->barrier_threads_arrived);

    if( my_threads_arrived == communicator->n_threads ) {
        communicator->barrier_threads_arrived = 0;
        communicator->barrier_sense = !communicator->barrier_sense;
    }
    else {
        volatile bool_t* listener = &communicator->barrier_sense;
        while( *listener == my_sense ) {}
    }
}


void bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads)
{
    if( communicator == NULL ) return;
    communicator->sent_object = NULL;
    communicator->n_threads = n_threads;
    communicator->barrier_sense = 0;
    communicator->barrier_threads_arrived = 0;
}


void bli_cleanup_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
}
#else

//Tree barrier used for Intel Xeon Phi
void bli_free_barrier_tree( barrier_t* barrier )
{
    if( barrier == NULL )
        return;
    barrier->count--;
    if( barrier->count == 0 )
    {
        bli_free_barrier_tree( barrier->dad );
        bli_free( barrier );
    }
    return;
}
barrier_t* bli_create_tree_barrier(int num_threads, int arity, barrier_t** leaves, int leaf_index)
{
    barrier_t* me = (barrier_t*) malloc(sizeof(barrier_t));

    me->dad = NULL;
    me->signal = 0;

    // Base Case
    if( num_threads <= arity ) {
       //Now must be registered as a leaf
       for(int i = 0; i < num_threads; i++)
       {
           leaves[leaf_index + i] = me;
       }
       me->count = num_threads;
       me->arity = num_threads;
    }
    else {
        // Otherwise this node has children
        int threads_per_kid = num_threads / arity;
        int defecit = num_threads - threads_per_kid * arity;

        for(int i = 0; i < arity; i++){
            int threads_this_kid = threads_per_kid;
            if(i < defecit) threads_this_kid++;
            
            barrier_t* kid = bli_create_tree_barrier(threads_this_kid, arity, leaves, leaf_index);
            kid->dad = me;

            leaf_index += threads_this_kid;
        }  
        me->count = arity;
        me->arity = arity;
    }  

    return me;
}

void bli_cleanup_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    for( dim_t i = 0; i < communicator->n_threads; i++)
    {
       bli_free_barrier_tree( communicator->barriers[i] );
    }
    bli_free( communicator->barriers );
}


void bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads)
{
    if( communicator == NULL ) return;
    communicator->sent_object = NULL;
    communicator->n_threads = n_threads;
    communicator->barriers = ( barrier_t** ) bli_malloc( sizeof( barrier_t* ) * n_threads );
    bli_create_tree_barrier( n_threads, BLIS_TREE_BARRIER_ARITY, communicator->barriers, 0 );
}

void tree_barrier( barrier_t* barack )
{
    int my_signal = barack->signal;
    int my_count;

    _Pragma("omp atomic capture")
        my_count = barack->count--;

    if( my_count == 1 ) {
        if(  barack->dad != NULL ) {
            tree_barrier( barack->dad );
        }
        barack->count = barack->arity;
        barack->signal = !barack->signal;
    }
    else {
        volatile int* listener = &barack->signal;
        while( *listener == my_signal ) {}
    }
}

void bli_barrier( thread_comm_t* comm, dim_t t_id )
{
    tree_barrier( comm->barriers[t_id] );
}
#endif
#endif
