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

#include "blis.h"

void bli_cleanup_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    bli_destroy_lock( &communicator->barrier_lock );
}

void bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads)
{
    if( communicator == NULL ) return;
    communicator->sent_object = NULL;
    communicator->n_threads = n_threads;
    communicator->barrier_sense = 0;
    bli_init_lock( &communicator->barrier_lock );
    communicator->barrier_threads_arrived = 0;
}

thread_comm_t* bli_create_communicator( dim_t n_threads )
{
    thread_comm_t* comm = (thread_comm_t*) bli_malloc( sizeof(thread_comm_t) );
    bli_setup_communicator( comm, n_threads );
    return comm;
}

void* bli_broadcast_structure( thread_comm_t* communicator, dim_t id, void* to_send )
{   
    if( communicator == NULL || communicator->n_threads == 1 ) return to_send;

    if( id == 0 ) communicator->sent_object = to_send;

    bli_barrier( communicator, id );
    void * object = communicator->sent_object;
    bli_barrier( communicator, id );

    return object;
}

void bli_init_lock( lock_t* lock )
{
    omp_init_lock( lock );
}
void bli_destroy_lock( lock_t* lock )
{
    omp_destroy_lock( lock );
}
void bli_set_lock( lock_t* lock )
{
    omp_set_lock( lock );
}
void bli_unset_lock( lock_t* lock )
{
    omp_unset_lock( lock );
}

//barrier routine taken from art of multicore programming or something
void bli_barrier( thread_comm_t* communicator, dim_t t_id )
{
    if(communicator == NULL || communicator->n_threads == 1)
        return;
    bool_t my_sense = communicator->barrier_sense;
    dim_t my_threads_arrived;

    _Pragma("omp atomic capture")
        my_threads_arrived = ++(communicator->barrier_threads_arrived);

/*
    bli_set_lock(&communicator->barrier_lock);
    my_threads_arrived = communicator->barrier_threads_arrived + 1;
    communicator->barrier_threads_arrived = my_threads_arrived;
    bli_unset_lock(&communicator->barrier_lock);
*/

    if( my_threads_arrived == communicator->n_threads ) {

        bli_set_lock(&communicator->barrier_lock);
        communicator->barrier_threads_arrived = 0;
        communicator->barrier_sense = !communicator->barrier_sense;
        bli_unset_lock(&communicator->barrier_lock);
    }
    else {
        volatile bool_t* listener = &communicator->barrier_sense;
        while( *listener == my_sense ) {}
    }
}
/*
//Recursively create thread communicators
void create_comms( dim_t* caucuses_at_level, dim_t n_levels, dim_t cur_level, 
                        thread_comm_tree_t* parent, thread_comm_tree_t* leaves, dim_t global_id )
{
    //Create a communicator
    dim_t n_threads = 1;
    for( dim_t i = cur_level; i < n_levels; i++)
        n_threads *= caucuses_at_level[i];


    thread_comm_t* comm = bli_create_communicator( n_threads );
    thread_comm_tree_t* info;
    if( cur_level == n_levels )
    {
        leaves[global_id].parent = parent;
        leaves[global_id].comm = comm;
        return;
    }
    else
    {   
        info = (thread_comm_tree_t*)bli_malloc(sizeof(thread_comm_tree_t));
        info->comm = comm;
        info->parent = parent;
    }

    //Now create child communicators
    dim_t caucuses = caucuses_at_level[cur_level];
    for( dim_t i = 0; i < caucuses; i++)
        create_comms( caucuses_at_level, n_levels, cur_level+1, info, leaves, global_id * caucuses + i);
}
*/
thrinfo_t* bli_create_thread_info( thread_comm_t* ocomm, dim_t ocomm_id, thread_comm_t* icomm, dim_t icomm_id,
                             dim_t n_way, dim_t work_id )
{

        thrinfo_t* thr = (thrinfo_t*) bli_malloc( sizeof(thrinfo_t) );
        bli_setup_thread_info( thr, ocomm, ocomm_id, icomm, icomm_id, n_way, work_id );
        return thr;
}

void bli_setup_thread_info( thrinfo_t* thr, thread_comm_t* ocomm, dim_t ocomm_id, thread_comm_t* icomm, dim_t icomm_id,
                             dim_t n_way, dim_t work_id )
{
        thr->ocomm = ocomm;
        thr->ocomm_id = ocomm_id;
        thr->icomm = icomm;
        thr->icomm_id = icomm_id;

        thr->n_way = n_way;
        thr->work_id = work_id;
}

/*
thrinfo_t* bli_create_thread_info( dim_t* caucuses_at_level, dim_t n_levels )
{
    //Calculate total number of threads
    dim_t n_threads = 1;
    for( dim_t i = 0; i < n_levels; i++)
        n_threads *= caucuses_at_level[i];
    
    //Create communicators
    thread_comm_tree_t* comm_leaves = (thread_comm_tree_t*)bli_malloc( sizeof(thread_comm_tree_t) * n_threads);
    create_comms( caucuses_at_level, n_levels, 0, NULL, comm_leaves, 0 );
    thrinfo_t* info_paths = (thrinfo_t*)bli_malloc( sizeof(thrinfo_t) * n_threads );

    //Now create paths upwards
    for( dim_t i = 0; i < n_threads; i++ )
    {
        thread_comm_tree_t* comm_node = &comm_leaves[i];

        //Setup thread info for the bottom-most level
        thrinfo_t* bot = &BLIS_SINGLE_THREADED; //bli_create_thrinfo_t( comm_node->comm, 0, NULL, 1, 0 );
        
        //Now build thread infos upwards
        comm_node = comm_node->parent;
        thrinfo_t* cur;
        thrinfo_t* prev = bot;
        for( dim_t j = 0; j < n_levels; j++ )
        {   
            if( j == n_levels - 1 )
                cur = &info_paths[i];
            else
                cur = (thrinfo_t*)bli_malloc(sizeof(thrinfo_t));

            dim_t caucus_size = prev->ocomm->n_threads;
            dim_t ocomm_id  = i % comm_node->comm->n_threads;
            dim_t caucus_id = ocomm_id / caucus_size;
            
            bli_setup_thrinfo_t(cur, comm_node->comm, ocomm_id, 
                                prev, caucuses_at_level[n_levels - j - 1], caucus_id );

            prev = cur;
            comm_node = comm_node->parent;
        }
    }
    return info_paths;
}
*/
void bli_get_range( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, dim_t* start, dim_t* end )
{
    thrinfo_t* thread = (thrinfo_t*) thr;
    dim_t n_way = thread->n_way;
    dim_t work_id = thread->work_id;

    dim_t size = all_end - all_start;
    dim_t n_pt = size / n_way;
    n_pt = (n_pt * n_way < size) ? n_pt + 1 : n_pt;
    n_pt = (n_pt % block_factor == 0) ? n_pt : n_pt + block_factor - (n_pt % block_factor); 
    *start = work_id * n_pt + all_start;
    *end   = bli_min( *start + n_pt, size + all_start );
}

void bli_get_range_weighted( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, bool_t forward, dim_t* start, dim_t* end)
{
    thrinfo_t* thread = (thrinfo_t*) thr;
    dim_t n_way = thread->n_way;
    dim_t work_id = thread->work_id;
    dim_t size = all_end - all_start;

    *start = all_start;
    *end   = all_end;

    if( forward ) {
        dim_t curr_caucus = n_way - 1;
        dim_t len = 0;
        dim_t num = size*size / n_way; // 2xArea per thread?
        while(1){
            dim_t width = sqrt( len*len + num ) - len; // The width of the current caucus
            width = (width % block_factor == 0) ? width : width + block_factor - (width % block_factor);
            if( curr_caucus == work_id ) {
                if( *end > width )
                    *start = *end - width;
                return;
            }
            else{
                *end -= width;
                len += width;
                curr_caucus--;
            }
        }
    }
    else{
        dim_t len = *end - *start;
        dim_t num = len * len / n_way;
        while(1){
            dim_t width = sqrt(*start * *start + num) - *start;
            width = (width % block_factor == 0) ? width : width + block_factor - (width % block_factor);
            if(!work_id) {
                *end = bli_min( *start + width, *end );
                return;
            }
            else{
                *start = *start + width;
            }
            work_id--;
        }
    }
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

dim_t bli_read_nway_from_env( char* env )
{
    dim_t number = 1;
    char* str = getenv( env );
    if( str != NULL )
    {   
        number = strtol( str, NULL, 10 );
    }   
    return number;
}
