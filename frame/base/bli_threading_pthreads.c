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

#ifdef BLIS_ENABLE_PTHREADS

void* thread_decorator_helper( void* data_void );

typedef struct thread_data
{
  level3_int_t func;
  obj_t* alpha; 
  obj_t* a; 
  obj_t* b; 
  obj_t* beta;
  obj_t* c;
  void* cntl;
  void* thread;
} thread_data_t;

void* thread_decorator_helper( void* data_void )
{
	thread_data_t* data = data_void;

    data->func( data->alpha, data->a, data->b, data->beta, data->c, data->cntl, data->thread );

	return NULL;
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
    pthread_t* pthreads = (pthread_t*) bli_malloc(sizeof(pthread_t) * n_threads);
    //Saying "datas" is kind of like saying "all y'all"
    thread_data_t* datas = (thread_data_t*) bli_malloc(sizeof(thread_data_t) * n_threads);
    //pthread_attr_t* attr = (pthread_attr_t*) bli_malloc(sizeof(pthread_attr_t) * n_threads);

    for( int i = 0; i < n_threads; i++ )
    {
        //Setup the thread data
        datas[i].func = func;
        datas[i].alpha = alpha;
        datas[i].a = a;
        datas[i].b = b;
        datas[i].beta = beta;
        datas[i].c = c;
        datas[i].cntl = cntl;
        datas[i].thread = thread[i];
        pthread_create( &pthreads[i], NULL, &thread_decorator_helper, &datas[i] );
    }

    for( int i = 0; i < n_threads; i++)
    {
        pthread_join( pthreads[i], NULL );
    }

    bli_free( pthreads );
    bli_free( datas );
}

//barrier routine taken from art of multicore programming
void bli_barrier( thread_comm_t* communicator, dim_t t_id )
{
    pthread_barrier_wait( &communicator->barrier );
}

//Constructors and destructors for constructors
thread_comm_t* bli_create_communicator( dim_t n_threads )
{
    thread_comm_t* comm = (thread_comm_t*) bli_malloc( sizeof(thread_comm_t) );
    bli_setup_communicator( comm, n_threads );
    return comm;
}

void bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads)
{
    if( communicator == NULL ) return;
    communicator->sent_object = NULL;
    communicator->n_threads = n_threads;
    pthread_barrier_init( &communicator->barrier, NULL, n_threads );
}

void bli_free_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    bli_cleanup_communicator( communicator );
    bli_free( communicator );
}

void bli_cleanup_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    pthread_barrier_destroy( &communicator->barrier );
}
#endif
