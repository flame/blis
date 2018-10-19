/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin

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

// -- pthread types ------------------------------------------------------------

typedef pthread_t              bli_pthread_t;
typedef pthread_attr_t         bli_pthread_attr_t;
typedef pthread_mutex_t        bli_pthread_mutex_t;
typedef pthread_mutexattr_t    bli_pthread_mutexattr_t;
typedef pthread_barrier_t      bli_pthread_barrier_t;
typedef pthread_barrierattr_t  bli_pthread_barrierattr_t;
typedef pthread_once_t         bli_pthread_once_t;

// -- pthread_create(), pthread_join() -----------------------------------------

int bli_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     );

int bli_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     );

// -- pthread_mutex_*() --------------------------------------------------------

int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     );

int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     );

int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     );

int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     );

// -- pthread_barrier_*() ------------------------------------------------------

int bli_pthread_barrier_init
     (
       bli_pthread_barrier_t*           barrier,
       const bli_pthread_barrierattr_t* attr,
       unsigned int                     count
     );

int bli_pthread_barrier_destroy
     (
       bli_pthread_barrier_t* barrier
     );

int bli_pthread_barrier_wait
     (
       bli_pthread_barrier_t* barrier
     );

// -- pthread_once_*() ---------------------------------------------------------

void bli_pthread_once
     (
       bli_pthread_once_t* once,
       void              (*init)(void)
     );
