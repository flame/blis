/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, Southern Methodist University

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

#ifndef BLIS_PTHREAD_WRAP_H
#define BLIS_PTHREAD_WRAP_H

#if defined(_MSC_VER)

typedef SRWLOCK pthread_mutex_t;
typedef void pthread_mutexattr_t;

#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

int pthread_mutex_init(pthread_mutex_t* mutex, const pthread_mutexattr_t *attr);

int pthread_mutex_destroy(pthread_mutex_t* mutex);

int pthread_mutex_lock(pthread_mutex_t* mutex);

int pthread_mutex_trylock(pthread_mutex_t* mutex);

int pthread_mutex_unlock(pthread_mutex_t* mutex);

typedef INIT_ONCE pthread_once_t;

#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

void pthread_once(pthread_once_t* once, void (*init)(void));

typedef CONDITION_VARIABLE pthread_cond_t;
typedef void pthread_condattr_t;

#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT

int pthread_cond_init(pthread_cond_t* cond, const pthread_condattr_t* attr);

int pthread_cond_destroy(pthread_cond_t* cond);

int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex);

int pthread_cond_broadcast(pthread_cond_t* cond);

typedef struct
{
    HANDLE handle;
    void* retval;
} pthread_t;

typedef void pthread_attr_t;

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void* (*start_routine)(void*), void *arg);

int pthread_join(pthread_t thread, void **retval);

#else

#include <pthread.h>

#endif

#if defined(__APPLE__) || defined(_MSC_VER)

typedef void pthread_barrierattr_t;

typedef struct
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int tripCount;
} pthread_barrier_t;

int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count);

int pthread_barrier_destroy(pthread_barrier_t *barrier);

int pthread_barrier_wait(pthread_barrier_t *barrier);

#endif // _POSIX_BARRIERS

#endif
