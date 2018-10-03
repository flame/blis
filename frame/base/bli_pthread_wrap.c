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

#include "blis.h"

#include <errno.h>

#if defined(_MSC_VER) && !defined(BLIS_ENABLE_PTHREADS)

int pthread_mutex_init(pthread_mutex_t* mutex, const pthread_mutexattr_t *attr)
{
    if (attr) return EINVAL;
    InitializeSRWLock(mutex);
    return 0;
}

int pthread_mutex_destroy(pthread_mutex_t* mutex)
{
    return 0;
}

int pthread_mutex_lock(pthread_mutex_t* mutex)
{
    AcquireSRWLockExclusive(mutex);
    return 0;
}

int pthread_mutex_trylock(pthread_mutex_t* mutex)
{
    return TryAcquireSRWLockExclusive(mutex) ? 0 : EBUSY;
}

int pthread_mutex_unlock(pthread_mutex_t* mutex)
{
    ReleaseSRWLockExclusive(mutex);
    return 0;
}

static BOOL bli_init_once_wrapper(pthread_once_t* once,
                                  void* param,
                                  void** context)
{
    (void)once;
    (void)context;
    typedef void (*callback)(void);
    ((callback)param)();
    return TRUE;
}

void pthread_once(pthread_once_t* once, void (*init)(void))
{
    InitOnceExecuteOnce(once, bli_init_once_wrapper, init, NULL);
}

int pthread_cond_init(pthread_cond_t* cond, const pthread_condattr_t* attr)
{
    if (attr) return EINVAL;
    InitializeConditionVariable(cond);
    return 0;
}

int pthread_cond_destroy(pthread_cond_t* cond)
{
    (void)cond;
    return 0;
}

int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex)
{
    if (!SleepConditionVariableSRW(cond, mutex, INFINITE, 0)) return EAGAIN;
    return 0;
}

int pthread_cond_broadcast(pthread_cond_t* cond)
{
    WakeAllConditionVariable(cond);
    return 0;
}

typedef struct
{
    void* (*start_routine)(void*);
    void* param;
    void** retval;
} bli_thread_param;

static DWORD bli_thread_func(void* param_)
{
    bli_thread_param* param = param_;
    *param->retval = param->start_routine(param->param);
    return 0;
}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void* (*start_routine)(void*), void *arg)
{
    if (attr) return EINVAL;
    bli_thread_param param = {start_routine, arg, &thread->retval};
    thread->handle = CreateThread(NULL, 0, bli_thread_func, &param, 0, NULL);
    if (!thread->handle) return EAGAIN;
    return 0;
}

int pthread_join(pthread_t thread, void **retval)
{
    if (!WaitForSingleObject(thread.handle, INFINITE)) return EAGAIN;
    if (retval) *retval = thread.retval;
    return 0;
}

#endif

#if defined(__APPLE__) || defined(_MSC_VER)

int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count)
{
    if (attr) return EINVAL;
    if (count == 0) return EINVAL;

    int err;
    if ((err = pthread_mutex_init(&barrier->mutex, 0)) != 0) return err;
    if ((err = pthread_cond_init(&barrier->cond, 0)) != 0)
    {
        pthread_mutex_destroy(&barrier->mutex);
        return err;
    }
    barrier->tripCount = count;
    barrier->count = 0;

    return 0;
}

int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}

int pthread_barrier_wait(pthread_barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    ++(barrier->count);
    if(barrier->count >= barrier->tripCount)
    {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return 1;
    }
    else
    {
        pthread_cond_wait(&barrier->cond, &(barrier->mutex));
        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}

#endif
