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

#if defined(_MSC_VER) && !defined(BLIS_ENABLE_PTHREADS)

int pthread_mutex_init(pthread_mutex_t* mutex, const pthread_mutexattr_t *attr)
{
    if (attr != NULL) return EINVAL;
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
    ((callpack)param)();
    return TRUE;
}

void pthread_once(pthread_once_t* once, void (*init)(void))
{
    InitOnceExecuteOnce(once, bli_init_once_wrapper, init, NULL);
}

#endif
