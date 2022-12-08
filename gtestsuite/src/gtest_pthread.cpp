#include "gtest_pthread.h"

#if defined(BLIS_DISABLE_SYSTEM)

#elif defined(_MSC_VER) // !defined(BLIS_DISABLE_SYSTEM)

#include <errno.h>

int gtest_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                    (*start_routine)(void*),
       void*                     arg
     )
{
    if (attr) return EINVAL;
    LPTHREAD_START_ROUTINE func = (LPTHREAD_START_ROUTINE )start_routine;
    thread->handle = CreateThread(NULL, 0, func, arg, 0, NULL);
    if ( !thread->handle ) return EAGAIN;
    return 0;
}

int gtest_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     )
{
  return bli_pthread_join(thread, retval);
}

#else // !defined(BLIS_DISABLE_SYSTEM) && !defined(_MSC_VER)

// This branch defines a pthreads-like API, bli_pthreads_*(), and implements it
// in terms of the corresponding pthreads_*() types, macros, and function calls.
// This branch is compiled for Linux and other non-Windows environments where
// we assume that *some* implementation of pthreads is provided (although it
// may lack barriers--see below).

// -- pthread_create(), pthread_join() --

int gtest_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     )
{
    return bli_pthread_create( thread, attr, start_routine, arg );
}

int gtest_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     )
{
    return bli_pthread_join( thread , retval );
}
#endif // !defined(BLIS_DISABLE_SYSTEM) && !defined(_MSC_VER)
