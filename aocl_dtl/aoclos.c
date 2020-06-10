/*===================================================================
 * File Name :  aoclos.c
 * 
 * Description : Abstraction for os services used by DTL.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/
#include "aocltpdef.h"
#include "aocldtl.h"
#include "aoclfal.h"
#include "aocldtlcf.h"

#if defined(__linux__)
#include <time.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if defined(__linux__)

/* 
    Disable intrumentation for these functions as they will also be
    called from compiler generated instumation code to trace 
    function execution.

    It needs to be part of declration in the C file so can't be 
    moved to header file.
   
*/

uint32 AOCL_gettid(void) __attribute__((no_instrument_function));
pid_t  AOCL_getpid(void) __attribute__((no_instrument_function));
uint64 AOCL_getTimestamp(void) __attribute__((no_instrument_function));

uint32 AOCL_gettid(void)
{
    return syscall(__NR_gettid);
}

pid_t  AOCL_getpid(void)
{
    return getpid();
}

uint64 AOCL_getTimestamp(void)
{
    struct timespec tms;

    /* The C11 way */
    if (clock_gettime(CLOCK_REALTIME, &tms))
    {
        return -1;
    }

    /* seconds, multiplied with 1 million */
    uint64 micros = tms.tv_sec * 1000000;
    /* Add full microseconds */
    micros += tms.tv_nsec / 1000;
    /* round up if necessary */
    if (tms.tv_nsec % 1000 >= 500)
    {
        ++micros;
    }
    return micros;
}

#else  /* Non linux support */
uint32 AOCL_gettid(void)
{
    /* stub for other os's */
    return 0;
}

pid_t  AOCL_getpid(void)
{
	/* stub for other os's */
    return 0;
}

uint64 AOCL_getTimestamp(void)
{
    /* stub for other os's */
    return 0;
}
#endif
