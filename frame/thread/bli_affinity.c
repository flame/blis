/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 NVIDIA

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

// we need a way to detect oversubscription of the kind where
// hierarchical parallelism is used and the affinity mask within
// which BLIS runs does not have enough hardware threads to support
// the requested software threads.
//
// this is motivated by, or related to:
//    https://github.com/flame/blis/issues/588
//    https://github.com/flame/blis/pull/607
//    https://github.com/flame/blis/issues/604
//    https://github.com/flame/blis/issues/603 

#include "bli_affinity.h"

#ifndef BLIS_OS_LINUX

// define the symbol for platforms like Windows and MacOS that do not support the Linux affinity API

dim_t bli_affinity_get_hw_size(bli_affinity_scope_t scope)
{
    // this is the largest possible value returned by this function
    // and it means that the affinity mask does not constrain the current scope.
    return (dim_t)1024;
}

#else // BLIS_OS_LINUX

// this macro has to come before any other headers
#define _GNU_SOURCE

#include <sched.h>
#include <unistd.h>

// scope is either the calling process or the calling thread:
//  0 = calling process
//  1 = calling thread

dim_t bli_affinity_get_hw_size(bli_affinity_scope_t scope)
{
    int rc;
    int active_cpus;
    pid_t pid;
    cpu_set_t mask;

    if (scope == 0) {
        pid = getpid();
    } else {
        // this means the current thread
        pid = 0;
    }

    CPU_ZERO(&mask);

    // if the CPU mask is larger than 1024 bits, this needs to change.
    // see https://man7.org/linux/man-pages/man2/sched_getaffinity.2.html for details.
    rc = sched_getaffinity(pid, sizeof(cpu_set_t), &mask);
    if (rc) {
        bli_print_msg( "sched_getaffinity failed",
                       __FILE__, __LINE__ );
        bli_abort();
    }

    active_cpus = 0;
    for (int i=0; i<sizeof(cpu_set_t); i++) {
        const int on = CPU_ISSET(i, &mask);
        if (on) active_cpus++;
    }

    return active_cpus;
}

#endif // BLIS_OS_LINUX
