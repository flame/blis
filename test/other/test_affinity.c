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

// this macro has to come before any other headers.
// i hate this but cannot figure out any other way to solve it.
#define _GNU_SOURCE

#include <sched.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>

#include <cblas.h>

int main(void)
{
    int m=10, n=10, k=10;
    double A[100], B[100], C[100];

    for (int i=0; i<100; i++) {
        A[i] = B[i] = C[i] = 1.0;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                10, 10, 10, 1.0, A, 10, B, 10, 1.0, C, 10); 

    {
        int rc;
        pid_t pid = getpid();
        cpu_set_t old_mask, new_mask;
        int active_cpus;

        CPU_ZERO(&old_mask);

        rc = sched_getaffinity(pid, sizeof(cpu_set_t), &old_mask);
        if (rc) {
            printf("sched_getaffinity returned %d\n", rc);
            abort();
        }

        active_cpus = 0;
        for (int i=0; i<sizeof(cpu_set_t); i++) {
            const int on = CPU_ISSET(i, &old_mask);
            if (on) active_cpus++;
        }
        printf("active CPUs before = %d\n", active_cpus);

        CPU_ZERO(&new_mask);

        for (int i=0, j=0; i<sizeof(cpu_set_t); i++) {
            const int on = CPU_ISSET(i, &old_mask);
            if (on) {
                if (j < active_cpus / 2) {
                    CPU_SET(i, &new_mask);
                    j++;
                }
            }
        }

        active_cpus = 0;
        for (int i=0; i<sizeof(cpu_set_t); i++) {
            const int on = CPU_ISSET(i, &new_mask);
            if (on) active_cpus++;
        }
        printf("active CPUs after  = %d\n", active_cpus);

        rc = sched_setaffinity(pid, sizeof(cpu_set_t), &new_mask);
        if (rc) {
            printf("sched_getaffinity returned %d\n", rc);
            abort();
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    10, 10, 10, 1.0, A, 10, B, 10, 1.0, C, 10); 

        printf("AFTER\n");

    }
    return 0;
}
