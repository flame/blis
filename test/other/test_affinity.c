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
