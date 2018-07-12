## Contents

* **[Contents](Multithreading.md#contents)**
* **[Introduction](Multithreading.md#introduction)**
* **[Enabling multithreading](Multithreading.md#enabling-multithreading)**
* **[Specifying multithreading](Multithreading.md#specifying-multithreading)**
  * [The automatic way](Multithreading.md#the-automatic-way)
  * [The manual way](Multithreading.md#the-manual-way)

## Introduction

Our paper [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://github.com/flame/blis#citations), presented at IPDPS'14, identified 5 loops around the micro-kernel as opportunities for parallelization. Within BLIS, we have enabled parallelism for 4 of those loops and have extended it to the rest of the level-3 operations except for `trsm`.

## Enabling multithreading

Note that BLIS disables multithreading by default.

As of this writing, BLIS optionally supports multithreading via either OpenMP or POSIX threads.

To enable multithreading via OpenMP, you must provide the `--enable-threading` option to the `configure` script:
```
  $ ./configure --enable-threading=openmp haswell
```
In this example, we configure for the `haswell` configuration. Similarly, to enable multithreading via POSIX threads (pthreads), specify the threading model as `pthreads` instead of `openmp`:
```
  $ ./configure --enable-threading=pthreads haswell
```
For more complete and up-to-date information on the `--enable-threading` option, simply run `configure` with the `--help` (or `-h`) option:
```
  $ ./configure --help
```


## Specifying multithreading

There are two broad ways to specify multithreading in BLIS: the "automatic way" or the "manual way".

### The automatic way

The simplest way to enable multithreading in BLIS is to simply set the total number of threads you wish BLIS to employ in its parallelization. This total number of threads is captured by the `BLIS_NUM_THREADS` environment variable. You can set this variable prior to executing your BLIS-linked executable:

```
  $ export BLIS_NUM_THREADS=16
  $ ./my_blis_program
```
This causes BLIS to automatically determine a reasonable threading strategy based on what is known about your architecture. If `BLIS_NUM_THREADS` is not set, then BLIS also looks at the value of `OMP_NUM_THREADS`, if set. If neither variable is set, the default number of threads is 1.
 
Alternatively, any time after calling `bli_init()` but before `bli_finalize()`, you can also set (or change) the value of `BLIS_NUM_THREADS` at run-time:
```
  bli_thread_set_num_threads( 8 );
```
Similarly, the current value of `BLIS_NUM_THREADS` can always be queried as follows:
```
  dim_t num_threads = bli_thread_get_num_threads();
```

### The manual way

The "manual way" of specifying parallelism in BLIS involves specifying which loops within the matrix multiplication algorithm to parallelize, and the degree of parallelism to be obtained from those loops.

The below chart describes the five loops used in BLIS's matrix multiplication operations. 

| Loop around micro-kernel | Environment variable | Direction | Notes       |
|:-------------------------|:---------------------|:----------|:------------|
| 5th loop                 | `BLIS_JC_NT`         | `n`       |             |
| 4th loop                 | _N/A_                | `k`       | Not enabled |
| 3rd loop                 | `BLIS_IC_NT`         | `m`       |             |
| 2nd loop                 | `BLIS_JR_NT`         | `n`       |             |
| 1st loop                 | `BLIS_IR_NT`         | `m`       |             |

Note: Parallelization of the 4th loop is not currently enabled because each iteration of the loop updates the same part of the matrix C. Thus to parallelize it requires either a reduction or mutex locks when updating C.

Parallelization in BLIS is hierarchical. So if we parallelize multiple loops, the total number of threads will be the product of the amount of parallelism for each loop. Thus the total number of threads used is `BLIS_IR_NT * BLIS_JR_NT * BLIS_IC_NT * BLIS_JC_NT`.

In general, the way to choose how to set these environment variables is as follows: The amount of parallelism from the M and N dimensions should be roughly the same. Thus `BLIS_IR_NT * BLIS_IC_NT` should be roughly equal to `BLIS_JR_NT * BLIS_JC_NT`.

Next, which combinations of loops to parallelize depends on which caches are shared. Here are some of the more common scenarios:
 * When compute resources have private L3 caches (example: multi-socket systems), try parallelizing  the `JC` loop. This means threads (or thread groups) will pack and compute with different row panels from matrix B.
 * For compute resources that have private L2 caches but that share an L3 cache (example: cores on a socket), try parallelizing the `IC` loop. In this situation, threads will share the same packed row panel from matrix B, but pack and compute with different blocks of matrix A.
 * If compute resources share an L2 cache but have private L1 caches (example: pairs of cores), try parallelizing the `JR` loop. Here, threads share the same packed block of matrix A but read different packed micro-panels of B into their private L1 caches. In some situations, parallelizing the `IR` loop may also be effective.

![The primary algorithm for level-3 operations in BLIS](http://www.cs.utexas.edu/users/field/mm_algorithm.png)

As with specifying parallelism via `BLIS_NUM_THREADS`, you can set the `BLIS_xx_NT` environment variables in the shell, prior to launching your BLIS-linked executable, or you can set (or update) the environment variables at run-time. Here are some examples of using the run-time API:
```c
  bli_thread_set_jc_nt( 2 );  // Set BLIS_JC_NT to 2.
  bli_thread_set_jc_nt( 4 );  // Set BLIS_IC_NT to 4.
  bli_thread_set_jr_nt( 3 );  // Set BLIS_JR_NT to 3.
  bli_thread_set_ir_nt( 1 );  // Set BLIS_IR_NT to 1.
```
  There are also equivalent "get" functions that allow you to query the current values for the `BLIS_xx_NT` variables:
```c
  dim_t jc_nt = bli_thread_get_jc_nt();
  dim_t ic_nt = bli_thread_get_ic_nt();
  dim_t jr_nt = bli_thread_get_jr_nt();
  dim_t ir_nt = bli_thread_get_ir_nt();
```

