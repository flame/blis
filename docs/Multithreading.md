# Contents

* **[Contents](Multithreading.md#contents)**
* **[Introduction](Multithreading.md#introduction)**
* **[Enabling multithreading](Multithreading.md#enabling-multithreading)**
* **[Specifying multithreading](Multithreading.md#specifying-multithreading)**
  * [Globally via environment variables](Multithreading.md#globally-via-environment-variables)
    * [The automatic way](Multithreading.md#environment-variables-the-automatic-way)
    * [The manual way](Multithreading.md#environment-variables-the-manual-way)
  * [Globally at runtime](Multithreading.md#globally-at-runtime)
    * [The automatic way](Multithreading.md#globally-at-runtime-the-automatic-way)
    * [The manual way](Multithreading.md#globally-at-runtime-the-manual-way)
  * [Locally at runtime](Multithreading.md#locally-at-runtime)
    * [Initializing a rntm_t](Multithreading.md#initializing-a-rntm-t)
    * [The automatic way](Multithreading.md#locally-at-runtime-the-automatic-way)
    * [The manual way](Multithreading.md#locally-at-runtime-the-manual-way)
    * [Using the expert interface](Multithreading.md#locally-at-runtime-using-the-expert-interface)


# Introduction

Our paper [Anatomy of High-Performance Many-Threaded Matrix Multiplication](https://github.com/flame/blis#citations), presented at IPDPS'14, identified 5 loops around the micro-kernel as opportunities for parallelization within level-3 operations such as `gemm`. Within BLIS, we have enabled parallelism for 4 of those loops and have extended it to the rest of the level-3 operations except for `trsm`.

# Enabling multithreading

Note that BLIS disables multithreading by default. In order to extract multithreaded parallelism from BLIS, you must first enable multithreading explicitly at configure-time.

As of this writing, BLIS optionally supports multithreading via either OpenMP or POSIX threads.

To enable multithreading via OpenMP, you must provide the `--enable-threading` option to the `configure` script:
```
$ ./configure --enable-threading=openmp auto
```
In this example, we target the `auto` configuration, which is like asking `configure` to choose the most appropriate configuration based on some detection heuristic (e.g. `cpuid` on x86_64). Similarly, to enable multithreading via POSIX threads (pthreads), specify the threading model as `pthreads` instead of `openmp`:
```
$ ./configure --enable-threading=pthreads auto
```
You can also use the shorthand option for `--enable-threading`, which is `-t`:
```
$ ./configure -t pthreads auto
```
For more complete and up-to-date information on the `--enable-threading` option, simply run `configure` with the `--help` (or `-h`) option:
```
$ ./configure --help
```


# Specifying multithreading

There are three broad methods of specifying multithreading in BLIS:
* [Globally via environment variables](Multithreading.md#globally-via-environment-variables)
* [Globally at runtime](Multithreading.md#globally-at-runtime)
* [Locally at runtime](Multithreading.md#locally-at-runtime) (that is, on a per-call, thread-safe basis)

Within these three broad methods there are two specific ways of expressing a request for parallelism. First, the user may express a single number--the total number of threads, or ways of parallelism, to use within a single operation such as `gemm`. We call this the "automatic" way. Alternatively, the user may express the number of ways of parallelism to obtain within *each loop* of the level-3 operation. We call this the "manual" way. The latter way is actually what BLIS eventually needs before it can perform its multithreading; the former is viable only because we have a heuristic of determing a reasonable instance of the latter when given the former.
This pattern--automatic or manual--holds regardless of which of the three methods is used.

Regardless of which method is employed, and which specific way within each method, after setting the number of threads, the application may call the desired level-3 operation (via either the [typed API](docs/BLISTypedAPI.md) or the [object API](docs/BLISObjectAPI.md)) and the operation will execute in a multithreaded manner. (When calling BLIS via the BLAS API, only the first two (global) methods are available.)

## Globally via environment variables

The most common method of specifying multithreading in BLIS is globally via environment variables. With this method, the user sets one or more environment variables in the shell before launching the BLIS-linked executable.

Regardless of whether you end up using the automatic or manual way of expressing a request for multithreading, note that the environment variables are read (via `getenv()`) by BLIS **only once**, when the library is initialized. Subsequent to library initialization, the global settings for parallelization may only be changed via the [global runtime API](Multithreading.md#globally-at-runtime). If this constraint is not a problem, then environment variables may work fine for you. Otherwise, please consider [local settings](Multithreading.md#locally-at-runtime). (Local settings may used at any time, regardless of whether global settings were explicitly specified, and local settings always override global settings.)

### Environment variables: the automatic way

The automatic way of specifying parallelism entails simply setting the total number of threads you wish BLIS to employ in its parallelization. This total number of threads is captured by the `BLIS_NUM_THREADS` environment variable. You can set this variable prior to executing your BLIS-linked executable:
```
$ export GOMP_CPU_AFFINITY="..."  # optional step when using GNU libgomp.
$ export BLIS_NUM_THREADS=16
$ ./my_blis_program
```
This causes BLIS to automatically determine a reasonable threading strategy based on what is known about the operation and problem size. If `BLIS_NUM_THREADS` is not set, BLIS will attempt to query the value of `OMP_NUM_THREADS`. If neither variable is set, the default number of threads is 1.

**Note:** We *highly* discourage use of the `OMP_NUM_THREADS` environment variable and may remove support for it in the future. If you wish to set parallelism globally via environment variables, please use `BLIS_NUM_THREADS`.

### Environment variables: the manual way

The manual way of specifying parallelism involves communicating which loops within the matrix multiplication algorithm to parallelize and the degree of parallelism to be obtained from each of those loops.

The below chart describes the five loops used in BLIS's matrix multiplication operations.

| Loop around micro-kernel | Environment variable | Direction | Notes       |
|:-------------------------|:---------------------|:----------|:------------|
| 5th loop                 | `BLIS_JC_NT`         | `n`       |             |
| 4th loop                 | _N/A_                | `k`       | Not enabled |
| 3rd loop                 | `BLIS_IC_NT`         | `m`       |             |
| 2nd loop                 | `BLIS_JR_NT`         | `n`       |             |
| 1st loop                 | `BLIS_IR_NT`         | `m`       |             |

**Note**: Parallelization of the 4th loop is not currently enabled because each iteration of the loop updates the same part of the output matrix C. Thus, to safely parallelize it requires either a reduction or mutex locks when updating C.

Parallelization in BLIS is hierarchical. So if we parallelize multiple loops, the total number of threads will be the product of the amount of parallelism for each loop. Thus the total number of threads used is the product of all the values:
`BLIS_JC_NT * BLIS_IC_NT * BLIS_JR_NT * BLIS_IR_NT`.
Note that if you set at least one of these loop-specific variables, any others that are unset will default to 1.

In general, the way to choose how to set these environment variables is as follows: The amount of parallelism from the M and N dimensions should be roughly the same. Thus `BLIS_IR_NT * BLIS_IC_NT` should be roughly equal to `BLIS_JR_NT * BLIS_JC_NT`.

Next, which combinations of loops to parallelize depends on which caches are shared. Here are some of the more common scenarios:
 * When compute resources have private L3 caches (example: multi-socket systems), try parallelizing  the `JC` loop. This means threads (or thread groups) will pack and compute with different row panels from matrix B.
 * For compute resources that have private L2 caches but that share an L3 cache (example: cores on a socket), try parallelizing the `IC` loop. In this situation, threads will share the same packed row panel from matrix B, but pack and compute with different blocks of matrix A.
 * If compute resources share an L2 cache but have private L1 caches (example: pairs of cores), try parallelizing the `JR` loop. Here, threads share the same packed block of matrix A but read different packed micro-panels of B into their private L1 caches. In some situations, parallelizing the `IR` loop may also be effective.

![The primary algorithm for level-3 operations in BLIS](http://www.cs.utexas.edu/users/field/mm_algorithm.png)

## Globally at runtime

If you still wish to set the parallelization scheme globally, but you want to do so at runtime, BLIS provides a thread-safe API for specifying multithreading. Think of these functions as a way to modify the same internal data structure into which the environment variables are read. (Recall that the environment variables are only read once, when BLIS is initialized).

### Globally at runtime: the automatic way

If you simply want to specify an overall number of threads and let BLIS choose a thread factorization automatically, use the following function:
```c
void bli_thread_set_num_threads( dim_t n_threads );
```
This function takes one integer--the total number of threads for BLIS to utilize in any one operation. So, for example, if we call
```c
bli_thread_set_num_threads( 4 );
```
we are requesting that the global number of threads be set to 4. You may also query the global number of threads at any time via
```c
dim_t bli_thread_get_num_threads( void );
```
Which may be called in the usual way:
```c
dim_t nt = bli_thread_get_num_threads();
```

### Globally at runtime: the manual way

If you want to specify the number of ways of parallelism to obtain for each loop, use the following function:
```c
void bli_thread_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir );
```
This function takes one integer for each loop in the level-3 operations. (**Note**: even though the function takes a `pc` argument, it will be ignored until parallelism is supported in the `KC` loop.)
So, for example, if we call
```c
bli_thread_set_ways( 2, 1, 4, 1, 1 );
```
we are requesting two ways of parallelism in the `JC` loop and 4 ways of parallelism in the `IC` loop.
Unlike environment variables, which only allow the user to set the parallelization strategy prior to running the executable, `bli_thread_set_ways()` may be called any time during the normal course of the BLIS-linked application's execution.

## Locally at runtime

In addition to the global methods based on environment variables and runtime function calls, BLIS also offers a local, *per-call* method of requesting parallelism at runtime. This method has the benefit of being thread-safe and flexible; your application can spawn two threads at the application level, with each thread requesting different degrees of parallelism from their respective calls to level-3 BLIS operations.

As with environment variables and the global runtime API, there are two ways to specify parallelism: the automatic way and the manual way. Both ways involve allocating a BLIS-specific object, initializing the object and encoding the desired parallelization, and then passing a pointer to the object into one of the expert interfaces of either the [typed](docs/BLISTypedAPI.md) or [object](docs/BLISObjectAPI) APIs. We provide examples of utilizing this threading object below.

### Initializing a rntm_t

Before specifying the parallelism (automatically or manually), you must first allocate a special BLIS object called a `rntm_t` (runtime). The object is quite small (about 64 bytes), and so we recommend allocating it statically on the function stack:
```c
rntm_t rntm;
```
We **strongly recommend** initializing the `rntm_t`. This can be done in either of two ways.
If you want to initialize it as part of the declaration, you may do so via the default `BLIS_RNTM_INITIALIZER` macro:
```c
rntm_t rntm = BLIS_RNTM_INITIALIZER;
```
Alternatively, you can perform the same initialization by passing the address of the `rntm_t` to an initialization function:
```c
bli_rntm_init( &rntm );
```
As of this writing, BLIS treats a default-initialized `rntm_t` as a request for single-threaded execution.

**Note**: If you choose to **not** initialize the `rntm_t` object, you **must** set its parallelism via either the automatic way or the manual way, described below. Passing a completely uninitialized `rntm_t` to a level-3 operation **will almost surely result in undefined behvaior!**

### Locally at runtime: the automatic way

Once your `rntm_t` is initialized, you may request automatic parallelization by encoding only the total number of threads into the `rntm_t` via the following function:
```c
void bli_rntm_set_num_threads( dim_t n_threads, rntm_t* rntm );
```
As with `bli_thread_set_num_threads()` [discussed previously](Multithreading.md#globally-at-runtime-the-automatic-way), this function takes a single integer. It also takes the address of the `rntm_t` to modify. So, for example, if (after declaring and initializing a `rntm_t` as discussed above) we call
```c
bli_rntm_set_num_threads( 6, &rntm );
```
the `rntm_t` object will be encoded to use a total of 6 threads. 

### Locally at runtime: the manual way

Once your `rntm_t` is initialized, you may manually encode the ways of parallelism for each loop into the `rntm_t` by using the following function:
```c
void bli_rntm_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir, rntm_t* rntm );
```
As with `bli_thread_set_ways()` [discussed previously](Multithreading.md#globally-at-runtime-the-manual-way), this function takes one integer for each loop in the level-3 operations. It also takes the address of the `rntm_t` to modify.
(**Note**: even though the function takes a `pc` argument, it will be ignored until parallelism is supported in the `KC` loop.)
So, for example, if we call
```c
bli_rntm_set_ways( 1, 1, 2, 3, 1, &rntm );
```
we are requesting two ways of parallelism in the `IC` loop and three ways of parallelism in the `JR` loop.

### Locally at runtime: using the expert interfaces

Regardless of whether you specified parallelism into your `rntm_t` object via the automatic or manual method, eventually you must use the data structure when calling a BLIS operation.

Let's assume you wish to call `gemm`. To so do, simply use the expert interface, which takes two additional arguments: a `cntx_t` (context) and a `rntm_t`. For the context, you may simply pass in `NULL` and BLIS will select a default context (which is exactly what happens when you call the basic/non-expert interfaces). Here is an example of such a call:
```c
bli_gemm_ex( &alpha, &a, &b, &beta, &c, NULL, &rntm );
```
This will cause `gemm` to execute and parallelize in the manner encoded by `rntm`.

To summarize, using a `rntm_t` involves three steps:
```c
// Declare and initialize a rntm_t object.
rntm_t rntm = BLIS_RNTM_INITIALIZER;

// Call ONE (not both) of the following to encode your parallelization into
// the rntm_t. (These are examples only--use numbers that make sense for your
// application!)
bli_rntm_set_num_threads( 6, &rntm );
bli_rntm_set_ways( 1, 1, 2, 3, 1, &rntm );

// Finally, call BLIS via an expert interface and pass in your rntm_t.
bli_gemm_ex( &alpha, &a, &b, &beta, &c, NULL, &rntm );
```
Note that `rntm_t` objects may be reused over and over again once they are initialized; there is no need to reinitialize them and re-encode their threading values!

Also, you may pass in `NULL` for the `rntm_t*` parameter of an expert interface. This causes the current global settings to be used.

# Conclusion

Please send us feedback if you have any concerns or questions, or [open an issue](http://github.com/flame/blis/issues) if you observe any reproducible behavior that you think is erroneous. (You are welcome to use the issue feature to start any non-trivial dialogue; we don't restrict them only to bug reports!)

Thanks for your interest in BLIS.

