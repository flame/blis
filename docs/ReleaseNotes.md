# Release Notes

*Note: For some releases, credit for individuals' contributions are shown in parentheses.*

## Contents

* [Changes in 0.5.1](ReleaseNotes.md#changes-in-051)
* [Changes in 0.5.0](ReleaseNotes.md#changes-in-050)
* [Changes in 0.4.1](ReleaseNotes.md#changes-in-041)
* [Changes in 0.4.0](ReleaseNotes.md#changes-in-040)
* [Changes in 0.3.2](ReleaseNotes.md#changes-in-032)
* [Changes in 0.3.1](ReleaseNotes.md#changes-in-031)
* [Changes in 0.3.0](ReleaseNotes.md#changes-in-030)
* [Changes in 0.2.2](ReleaseNotes.md#changes-in-022)
* [Changes in 0.2.1](ReleaseNotes.md#changes-in-021)
* [Changes in 0.2.0](ReleaseNotes.md#changes-in-020)
* [Changes in 0.1.8](ReleaseNotes.md#changes-in-018)
* [Changes in 0.1.7](ReleaseNotes.md#changes-in-017)
* [Changes in 0.1.6](ReleaseNotes.md#changes-in-016)
* [Changes in 0.1.5](ReleaseNotes.md#changes-in-015)
* [Changes in 0.1.4](ReleaseNotes.md#changes-in-014)
* [Changes in 0.1.3](ReleaseNotes.md#changes-in-013)
* [Changes in 0.1.2](ReleaseNotes.md#changes-in-012)
* [Changes in 0.1.1](ReleaseNotes.md#changes-in-011)
* [Changes in 0.1.0](ReleaseNotes.md#changes-in-010)
* [Changes in 0.0.9](ReleaseNotes.md#changes-in-009)
* [Changes in 0.0.8](ReleaseNotes.md#changes-in-008)
* [Changes in 0.0.7](ReleaseNotes.md#changes-in-007)
* [Changes in 0.0.6](ReleaseNotes.md#changes-in-006)
* [Changes in 0.0.5](ReleaseNotes.md#changes-in-005)
* [Changes in 0.0.4](ReleaseNotes.md#changes-in-004)
* [Changes in 0.0.3](ReleaseNotes.md#changes-in-003)
* [Changes in 0.0.2](ReleaseNotes.md#changes-in-002)
* [Changes in 0.0.1](ReleaseNotes.md#changes-in-001)

## Changes in 0.5.1
December 18, 2018

Improvements present in 0.5.1:

Framework:
- Added mixed-precision support to the 1m method implementation.
- Track internal scalar datatypes in the `obj_t` info bitfield. This allows slightly better handling of scalars during mixed-datatype `gemm` computation.
- Fixed a bug that allowed execution of 1m with mixed-precision `gemm`, despite such usage not yet being officially supported. (Devangi Parikh)
- Added missing internal calls to `bli_init_once()` in `bli_thread_set_num_threads()` and `bli_thread_set_ways()`. (Ali Emre Gülcü)

Kernels:
- Redefined `packm` kernels to handle edge cases and zero-filling, and updated their APIs accordingly. This was needed in order to fully support the use of non-default/non-reference packm kernels. (Devin Matthews)

Build system:
- Disallow explicit requests to use 64-bit integers in the BLAS API while simultaneously using 32-bit integers in the BLIS API. (Jeff Hammond, Devin Matthews)
- Fixed an msys2/Windows build failure. (Isuru Fernando, Costas Yamin)
- Fixed a MinGW build failure. (Isuru Fernando)
- Disabled `arm32`, `arm64` configuration families since we don't yet have logic to choose the correct context at runtime.

Testing:
- Make sure the testsuite fails for `NaN`, `Inf` in input operands. (Devin Matthews)
- Added `hemm` driver to `test/3m4m`.
- Minor updates to `test/mixeddt` drivers, matlab scripts.
- Added additional matlab plotting scripts to `test/3m4m`.

Documentation:
- Updated `docs/Multithreading.md` to include discussion of setting affinity via OpenMP.
- Updated `docs/Testsuite.md` to include discussion of mixed-datatype settings.
- Updated `docs/MixedDatatypes.md` to include a brief section on running the testsuite to exercise mixed-datatype functionality, and other minor updates.
- Fixed broken links in `docs/KernelsHowTo.md`. (Richard Goldschmidt)
- Spelling fixes in FAQ. (Rhys Ulerich)
- Updated 3-clause license comment blocks to refer generically to copyright holders rather than just the original copyright holder, UT-Austin.

## Changes in 0.5.0
October 25, 2018

Improvements present in 0.5.0:

Framework:
- Implemented support for matrix operands of mixed datatypes (domains and precisions) within the `gemm` operation.
- Added configure-time option to use slab or round-robin partitioning within JR and IR loops of most level-3 operations' macrokernels.
- Allow parallelism in the JC loop for `trsm_l`, which previously was unnecessarily disabled. (Field Van Zee, Devangi Parikh)
- Added Fortran-77/90-compatible APIs for some thread-related functions. (Kay Dewhurst)
- Defined a new level-1d operation `shiftd`, which adds a scalar value to every element along an arbitrary diagonal of a matrix.
- Patched an issue (#267) that may arise when linking against OpenMP-configured BLIS from which parallelism is requested at runtime and a level-3 operation (e.g. `gemm`) is called from within an OpenMP parallel region of an application where OpenMP nested parallelism is disabled. (Devin Matthews)

Kernels:
- Imported SkylakeX `dgemm` microkernel from `skx-redux` branch, which contains optimizations (mostly better prefetching on C) over the previous implementation. (Devin Matthews)
- Renamed/relocated level-3 `zen` microkernels to the `haswell` kernel set. Please see a recent message to blis-devel for more information on this rename [1].
- BG/Q kernel fixes. (Ye Luo)

Build system:
- Added support for building Windows DLLs via AppVeyor [2], complete with a built-in implementation of pthreads for Windows, as well as an implementation of the `pthread_barrier_*()` APIs for use on OS X. (Isuru Fernando, Devin Matthews, Mathieu Poumeyrol, Matthew Honnibal)
- Defined a `cortexa53` sub-configuration, which is similar to `cortexa57` except that it uses slightly different compiler flags. (Mathieu Poumeyrol)
- Added python version checking to configure script.
- Added a script to automate the regeneration of the symbols list file (now located in `build/libblis-symbols.def`).
- Various tweaks in preparation for BLIS's inclusion within Debian. (M. Zhou)
- Various fixes and cleanups.

Testing:
- Added tests for `cortexa15` and `cortexa57` in Travis CI. (Mathieu Poumeyrol)
- Added tests for mixed-datatype `gemm` and the simulation of application-level threading (salt) in Travis CI.
- Add statistics-collecting `irun.py` script.
- Include various threading parameters in the initial comment block of testsuite output.
- Various fixes and cleanups.

Documentation:
- Added `MixedDatatypes.md` documentation for mixed-datatype `gemm`.
- Added example code demonstrating use of mixed-datatype `gemm` (object API only).
- Added description of `shiftd` to `BLISTypedAPI.md` and `BLISObjectAPI.md`.
- Added "Known issues" sections to `Multithreading.md` and `Sandboxes.md`.
- Updated `FAQ.md`.
- Various other documentation updates.

[1] https://groups.google.com/forum/?fromgroups#!topic/blis-devel/pytWRjIzxVY
[2] https://ci.appveyor.com/project/shpc/blis/

## Changes in 0.4.1
August 30, 2018

Improvements present in 0.4.1:

Framework:
- Improved thread safety by homogenizing all critical sections to unconditionally use pthread mutexes. (AMD)
- Fixed `bli_finalize()`, which had become uncallable due to sharing `pthread_once_t` objects between the initialization and finalization steps. This manifested as a rather large memory leak (many megabytes) if/when the application manually finalized BLIS in the middle of its execution. (Devangi Parikh, Field Van Zee)
- Fixed a minor memory leak in the global kernel structure. (Devangi Parikh, Field Van Zee)
- Replaced extensive use of function "chooser" macros in object API functions with use of a new set of functions using the suffix `_qfp()` ("query function pointer"). These functions can be used to query function pointers for most families of typed functions.
- Fixed an obscure integer size bug due to improper use of integer literal constants with `va_arg()`. This oddly manifested as LP64 systems using the general stride output case of microkernels even when the output matrix storage matched that of the microkernel output preference. (Devangi Parikh, Field Van Zee)

Kernels:
- Fixed compilation of `armv7a` kernels. (Mathieu Poumeyrol)

Build system:
- Generate makefile fragments within the `obj` directory rather than in `config`, `kernels`, `ref_kernels`, and `frame`. This allows a user to perform an out-of-tree build even if the BLIS source distribution is read-only. (Devin Matthews)
- Allow a dependent sub-project such as example code or the testsuite to compile and link against an installation of BLIS rather than implicitly searching for a local (uninstalled) copy. (Victor Eijkhout, Field Van Zee)
- Fixed a link error that manifested after building only a shared library (e.g. `--disable-static`) and then trying to build a dependent sub-project such as example code or the testsuite. (Sajid Ali)
- Changed `test` make target of top-level `Makefile` to behave more like `check` by printing a color-coded characterization of the test results.
- Fixed the `-p` option to `configure`, which had likely been broken since May 7, 2018. The `--prefix` option was unaffected. (Dave Love)
- Running `configure` no longer requires a C++ compiler given that a C++ compiler was only ever envisioned for *optional* use in the sandbox. (Devangi Parikh, Field Van Zee)

Testing:
- Added the ability to "simulate" multiple application-level threads in the testsuite by executing the individual experiments with multiple threads. This should make it easier to test for thread-safety in the future. (AMD)
- Removed borderline useless wall clock time from test drivers' output.

Documentation:
- Updated typed and object API documents to include language on `rntm_t` parameters in the expert interfaces.
- Updates to `README.md`, including language on sandboxes.
- Added table of make targets to `BuildSystem.md`.
- Added missing language to `ConfigurationHowTo.md` on updating the architecture string array in `bli_arch.c`. (Devangi Parikh, Field Van Zee)

## Changes in 0.4.0
July 27, 2018

Framework:
- Added support for "sandboxes" for employing alternative `gemm` implementations. A ready-to-use reference C99 sandbox provides developers with a starting point for experimentation.
- Separated expert, non-expert typed APIs (levels 1v, 1d, 1f, 1m, 2, and 3, and utility functions).
- Defined new `rntm_t` structure and API to provide a uniform way of storing user-level threading information (equivalent of `BLIS_NUM_THREADS` and `BLIS_*_NT` environment variables), and also conveying that information to expert APIs. (Matthew Honnibal, Nathaniel Smith)
- Renamed various `obj_t` accessor macros, converted to static functions, and inserted explicit typecasting to facilitate #including blis.h from a C++ application. (Jacob Gorm Hansen)
- Cache and reuse `arch_t` architecture query result at runtime. (Devin Matthews)
- Implemented object-based functions `bli_projm()`/`_projv()`, which project objects from one domain to another (within the same precision), and `bli_castm()`/`_castv()`, which typecast objects from one datatype to another.
- Implemented object-based functions `bli_setrm()`/`_setrv()`, `bli_setim()`/`_setiv()`, which allow the caller to broadcast a scalar to all real elements or all imaginary elements within an object.
- Enforce consistent datatypes in most object APIs.
- For native execution, initialize a context's virtual microkernel slots to the function pointers of native microkernels. This simplifies query routines and paves the way for more generalized use of virtual microkernels beyond those for induced methods.
- Various bugfixes. (Devangi Parikh)

Kernels:
- Re-expressed x86_64 microkernels in terms of assembly language macros, which support lower- and upper-case, AT&T and Intel syntax. (Devin Matthews)
- Various bugfixes. (Robin Christ, Francisco Igual, Devangi Parikh, qnerd)

Build system:
- Added support for `--libdir`, `--includedir` configure options. (Nico Schlömer)
- Adopted Linux-like shared library versioning and enabled building shared libraries by default.
- Improved shared library handling on OS X. (Alex Arslan)
- Added configure support for preset `CFLAGS`, `LDFLAGS`. (Dave Love)
- Improvements to version file handling.
- Implemented configure option hack for circumventing small/limited values of `ARG_MAX`.
- Reorganized `cc`, `cc_vendor` detection responsibilities from `Makefile` to `configure`. (Alex Arslan)
- Cross-compilation fixes.
- Preliminary Windows ABI suport using `clang`, appveyor. (Isuru Fernando)
- Better support for typical development environment on OpenBSD, FreeBSD. (Alex Arslan)
- Bumped shared library `soname` version number to 1.0.0.
- Various build system fixes and cleanups. (Mathieu Poumeyrol, Nico Schlömer, Tony Skjellum)

Testing:
- Rewrote Travis CI testing config file and supporting logic to use Intel's SDE emulator. This allows multiple x86_64 microarchitectures to be tested regardless of what hardware Travis happens to be using at the time. (Devin Matthews)
- Added `docs/studies` hardware-specific test driver directory to track individual performance studies. (Devangi Parikh)
- Streamlined `testsuite/input.operations` file format.

Documentation:
- Relocated all wiki documents to a `docs` directory and adjusted all links, and `README.md`, accordingly.
- Added a `CONTRIBUTING.md` file to top-level directory.
- Added `docs/CodingConventions.md`.
- Added `docs/Sandboxes.md`.
- Added `docs/BLISObjectAPI.md`.
- Renamed and updated `docs/BLISTypedAPI.md`.
- Updated `docs/KernelsHowTo.md`.
- Updated `docs/BuildSystem.md`. (Stefanos Mavros)
- Updated `docs/Multithreading.md`.
- Updated indentation in `docs/ConfigurationHowTo.md` for easier reading.
- Added example code for the BLIS typed API in `examples/tapi`.
- Expanded existing example code for the object API in `examples/oapi`.
- Added links to RHEL/Fedora and Debian packages to `README.md`.
- Various cleanups. (Tony Skjellum, Dave Love, Nico Schlömer)

## Changes in 0.3.2
April 28, 2018

- Added `setijm`, `getijm` operations for updating and querying individual matrix elements via the object API.
- Added `examples/oapi` directory containing a code-based tutorial on using the object-based API in BLIS.
- Track separate reference kernel `CFLAGS` for each sub-configuration.
- Added support for blacklisting sub-configurations based on the assembler/binutils.
- Added 64-bit support to BLAS test drivers.
- Various bugfixes.

## Changes in 0.3.1
April 4, 2018

- Enable use of new zen kernels in haswell sub-configuration.
- Added row-storage optimizations to zen `dotxf` kernels (now also used by haswell).
- Integrated an `f2c`ed version of the BLAS test drivers from netlib LAPACK into BLIS build system (e.g. `make testblas`, `make checkblas`). See the [Testsuite](Testsuite.md) document for more info. Also scheduled these BLAS drivers to execute regularly via Travis CI.
- Added a new `make check` target that executes a fast version of the BLIS testsuite as well as the BLAS test drivers (primarily targeting package maintainers).
- Allow individual operation overriding in the BLIS testsuite. (This makes it easy to quickly test one or two operations of interest.)
- Added build system support for libmemkind. If present, `hbw_malloc()` is used as the default value for `BLIS_MALLOC_POOL` instead of `malloc()`. It can be disabled via `--disable-memkind`.
- Tweaks and fixes to BLAS compatibility layer, courtesy of the new BLAS test drivers.
- Output the active sub-configuration in testsuite output header.
- Allow arbitrary nesting of "umbrella" configuration families in `config_registry`, allowing us to define x86_64 in terms of amd64 and intel64.
- Added skx and knl to intel64 (and by proxy, x86_64) configuration families.
- Implemented basic support for ARM hardware detection (via `/proc/cpuinfo`).
- Various bugfixes.

## Changes in 0.3.0
February 23, 2018

This version contains significant improvements from 0.2.2. Major changes include:
- Real and complex domain (s,d,c,z) assembly-based gemm microkernels for AMD's Zen microarchitecture. (AMD, Field Van Zee)
- Real domain (s,d) assembly-based `gemmtrsm_l` and `gemmtrsm_u` microkernels for Zen. (AMD, Field Van Zee)
- Real domain (s,d) intrinsics-based `amaxv`, `axpyv`, `dotv`, `dotxv`, `scalv`, `axpyf`, and `dotxf` kernels for Zen. (AMD, Field Van Zee)
- Generalized the configuration system to allow multi-configuration builds targeting configuration "families". A single sub-configuration is chosen at runtime via some heuristic, such as querying CPUID (e.g. runtime hardware detection). This change was extensive and required a reorganization of the build system, configuration semantics, reference kernels, a new naming scheme for native kernels, and a rewrite of the global kernel structure (gks). Please see the rewritten [Configuration Guide](ConfigurationHowTo.md) for details.
- Implemented runtime hardware detection for x86_64 hardware.
- Reimplemented configure-time hardware detection in terms of new runtime hardware detection code, which queries for CPU features rather than individual models.
- Implemented library self-initialization by rewriting `bli_init()` in terms of `pthread_once()` and inserting invocations to `bli_init()` in key places throughout BLIS. The expectation is that through normal use of any BLIS API (BLAS, typed BLIS, or object-based BLIS), the user no longer needs to explicitly initialize the library, and that `bli_finalize()` should never be called by the user unless he is absolutely sure he no longer needs BLIS functionality. Related to this: global scalar constants (`BLIS_ONE`, `BLIS_ZERO`, etc.) are now statically initialized and thus ready to use immediately. Collectively, these changes provide improved thread safety at the application level.
- Compile with and install a single monolithic (flattened) `blis.h` header to (1) speed up compilation and (2) reduce the number of build product files.
- Added a sub-API for setting multithreading environment variables at runtime. For a few examples, please see the [Multithreading](Multithreading.md) guide.
- Reimplemented OpenMP/pthread barriers in terms of GNU atomic built-ins.
- Other small changes and fixes.

## Changes in 0.2.2
May 2, 2017

- Implemented the 1m method for inducing complex matrix multiplication. (Please see ACM TOMS publication ["Implementing high-performance complex matrix multiplication via the 1m method"](https://github.com/flame/blis#citations) for more details.)
- Switched to simpler `trsm_r` implementation.
- Relaxed constraints that `MC % NR = 0` and `NC % MR = 0`, as this was only needed for the more sophisticated `trsm_r` implementation.
- Automatic loop thread assignment. (Devin Matthews) 
- Updates to `.travis.yml` configuration file. (Devin Matthews) 
- Updates to non-default haswell micro-kernels.
- Match storage format of the temporary micro-tiles in macro-kernels to that of the micro-kernel storage preference for edge cases.
- Added support for Intel's Knight's Landing. (Devin Matthews) 
- Added more flexible options to specify multithreading via the configure script. (Devin Matthews) 
- OS X compatibility fixes. (Devin Matthews) 
- Other small changes and fixes. 

Also, thanks to Elmar Peise, Krzysztof Drewniak, and Francisco Igual for their contributions in reporting/fixing certain bugs that were addressed in this version. 

## Changes in 0.2.1
October 5, 2016

- Implemented distributed `thrinfo_t` structure management. (Ricardo Magana)
- Redesigned BLIS's level-3 algorithmic control tree structure. (suggested by Tyler Smith)
- Consolidated `gemm`, `herk`, and `trmm` blocked variants into one set of three bidirectional variants.
- Integrated a new "memory broker" (`membrk_t`) abstraction in place of the previous memory allocator, which allows one set of pools per broker (or, in other words, per memory space). (Ricardo Magana)
- Reorganized multithreading APIs, including more consistent namespace prefixes: `bli_thrinfo_*()`, `bli_thrcomm_*()`, etc.
- Added `randnm`, `randnv` operations, which produce random powers of two in a narrow range, and integrated a corresponding option into the testsuite. (suggested by AMD)
- Reclassified `amaxv` as a level-1v operation and kernel.
- Added complex `gemm` micro-kernels for haswell, which have register allocations consistent with the existing 6x16 `sgemm` and 6x8 `dgemm` micro-kernels.
- Adjusted existing micro-kernels to work properly when BLIS is configured to use 32-bit integers. (Devin Matthews)
- Relaxed alignment constraints in sandybridge and haswell micro-kernels. (Devin Matthews)
- Define CBLAS API with `f77_int` instead of `int`, which means the BLAS compatibility integer size is inherited by the CBLAS compatibility layer. (Devin Matthews)
- Added an alignment switch to the testsuite to globally enable/disable starting address and leading dimension alignment. (suggested by Devin Matthews)
- Various enhancements to configure script. (Devin Matthews)
- Avoid compiling BLAS/CBLAS compatibility layer when it is disabled via configure. (suggested by Devin Matthews)
- Disabled compilation of object-based blocked partitioning code for level-2 operations, as it was already functionally disabled.
- Fixes and tweaks to POSIX thread support. (Tyler Smith, Jeff Hammond)
- Other small changes and fixes.

## Changes in 0.2.0
April 11, 2016

Most of BLIS 0.2.0's changes are contained within a single commit, 537a1f4 (aka "the big commit"). An executive summary of the most consequential of these changes follows:

- BLIS has been retrofitted with a new data structure, known as a "context," affecting virtually every internal API for every computational operation, as well as many supporting, non-computational functions that must access information within the context.
- In addition to appearing within these internal APIs, the context--specifically, a pointer to a `cntx_t`--is now present within all user-level datatype-aware APIs, e.g. `bli_zgemm()`, appearing as the last argument.
- User-level object APIs, e.g. `bli_gemm()`, were unaffected and continue to be "context-free." However, these APIs were duplicated so that corresponding "context-aware" APIs now also exist, differentiated with an `_ex` suffix (for "expert").
- Contexts are initialized very soon after a computational function is called (if one was not passed in by the caller) and are passed all the way down the function stack, even into the kernels, and thus allow the code at any level to query information about the runtime instantiation of the current operation being executed, such as kernel addresses, micro-kernel storage preferences, and cache/register blocksizes.
- Contexts are thread-friendly. For example, consider the situation where a developer wishes two or more threads to execute simultaneously with somewhat different runtime parameters. Contexts also inherently promote thread-safety, such as in the event that the original source of the information stored in the context changes at run-time (see next two bullets).
- BLIS now consolidates virtually all kernel/hardware information in a new "global kernel structure" (gks) API. This new API will allow the caller to initialize a context in a thread-safe manner according to the currently active kernel configuration. For now, the currently active configuration cannot be changed once the library is built. However, in the future, this API will be expanded to allow run-time management of kernels and related parameters.
- The most obvious application of this new infrastructure is the run-time detection of hardware (and the implied selection of appropriate kernels). With contexts, kernels may even be "hot swapped" within the gks, and once execution begins on a level-3 operation, the memory allocator will be reinitialized on-the-fly, if necessary, to accommodate the new kernels' blocksizes. If a different application thread is executing with another (previously loaded) kernel, it will finish in a deterministic fashion because its kernel info was loaded into its context before computation began, and also because the blocks it checked out from the memory pools will be unaffected by the newer threads' reinitialization of the allocator.

This version contains other changes that were committed prior to 537a1f4:

- Inline assembly FMA4 micro-kernels for AMD bulldozer. (Etienne Sauvage)
- A more feature-rich configure script and build system. Certain long-style options are now accepted, including convenient command-line switches for things like enabling debugging symbols. Important definitions were also consolidated into a new makefile fragment, `common.mk`, which can be included by the BLIS build system as well as quasi-independent build systems, such as the BLIS test suite. (Devin Matthews)
- Updated and improved armv8 micro-kernels. (Francisco Igual)
- Define `bli_clock()` in terms of `clock_gettime()` intead of `gettimeofday()`, which has been languishing on my to-do list for years, literally. (Devin Matthews)
- Minor but extensive modifications to parts of the BLAS compatibility layer to avoid potential namespace conflicts with external user code when `blis.h` is included. (Devin Matthews)
- Fixed a missing BLIS integer type definition (`BLIS_BLAS2BLIS_INT_TYPE_SIZE`) when CBLAS was enabled. Thanks to Tony Kelman reporting this bug.
- Merged `packm_blk_var2()` into `packm_blk_var1()`. The former's functionality is used by induced methods for complex level-3 operations. (Field Van Zee)
- Subtle changes to treatment of row and column strides in `bli_obj.c` that pertain to somewhat unusual use cases, in an effort to support certain situations that arise in the context of tensor computations. (Devin Matthews)
- Fixed an unimplemented `beta == 0` case in the penryn (formerly "dunnington") `sgemm` micro-kernel. (Field Van Zee)
- Enhancements to the internal memory allocator in anticipation of the context retrofit. (Field Van Zee)
- Implemented so-called "quadratic" matrix partitioning for thread-level parallelism, whereby threads compute thread index ranges to produce partitions of roughly equal area (and thus computation), subject to the (register) blocksize multiple, even when given a structured rectangular subpartition with an arbitrary diagonal offset. Thanks to Devangi Parikh for reporting bugs related to this feature. (Field Van Zee)
- Enabled use of Travis CI for automatic testing of github commits and pull requests. (Xianyi Zhang)
- New `README.md`, written in github markdown. (Field Van Zee)
- Many other minor bug fixes.

Special thanks go to Lee Killough for suggesting the use of a "context" data structure in discussions that transpired years ago, during the early planning stages of BLIS, and also for suggesting such a perfectly appropriate name.

## Changes in 0.1.8
July 29, 2015

This release contains only two commits, but they are non-trivial: we now have configuration support for AMD Excavator (Carrizo) and micro-kernels for Intel Haswell/Broadwell.

## Changes in 0.1.7
June 19, 2015

- Replaced the static memory allocator used to manage internal packing buffers with one that dynamically allocates memory, on-demand, and then recycles the allocated blocks in a software cache, or "pool". This significantly simplifies the memory-related configuration parameter set, and it completely eliminates the need to specify a maximum number of threads.
- Implemented default values for all macro constants previously found in `bli_config.h`. The default values are now set in `frame/include/bli_config_macro_defs.h`. Any value #defined in `bli_config.h` will override these defaults.
- Initial support for configure-time detection of hardware. By specifying the `auto` configuration at configure-time, the configure script chooses a configuration for you. If an optimized configuration does not exist, the reference implementation serves as a fallback.
- Completely reorganized implementations for complex induced methods and added support for new algorithms.
- Added optimized micro-kernels for AMD Piledriver family of hardware.
- Several bugfixes to multithreaded execution.
- Various other minor tweaks, code reorganizations, and bugfixes.

## Changes in 0.1.6
October 23, 2014

- New complex domain AVX micro-kernels are now available and used by default by the sandybridge configuration.
- Added new high-level 4m and 3m implementations presently known as "4mh" and "3mh".
- Cleaned up 4m/3m front-end layering and added routines to enable, disable, and query which implementation will be called for a given level-3 operation. The test suite now prints this information in its pre-test summary. 4m (not 4mh) is still the default when complex micro-kernels are not present.
- Consolidated control tree code and usage so that all level-3 multiplication operations use the same gemm_t structure, leaving only `trsm` to have a custom tree structure and associated code.
- Re-implemented micro-panel alignment, which was removed in commit c2b2ab6 earlier this year.
- Relaxed the long-standing constraint that `KC` be a multiple of `MR and `NR` by allowing the developer to specify target values and then adjusting them up to the next multiple of `MR` or `NR`, as needed by the affected operations (`hemm`, `symm`, `trmm`, trsm`).
- Added a new "row preference" flag that the developer can use to signal to the framework that a micro-kernel prefers to output micro-tiles of C that are row-stored (rather than column-stored). Column storage preference is still the default.
- Changed semantics of blocksize extensions to instead be "maximum" blocksizes (and thus emphasizing the "extended" values rather than the difference).
- Various other minor tweaks, code reorganizations, and bugfixes.

Thanks go to those whose contributions, feedback, and bug reports led to these improvements--in particular, Tony Kelman, Kevin Locke, Devin Matthews, Tyler Smith, and perhaps others whose feedback I've lost track of.

## Changes in 0.1.5
August 4, 2014

- Added a CBLAS compatibility layer, which can be enabled at configure-time via `BLIS_ENABLE_CBLAS` in `bli_config.h`. Enabling the CBLAS layer implicitly forces the BLAS compatibility layer to also be enabled. Once enabled, the application may access CBLAS prototypes via `blis.h` or `cblas.h`.
- Fixed a packing bug for cases when `MR` or `NR` (or both) are 1.
- Redefined bit field macros in `bli_type_defs.h` with bitshift operator to ease future rearranging, expanding, or adding of info bits.

## Changes in 0.1.4
July 27, 2014

- Added shared library support to build system.
- Preliminary parallelization of `trsm` (Tyler Smith).
- Added generic `_void()` micro-kernel wrappers so that users (or developers) can call the micro-kernel without knowing the implementation/developer-specific function names, which are specified at configure-time.
- Added `bli_info_*()` API for querying general information about BLIS, including blocksizes.
- Reimplemented initialization/finalization for thread safety.
- Fixed a possible `Inf`/`NaN` issue in several level-3 operations when beta is zero.
- Minor fixes to BLAS compatibility layer.
- Added initial support for Emscripten (Marat Dukhan).

## Changes in 0.1.3
June 23, 2014

This is a relatively minor release. The changes can be summarized as:
- Added experimental support for PNaCL (Marat Dukhan).
- Fixed aligned memory allocation on Windows (Tony Kelman).
- Fixed missing version string in build products when downloading tarballs/zip files (Field Van Zee). Thanks to Victor Eijkhout for pointing out this bug.

## Changes in 0.1.2
June 2, 2014

Tyler has been hard at work developing and refining extensions to BLIS that provide multithreading support (currently via OpenMP, though POSIX threads may be supported in the future). These extensions enable multithreading within all level-3 operations except for `trsm`. We are pleased to announce that these code changes are now part of BLIS.

## Changes in 0.1.1
February 25, 2014

I. I am excited to announce that BLIS now provides high-performance complex domain support to ALL level-3 operations when ONLY the same-precision real domain equivalent gemm micro-kernel is present and optimized. In other words, BLIS's productivity lever just got twice as strong: optimize the `dgemm` micro-kernel, and you will get double-precision complex versions of all level-3 operations, for free. Same for `sgemm` micro-kernel and single-precision complex.

II. We also now offer complex domain support based on the 3m method, but this support is ONLY accessible via separate interfaces. This separation is a safety feature, since the 3m method's numerical properties are inherently less robust. Furthermore, we think the 3m method, as implemented, is somewhat performance-limited on systems with L1 caches that have less than 8-way associativity.

We plan on writing a paper on (I) and (II), so if you are curious how exactly we accomplish this, please be patient and wait for the paper. :)

III. The second, user-oriented change facilitates a much more developer-friendly configuration system. This "change" actually represents a family of smaller changes. What follows is a list of those changes taken from the git log:
- We now have standard names for reference kernels (levels-1v, -1f and 3) in the form of macro constants. Examples:
      `BLIS_SAXPYV_KERNEL_REF`
      `BLIS_DDOTXF_KERNEL_REF`
      `BLIS_ZGEMM_UKERNEL_REF`
- Developers no longer have to name all datatype instances of a kernel with a common base name; [sdcz] datatype flavors of each kernel or micro-kernel (level-1v, -1f, or 3) may now be named independently. This means you can now, if you wish, encode the datatype-specific register blocksizes in the name of the micro-kernel functions.
- Any datatype instances of any kernel (1v, 1f, or 3) that is left undefined in `bli_kernel.h` will default to the corresponding reference implementation. For example, if `BLIS_DGEMM_UKERNEL` is left undefined, it will be defined to be `BLIS_DGEMM_UKERNEL_REF`.
- Developers no longer need to name level-1v/-1f kernels with multiple datatype chars to match the number of types the kernel WOULD take in a mixed type environment, as in `bli_dddaxpyv_opt()`. Now, one char is sufficient, as in `bli_daxpyv_opt()`.
- There is no longer a need to define an obj_t wrapper to go along with your level-1v/-1f kernels. The framework now provides a `_kernel()` function, as in `bli_axpyv_kernel()`, which serves as the `obj_t` wrapper for whatever kernels are specified (or defaulted to) via `bli_kernel.h`.
- Developers no longer need to prototype their kernels, and thus no longer need to include any prototyping headers from within `bli_kernel.h`. The framework now generates kernel prototypes, with the proper type signature, based on the kernel names defined (or defaulted to) via `bli_kernel.h`.
- If the complex datatype x (of [cz]) implementation of the gemm micro-kernel is left undefined by `bli_kernel.h`, but its same-precision real domain equivalent IS defined, BLIS will enable the automatic complex domain feature described above in (1a) for the datatype x implementations of all level-3 operations, using only the corresponding real domain gemm micro-kernel. If the complex gemm micro-kernel for x IS defined, then all complex level-3 operations will be defined in terms of that micro-kernel.

The net effect of (III) is that your `bli_kernel.h` files can be MUCH simpler and less cluttered. (Extreme example: the reference configuration's `bli_kernel.h` is now completely empty!) I have updated all configurations and kernels that are currently part of BLIS by stripping out unnecessary/outdated definitions and migrating existing definitions to their new names. (If you ever need to reference the complete list of options and macros, please refer to the `bli_kernel.h` inside the template configuration.) Please set aside some time to test and, if necessary, tweak the configurations which you originally developed and submitted. I may have broken some of them. If so, please accept my apologies and contact me for assistance. I will work with you to get them functional again.

The changes mentioned in (I), (II), and (III), along with all other changes since 0.1.0, are included BLIS 0.1.1 (fde5f1fd).

I know these changes may be a little disruptive to some, but I think that most developers will find the new complex functionality very useful, and the new configuration system much easier to use.

## Changes in 0.1.0
November 9, 2013

- Added `sgemm` micro-kernel for dunnington.
- Added `dgemm` micro-kernels and configurations for sandybridge, bgq, mic, power7, piledriver, loonson3a, which were used to gather performance data in our second ACM TOMS paper. Many thanks to Francisco Igual, Tyler Smith, Mike Kistler, and Xianyi Zhang for developing, testing, and contributing these kernels.
- Migrated to signed integer for `dim_t`, `inc_t` (to facilitate calling BLIS from Fortran).
- Added "template" configuration and kernel set for developers to use as a starting point when developing new kernels from scratch.
- Improvements to test suite, including section overrides and standalone level-1f/level-3 kernel modules.
- Improvements to Windows build system (though it may still not yet be functional out-of-the-box). Thanks to Martin Schatz for his help here.
- Removed support for element "duplication" in level-3 macro-kernels.
- Several bug fixes to BLAS compatibility layer. Thanks to Vladimir Sukharev for his numerous bug reports wrt the LAPACK test suite.
- Various other minor bugfixes.

## Changes in 0.0.9
July 18, 2013

- A few algorithmic optimizations and bug fixes to `trmm` and `trsm`.
- Parameter checking in the compatibility layer that mimics netlib BLAS.
- Default use of `stdint.h` types (`int64_t`, `uint64_t` by default).
- Optional (and very much untested) C99 built-in complex type/arithmetic support.

Note that `bli_config.h` has changed since 0.0.8. Added configuration macros are:
```
  #define BLIS_ENABLE_C99_COMPLEX
  #define BLIS_ENABLE_BLAS2BLIS_INT64
  #define PASTEF770(name) // ...
```
The first macro enables C99 built-in complex types. The second causes a Fortran integer to be defined as an int64_t (rather than `int32_t`). The third is a macro to name-mangle a full routine name for Fortran (ie: add an underscore) and should be obtained from `config/reference/bli_config.h`.

## Changes in 0.0.8
June 12, 2013

This version includes several kernel optimizations and bug fixes.

While neither `bli_config.h` nor `bli_kernel.h` has changed formats since 0.0.7, `make_defs.mk` **has** changed, so please update your copy of this file when you git-pull. Specifically, we now define a new `CFLAGS_KERNELS` variable that allows one to use different compiler flags when compiling kernels. It works like this: At compile time, make will use `CFLAGS_KERNELS` to compile any source code that resides in any directory that begins with the name `kernels`. My recommendation is to simply apply this naming convention to the symbolic link to your kernels directory that resides in your configuration directory. Thanks to Tyler for suggesting this change.

## Changes in 0.0.7
April 30, 2013

This version incorporates many small fixes and feature enhancements made during our SC13 collaboration. 

## Changes in 0.0.6
April 13, 2013

Several changes regarding memory alignment were made since 0.0.5, including modifications to `bli_config.h`. Also, this update fixes a few bugs.

## Changes in 0.0.5
March 24, 2013

The most obvious change in this version is the migration to the `bli` function (and source code filename) prefix, from the old `bl2` prefix, as well as a rename of the main BLIS header (`blis2.h` -> `blis.h`). The test suite seems to indicate that the change was successful.

A few other much more minor changes were made, one pertaining to a renamed constant in the `_config.h` file.

## Changes in 0.0.4
March 15, 2013

The changes included in 0.0.4 mostly relate to the contiguous (static) memory allocator. The previous implementation was intended as a temporary solution that would work for benchmarking purposes, until enough other priorities had been tended to that I could go back and do it right.

I began with the assumption that the benefit of packing matrices into contiguous memory is non-negligible and worth the effort. Furthermore, we assume that:
- the only portable way to acquire contiguous memory is to reserve a region of static memory and manage it ourselves;
- the cache blocksizes used for one level-3 operation will be the same as those used for another level-3 operation, since all of them boil down to some form of matrix-matrix multiplication;
- only three types of contiguous memory will ever be needed (for level-3 operations): a block of matrix A, a panel of matrix B, or a panel of matrix C--and the last case is not commonly used;
- when a block or panel is to be acquired from the allocator, the caller knows which of the three types of memory is needed.

Given these assumptions, I was able to come up with an implementation that is simple, easy to understand, and thread-safe (provided you add OpenMP directives to protect the critical sections, which are clearly marked with comments). It can also both allocate and release in O(1) time. And of course, page-alignment is taken care of behind the scenes. So while it is not a generalized solution by any means, I think it will work very well for our purposes.

Also, note that based on the level of the overall matrix multiplication algorithm at which you parallelize, the minimum number of blocks/panels of each type of contiguous memory will vary. For example, if you want all of your threads to work on different iterations of a single rank-k update (via block-panel multiply), the threads share the packed panel of B, but each one needs memory to hold its own packed block of A. Thus, the memory allocator needs to be initialized so that it contains enough memory for at least one panel of B and at least t blocks of A, where t is the number of threads. All of this can be adjusted at configure-time in `bl2_config.h`.

## Changes in 0.0.3
February 22, 2013

The biggest change in this version is that the BLAS-to-BLIS compatibility layer is now available. Virtually every BLAS interface is included, even those corresponding to functionality that BLIS does not implement (such as banded and packed level-2 operations). If the application code attempts to call one of these unimplemented routines, the code aborts with a generic not-yet-implemented error message.

The compatibility layer is enabled via a configuration option in `bl2_config.h`. For now, it is enabled by default (provided you have an up-to-date copy of `bl2_config.h`).

## Changes in 0.0.2
February 11, 2013

Most notably, this version contains the new test suite I've been working on for the last month. 

What is the test suite? It is a highly configurable test driver that allows one to test an arbitrary set of BLIS operations, with an arbitrary set of parameter combinations, and matrix/vector storage formats, as well as whichever datatypes you are interested in. (For now, only homogeneous datatyping is supported, which is what most people want.) You can also specify an arbitrary problem size range with arbitrary increments, and arbitrary ratios between dimensions (or anchor a dimension to a single value), and you can output directly to files which store the output in matlab syntax, which makes it easy to generate performance graphs.

BLIS developers: note that 0.0.2 makes small changes to the configuration files. This new version also contains many bug fixes. (Most of these fixes address bugs which were found using the test suite.)

## Changes in 0.0.1
December 10, 2012

- Added auto-detection of string version (via `git`).
- Wrote basic INSTALL, CHANGELOG, AUTHORS, and CREDITS files.
- Updates to standalone `test` directory `Makefile`.
- Added initial build system
- Various code reorganizations.

