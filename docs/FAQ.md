## Introduction

Here we attempt to provide some frequently-asked questions about the BLIS framework
project, as well as those we think a new user or developer might ask. If you do not see the answer to your question here, please join and post your question to one of the [BLIS mailing lists](https://github.com/flame/blis#discussion).

## Contents

  * [Why did you create BLIS?](FAQ.md#why-did-you-create-blis)
  * [Why should I use BLIS instead of GotoBLAS / OpenBLAS / ATLAS / MKL / ESSL / ACML / Accelerate?](FAQ.md#why-should-i-use-blis-instead-of-gotoblas--openblas--atlas--mkl--essl--acml--accelerate)
  * [How is BLIS related to FLAME / libflame?](FAQ.md#how-is-blis-related-to-flame--libflame)
  * [Does BLIS automatically detect my hardware?](FAQ.md#does-blis-automatically-detect-my-hardware)
  * [I understand that BLIS is mostly a tool for developers?](FAQ.md#i-understand-that-blis-is-mostly-a-tool-for-developers)
  * [How do I link against BLIS?](FAQ.md#how-do-i-link-against-blis)
  * [Must I use git? Can I download a tarball?](FAQ.md#must-i-use-git-can-i-download-a-tarball)
  * [What is a micro-kernel?](FAQ.md#what-is-a-micro-kernel)
  * [What is a macro-kernel?](FAQ.md#what-is-a-macro-kernel)
  * [What is a context?](FAQ.md#what-is-a-context)
  * [I am used to thinking in terms of column-major/row-major storage and leading dimensions. What is a "row stride" / "column stride"?](FAQ.md#im-used-to-thinking-in-terms-of-column-majorrow-major-storage-and-leading-dimensions-what-is-a-row-stride--column-stride)
  * [What does it mean when a matrix with general stride is column-tilted or row-tilted?](FAQ.md#what-does-it-mean-when-a-matrix-with-general-stride-is-column-tilted-or-row-tilted)
  * [I am not really interested in all of these newfangled features in BLIS. Can I just use BLIS as a BLAS library?](FAQ.md#im-not-really-interested-in-all-of-these-newfangled-features-in-blis-can-i-just-use-blis-as-a-blas-library)
  * [What about CBLAS?](FAQ.md#what-about-cblas)
  * [Can I call the native BLIS API from Fortran-77/90/95/2000/C++/Python?](FAQ.md#can-i-call-the-native-blis-api-from-fortran-7790952000cpython)
  * [Do I need to call initialization/finalization functions before being able to use BLIS from my application?](FAQ.md#do-i-need-to-call-initializationfinalization-functions-before-being-able-to-use-blis-from-my-application)
  * [Does BLIS support multithreading?](FAQ.md#does-blis-support-multithreading)
  * [Does BLIS support NUMA environments?](FAQ.md#does-blis-support-numa-environments)
  * [Does BLIS work with GPUs?](FAQ.md#does-blis-work-with-gpus)
  * [Does BLIS work on (some architecture)?](FAQ.md#does-blis-work-on-some-architecture)
  * [What about distributed-memory parallelism?](FAQ.md#what-about-distributed-memory-parallelism)
  * [Can I build BLIS on Windows / Mac OS X?](FAQ.md#can-i-build-blis-on-windows--mac-os-x)
  * [Can I build BLIS as a shared library?](FAQ.md#can-i-build-blis-as-a-shared-library)
  * [Can I use the mixed domain / mixed precision support in BLIS?](FAQ.md#can-i-use-the-mixed-domain--mixed-precision-support-in-blis)
  * [Who is involved in the project?](FAQ.md#who-is-involved-in-the-project)
  * [Who funded the development of BLIS?](FAQ.md#who-funded-the-development-of-blis)
  * [I found a bug. How do I report it?](FAQ.md#i-found-a-bug-how-do-i-report-it)
  * [How do I request a new feature?](FAQ.md#how-do-i-request-a-new-feature)
  * [Where did you get the photo for the BLIS logo / mascot?](FAQ.md#where-did-you-get-the-photo-for-the-blis-logo--mascot)



### Why did you create BLIS?

Initially, BLIS was conceived as simply "BLAS with a more flexible interface". The original BLIS was written as a wrapper layer around BLAS that allowed generalized matrix storage (i.e., separate row and column strides). We also took the opportunity to implement some complex domain features that were missing from the BLAS (mostly related to conjugating input operands). This "proto-BLIS" was deployed in [libflame](http://shpc.ices.utexas.edu/libFLAME.html) to facilitate cleaner implementations of some LAPACK-level operations.

Over time, we wanted more than just a more flexible interface; we wanted an entire framework from which we could build operations in the BLAS as well as those not present within the BLAS. After this new BLIS framework was created, it turned out that the interface improvements were much less interesting (albeit still of consequence) than some of the framework's other features, and the fact that it allowed developers to rapidly instantiate new BLAS libraries by optimizing only a small amount of code.

### Why should I use BLIS instead of GotoBLAS / OpenBLAS / ATLAS / MKL / ESSL / ACML / Accelerate?

BLIS has numerous advantages to existing BLAS implementations. Many of these advantages are summarized on the [BLIS
homepage](https://github.com/flame/blis#key-features). But here are a few reasons one might choose BLIS over some other implementation of BLAS:
  * BLIS facilitates high performance while remaining very portable. BLIS isolates performance-sensitive code to a micro-kernel which contains only one loop and which, when optimized, accelerates virtually all level-3 operations. Thus, BLIS serves as a powerful tool for quickly instantiating BLAS on new or experimental hardware architectures, as well as a flexible "laboratory" in which to conduct research and experiments.
  * BLIS provides robust multithreading support, allowing symmetric multicore/many-core parallelism via either OpenMP or POSIX threads. It also computes proper load balance for structured matrix subpartitions, regardless of the location of the diagonal, or whether the subpartition is lower- or upper-stored.
  * BLIS supports a superset of BLAS functionality, providing operations omitted from the BLAS as well as some complex domain support that is missing in BLAS operations. BLIS is especially useful to researchers who need to develop and prototype new BLAS-like operations that do not exist in the BLAS.
  * BLIS is backwards compatible with BLAS. BLIS contains a BLAS compatibility layer that allows an application to treat BLIS as if it were a traditional BLAS library.
  * BLIS supports generalized matrix storage, which can be used to express column-major, row-major, and general stride storage.
  * BLIS supports mixed-datatype computation for general matrix multiplication `gemm`, and does so while holding the impact on performance to a relative minimum.
  * BLIS is free software, available under a [new/modified/3-clause BSD license](http://opensource.org/licenses/BSD-3-Clause).

### How is BLIS related to FLAME / `libflame`?

As explained [above](FAQ.md#why-did-you-create-blis?), BLIS was initially a layer within `libflame` that allowed more convenient interfacing to the BLAS. So in some ways, BLIS is a spin-off project. Prior to developing BLIS, [its author](http://www.cs.utexas.edu/users/field/) worked as the primary maintainer of `libflame`. If you look closely, you can also see that the design of BLIS was influenced by some of the more useful and innovative aspects of `libflame`, such as internal object abstractions and control trees. Also, various members of the [SHPC research group](http://shpc.ices.utexas.edu/people.html) and its [collaborators](http://shpc.ices.utexas.edu/collaborators.html) routinely provide insight, feedback, and also contribute code (especially kernels) to the BLIS project.

### Does BLIS automatically detect my hardware?

On certain architectures (most notably x86_64), yes. In order to use auto-detection, you must specify `auto` as your configuration when running `configure` (Please see the BLIS [Build System](BuildSystem.md) guide for more info.) A runtime detection option is also available. (Please see the [Configuration Guide](ConfigurationHowTo.md) for a comprehensive walkthrough.)

If automatic hardware detection is requested at configure-time and the build process does not recognize your architecture, the `generic` configuration is selected.

### I understand that BLIS is mostly a tool for developers?

Yes. In order to achieve high performance, BLIS requires that hand-coded kernels and micro-kernels be written and referenced in a valid [BLIS configuration](ConfigurationHowTo.md). These components are usually written by developers and then included within BLIS for use by others.

The good news, however, is that end-users can use BLIS too. Once the aforementioned kernels are integrated into BLIS, they can be used without any developer-level knowledge. Usually, `./configure auto; make; make install` is sufficient for the typical users with typical hardware.

### How do I link against BLIS?

Linking against BLIS is easy! Most people can link to it as if it were a generic BLAS library. Please see the [Linking against BLIS](BuildSystem.md#linking-against-blis) section of the [Build System](BuildSystem.md) guide.

### Must I use git? Can I download a tarball?

We **strongly encourage** you to obtain the BLIS source code by cloning a `git` repository (via the [git
clone](BuildSystem.md#obtaining-blis) command). The reason for this is that it will allow you to easily update your local copy of BLIS by executing `git pull`.

Tarballs and zip files may be obtained from the [releases](https://github.com/flame/blis/releases) page.

### What is a micro-kernel?

The micro-kernel (usually short for "`gemm` micro-kernel") is the basic unit of level-3 (matrix-matrix) computation within BLIS. It consists of one loop, where each iteration performs a very small outer product to update a very small matrix. The micro-kernel is typically the only piece of code that must be carefully optimized (via vector intrinsics or assembly code) to enable high performance in most of the level-3 operations such as `gemm`, `hemm`, `herk`, and `trmm`.

For a more thorough explanation of the micro-kernel and its role in the overall level-3 computations, please read our [ACM TOMS papers](https://github.com/flame/blis#citations). For API and technical reference, please see the [gemm micro-kernel section](KernelsHowTo.md#gemm-micro-kernel) of the BLIS [Kernels Guide](KernelsHowTo.md).

### What is a macro-kernel?

The macro-kernels are portable codes within the BLIS framework that implement relatively small subproblems within an overall level-3 operation. The overall problem (say, general matrix-matrix multiplication, or `gemm`) is partitioned down, according to cache blocksizes, such that its operands are (1) a suitable size and (2) stored in a special packed format. At that time, the macro-kernel is called. The macro-kernel is implemented as two loops around the micro-kernel.

The macro-kernels in BLIS correspond to the so-called "inner kernels" (or simply "kernels") that formed the fundamental unit of computation in Kazushige Goto's GotoBLAS (and now in the successor library, OpenBLAS).

For more information on macro-kernels, please read our [ACM TOMS papers](https://github.com/flame/blis#citations).

### What is a context?

As of 0.2.0, BLIS contains a new infrastructure for communicating runtime information (such as kernel addresses and blocksizes) from the highest levels of code all the way down the function stack, even into the kernels themselves. This new data structure is called a *context*, and together with its API, it helped us clean up some hacks and other awkwardness that existed in BLIS prior to 0.2.0. Contexts also lays the groundwork for managing kernels and related kernel information at runtime.

If you are a kernel developer, you can usually ignore the `cntx_t*` argument that is passed into each kernel, since the kernels already inherently "know" this information (such as register blocksizes). And if you are a user, and the function you want to call takes a `cntx_t*` argument, you can safely pass in `NULL` and BLIS will automatically build a suitable context for you at runtime. 

### I'm used to thinking in terms of column-major/row-major storage and leading dimensions. What is a "row stride" / "column stride"?

Traditional BLAS assumes that matrices are stored in column-major order, where a leading dimension measures the distance from one element to the next element in the same row. But column-major order is really just a special case of BLIS's more generalized storage scheme.

In generalized storage, we have a row stride and a column stride. The row stride measures the distance in memory between rows (within a single column) while the column stride measures the distance between columns (within a single row). Column-major storage corresponds to the situation where the row stride equals 1. Since the row stride is unit, you only have to track the column stride (i.e., the leading dimension). Similarly, in row-major order, the column stride is equal to 1 and only the row stride must be tracked.

BLIS also supports situations where both the row stride and column stride are non-unit. We call this situation "general stride".

### What does it mean when a matrix with general stride is column-tilted or row-tilted?

When a matrix is stored with general stride, both the row stride and column stride (let's call them `rs` and `cs`) are non-unit. When `rs` < `cs`, we call the general stride matrix "column-tilted" because it is "closer" to being column-stored (than row-stored). Similarly, when `rs` > `cs`, the matrix is "row-tilted" because it is closer to being row-stored.

### I'm not really interested in all of these newfangled features in BLIS. Can I just use BLIS as a BLAS library?

Absolutely. Just link your application to BLIS the same way you would link to a BLAS library. For a simple linking example, see the [Linking to BLIS](KernelsHowTo.md#linking-to-blis) section of the BLIS [Build System](BuildSystem.md) guide.

### What about CBLAS?

BLIS also contains an optional CBLAS compatibility layer, which leverages the BLAS compatibility layer to help map CBLAS function calls to the corresponding functionality in BLIS. Once BLIS is built with CBLAS support, your application can access CBLAS prototypes via either `cblas.h` or `blis.h`. At the time of this writing, CBLAS support is disabled by default, so be sure to enable it at configure-time. Please see `./configure --help` for the syntax for enabling CBLAS.

### Can I call the native BLIS API from Fortran-77/90/95/2000/C++/Python?

In principle, BLIS's native (and BLAS-like) [typed API](BLISTypedAPI) can be called from Fortran. However, you must ensure that the size of the integer in BLIS is equal to the size of integer used by your Fortran program/compiler/environment. The size of BLIS integers is determined at configure-time. Please see `./configure --help` for the syntax for options related to integer sizes.

As for bindings to other languages, please contact the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.

### Do I need to call initialization/finalization functions before being able to use BLIS from my application?

Originally, BLIS did indeed require the application to explicitly setup (initialize) various internal data structures via `bli_init()`. Likewise, calling `bli_finalize()` was recommended to cleanup (finalize) the library. However, since commit 9804adf (circa December 2017), BLIS has implemented self-initialization. These explicit calls to `bli_init()` and `bli_finalize()` are no longer necessary, though experts may still use them in special cases to control the allocation and freeing of resources. This topic is discussed in the BLIS [typed API reference](BLISTypedAPI.md#initialization-and-cleanup).

### Does BLIS support multithreading?

Yes! BLIS supports multithreading (via OpenMP or POSIX threads) for all of its level-3 operations. For more information on enabling and controlling multithreading, please see the [Multithreading](Multithreading.md) guide.

BLIS is also thread-safe so that you can call BLIS from threads within a multithreaded library or application. BLIS derives is thread-safety via unconditional use of features present in POSIX threads (pthreads). These pthreads features are employed for thread-safety regardless of whether BLIS is configured for OpenMP multithreading, pthreads multithreading, or single-threaded execution.

### Does BLIS support NUMA environments?

We have integrated some early foundational support for NUMA *development*, but currently BLIS will execute sub-optimally on NUMA systems. If you are interested in adapting BLIS to a NUMA architecture, please contact us via the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.

### Does BLIS work with GPUs?

BLIS does not currently support graphical processing units (GPUs).

### Does BLIS work on _(some architecture)_?

Please see the BLIS [Hardware Support](HardwareSupport.md) guide for a full list of supported architectures. If your favorite hardware is not listed and you have the expertise, please consider developing your own kernels and sharing them with the project! We will, of course, gratefully credit your contribution.

### What about distributed-memory parallelism?

No. BLIS is a framework for sequential and shared-memory/multicore implementations of BLAS-like operations. If you need distributed-memory dense linear algebra implementations, we recommend the [Elemental](http://libelemental.org/) library.

### Can I build BLIS on Windows / Mac OS X?

BLIS was designed for use in a GNU/Linux environment. However, we've gone to greath lengths to keep BLIS compatible with other UNIX-like systems as well, such as BSD and OS X. System software requirements for UNIX-like systems are discussed in the BLIS [Build System](BuildSystem.md) guide.

Support for building in Windows is not directly supported. However, Windows 10 now provides a Linux-like environment. We suspect this is the best route for those trying to build BLIS in Windows.

If all you need is a Windows DLL of BLIS, you may be in luck! BLIS uses [AppVeyor](https://ci.appveyor.com/) to automatically produces dynamically-linked libraries, which are preserved on the site as "artifacts". To try it out, just visit the [BLIS AppVeyor page](https://ci.appveyor.com/project/shpc/blis/), click on the `LIB_TYPE=shared` link for the most recent build, and then click on "Artifacts". And if you'd like to share your experiences, please join the [blis-devel](http://groups.google.com/group/blis-devel) mailing list and send us a message!

### Can I build BLIS as a shared library?

Yes. By default, most configurations output only a static library archive (e.g. `.a` file). However, you can also request a shared object (e.g. `.so` file), sometimes also called a "dynamically-linked" library. For information on enabling shared library output, simply run `./configure --help`.

### Can I use the mixed domain / mixed precision support in BLIS?

Yes! As of 5fec95b (circa October 2018), BLIS supports mixed-datatype (mixed domain and/or mixed precision) computation via the `gemm` operation. Documentation on utilizing this new functionality is provided via the [MixedDatatype.md](docs/MixedDatatypes.md) document in the source distribution.

If this feature is important or useful to your work, we would love to hear from you. Please contact us via the [blis-devel](http://groups.google.com/group/blis-devel) mailing list and tell us about your application and why you need/want support for BLAS-like operations with mixed-domain/mixed-precision operands.

### Who is involved in the project?

Lots of people! For a full list of those involved, see the
[CREDITS](https://github.com/flame/blis/blob/master/CREDITS) file within the BLIS framework source distribution.

### Who funded the development of BLIS?

BLIS was primarily funded by grants from [Microsoft](https://www.microsoft.com/),
[Intel](https://www.intel.com/), [Texas
Instruments](https://www.ti.com/), [AMD](https://www.amd.com/), [Huawei](https://www.hauwei.com/us/), and [Oracle](https://www.oracle.com/) as well as grants from the [National Science Foundation](http://www.nsf.gov/) (Awards CCF-0917167 ACI-1148125/1340293, and CCF-1320112).

Reminder: _Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF)._

### I found a bug. How do I report it?

If you think you've found a bug, we request that you [open an issue](http://github.com/flame/blis/issues). Don't be shy! Really, it's the best and most convenient way for us to track your issues/bugs/concerns. Other discussions that are not primarily bug-reports should take place via the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.

### How do I request a new feature?

Feature requests should also be submitted by [opening a new issue](http://github.com/flame/blis/issues).

### Where did you get the photo for the BLIS logo / mascot?

The sleeping ["BLIS cat"](https://github.com/flame/blis/blob/master/README.md) photo was taken by Petar Mitchev and is used with his permission.

