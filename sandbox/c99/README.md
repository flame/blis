
## Introduction

This file briefly describes the requirements for building a custom BLIS
*sandbox*.

Simply put, a sandbox in BLIS provides an alternative implementation to the
function `bli_gemmnat()`, which is the object-based API call for computing
the gemm operation via native execution. (Native execution simply means that
an induced method will not be used. It's what you probably already think of
when you think of implementing the gemm operation: a series of loops around
an optimized (usually assembly-based) microkernel with some packing functions
thrown in at various levels.)

Why sandboxes? Sometimes, you just want to experiment with tweaks or changes
to the gemm operation, but you want to do so in a simple environment rather
than the somewhat obfuscated and highly macroized and refactored code of the
core framework (which, I will remind everyone, is highly macroized and
refactored mostly so that all floating-point datatypes and all level-3
operations are supported with minimal source code). By building a BLIS sandbox,
you can experiment (within limits) and still benefit from BLIS's existing
build system, testsuite, and toolbox of utility functions.

## Sandbox rules

Like any decent sandbox, there are rules for playing here. Please follow these
guidelines for the best sandbox developer experience.

0. Don't bother worrying about makefiles. We've already taken care of the
boring/annoying/headache-inducing build system stuff for you. :) By configuring
BLIS with a sandbox enabled, `make` will scan your directory and compile all
of its source code using similar compilation rules as were used for the rest
of the framework. In addition, the compilation command line will automatically
contain one `-I<includepath>` option for every subdirectory in your sandbox,
so it doesn't matter where you place your header files. They will be found!

1. Your sandbox must be written in C99 or C++11. If you write your sandbox in
C++11, you must use one of the BLIS-approved file extensions for your source
files (`.cc`, `.cpp`, `.cxx`) and your header files (`.hh`, `.hpp`, `.hxx`).
Note that `blis.h`
already contains all of its definitions inside of an `extern "C"` block, so
you should be able to `#include "blis.h"` from your C++11 source code without
any issues.

2. All of your code to replace BLIS's default implementation of `bli_gemmnat()`
should reside in the sandbox directory, or some directory therein. (Obviously.)
For example, this `README.md` file is located in the **C99** sandbox located in
`sandbox/c99`. Thus, all of the code associated with this sandbox will be
contained within `sandbox/c99`.

3. The *only* header file that is required of your sandbox is `bli_sandbox.h`.
It must be named `bli_sandbox.h` because `blis.h` will `#include` this file
when sandboxes are enabled at configure-time.
That said, you will probably want to keep the file empty. Why require a file
that is supposed to be empty? Well, it doesn't *have* to be empty. Anything
placed in this file will be folded into the flattened (monolithic) `blis.h`
at compile-time. Therefore, you should only place things (e.g. prototypes or
type definitions) in `bli_sandbox.h` if those things would be needed at
compile-time by:
(a) the BLIS framework itself, or
(b) an *application* that uses your sandbox-enabled BLIS library.
Usually, neither of these situations will require any of your local definitions
since those definitions are only needed to define your sandbox implementation
of `bli_gemmnat()`. (Even this function is already prototyped by BLIS.)

4. Your definition of `bli_gemmnat()` should be the *only* function you define
in your sandbox that begins with `bli_`. If you define other functions that
begin with `bli_`, you risk a namespace collision with existing framework
functions. To guarantee safety, please prefix your locally-defined sandbox
functions with another prefix. Here, in the C99 sandbox, we use the prefix
`blx_`. (The `x` is for sandbox. Or experimental. Whatever, it doesn't matter.)
Also, please avoid the prefix `bla_` since that prefix is also used in BLIS for
BLAS compatibility functions.

If you follow these rules, you will likely have a pleasant experience
integrating your BLIS sandbox into the larger framework.

## Caveats

Notice that the BLIS sandbox is not all-powerful. You are more-or-less stuck
working with the existing BLIS infrastructure.

For example, with a BLIS sandbox you **can** do the following kinds of things:
- use a different algorithmic partitioning path from the strategy used by
default in BLIS;
- experiment with different implementations of `packm` kernels;
- try inlining your functions manually;
- pivot away from using `obj_t` objects at higher algorithmic level (such as
immediately after calling `bli_gemmnat()`) to try to avoid some overhead;
- use a locally-defined microkernel that is not registered at configure-time.

You **cannot**, however, do the following kinds of things:
- define new datatypes (half-precision, quad-precision, short integer, etc.)
and expect the rest of BLIS to "know" how to handle them;
- use a sandbox to implement a different level-3 operation, such as Hermitian
rank-k update;
- define a new BLAS-like operation.

Notwithstanding these limitations, hopefully you find BLIS sandboxes useful!

## Questions? Concerns? Feedback?

If you encounter any problems, please open a new
[issue on GitHub](https://github.com/flame/blis/issues).

If you are unsure about how something works, you can still open an issue. Or, you
can send a message to
[blis-devel mailing list](https://groups.google.com/d/forum/blis-devel).

Happy sandboxing!

Field
