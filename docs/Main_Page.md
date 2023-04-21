@mainpage
# Welcome to AOCL-BLIS

---

## Table of Content
    * [Introduction](#Introduction)
    * [Build and Installation](#Build)
    * [Examples](#Example)
    * [Contact Us](#Contact)


<div id="Introduction" name="Introduction"></div>

## Introduction

<b> AOCL BLIS </b> BLIS is a portable software framework for instantiating high-performance BLAS-like dense linear algebra libraries. The framework was designed to isolate essential kernels of computation that, when optimized, immediately enable optimized implementations of most of its commonly used and computationally intensive operations. BLIS is written in ISO C99 and available under a new/modified/3-clause BSD license. While BLIS exports a new BLAS-like API, it also includes a BLAS compatibility layer which gives application developers access to BLIS implementations via traditional BLAS routine calls. An object-based API unique to BLIS is also available.

How to Download BLIS
--------------------

There are a few ways to download BLIS. We list the most common four ways below.
We **highly recommend** using either Option 1 or 2. Otherwise, we recommend
Option 3 (over Option 4) so your compiler can perform optimizations specific
to your hardware.

1. **Download a source repository with `git clone`.**
Generally speaking, we prefer using `git clone` to clone a `git` repository.
Having a repository allows the user to periodically pull in the latest changes
and quickly rebuild BLIS whenever they wish. Also, implicit in cloning a
repository is that the repository defaults to using the `master` branch, which
contains the latest "stable" commits since the most recent release. (This is
in contrast to Option 3 in which the user is opting for code that may be
slightly out of date.)

   In order to clone a `git` repository of BLIS, please obtain a repository
URL by clicking on the green button above the file/directory listing near the
top of this page (as rendered by GitHub). Generally speaking, it will amount
to executing the following command in your terminal shell:
   ```
   git clone https://github.com/amd/blis.git
   ```

2. **Download a source repository via a zip file.**
If you are uncomfortable with using `git` but would still like the latest
stable commits, we recommend that you download BLIS as a zip file.

   In order to download a zip file of the BLIS source distribution, please
click on the green button above the file listing near the top of this page.
This should reveal a link for downloading the zip file.

3. **Download a source release via a tarball/zip file.**
Alternatively, if you would like to stick to the code that is included in
official releases, you may download either a tarball or zip file of any of
BLIS's previous [tagged releases](https://github.com/flame/blis/releases).
We consider this option to be less than ideal for most people since it will
likely mean you miss out on the latest bugfix or feature commits (in contrast
to Options 1 or 2), and you also will not be able to update your code with a
simple `git pull` command (in contrast to Option 1).

4. **Download a binary package specific to your OS.**
While we don't recommend this as the first choice for most users, we provide
links to community members who generously maintain BLIS packages for various
Linux distributions such as Debian Unstable and EPEL/Fedora. Please see the
[External Packages](#external-packages) section below for more information.

Getting Started
---------------

*NOTE: This section assumes you've either cloned a BLIS source code repository
via `git`, downloaded the latest source code via a zip file, or downloaded the
source code for a tagged version release---Options 1, 2, or 3, respectively,
as discussed in [the previous section](#how-to-download-blis).*

If you just want to build a sequential (not parallelized) version of BLIS
in a hurry and come back and explore other topics later, you can configure
and build BLIS as follows:
```
$ ./configure auto
$ make [-j]
```
You can then verify your build by running BLAS- and BLIS-specific test
drivers via `make check`:
```
$ make check [-j]
```
And if you would like to install BLIS to the directory specified to `configure`
via the `--prefix` option, run the `install` target:
```
$ make install
```
Please read the output of `./configure --help` for a full list of configure-time
options.
If/when you have time, we *strongly* encourage you to read the detailed
walkthrough of the build system found in our [Build System](docs/BuildSystem.md)
guide.

Example Code
------------

The BLIS source distribution provides example code in the `examples` directory.
Example code focuses on using BLIS APIs (not BLAS or CBLAS), and resides in
two subdirectories: [examples/oapi](examples/oapi) (which demonstrates the
[object API](docs/BLISObjectAPI.md)) and [examples/tapi](examples/tapi) (which
demonstrates the [typed API](docs/BLISTypedAPI.md)).

Either directory contains several files, each containing various pieces of
code that exercise core functionality of the BLIS API in question (object or
typed). These example files should be thought of collectively like a tutorial,
and therefore it is recommended to start from the beginning (the file that
starts in `00`).

You can build all of the examples by simply running `make` from either example
subdirectory (`examples/oapi` or `examples/tapi`). (You can also run
`make clean`.) The local `Makefile` assumes that you've already configured and
built (but not necessarily installed) BLIS two directories up, in `../..`. If
you have already installed BLIS to some permanent directory, you may refer to
that installation by setting the environment variable `BLIS_INSTALL_PATH` prior
to running make:
```
export BLIS_INSTALL_PATH=/usr/local; make
```
or by setting the same variable as part of the make command:
```
make BLIS_INSTALL_PATH=/usr/local
```
**Once the executable files have been built, we recommend reading the code and
the corresponding executable output side by side. This will help you see the
effects of each section of code.**

This tutorial is not exhaustive or complete; several object API functions were
omitted (mostly for brevity's sake) and thus more examples could be written.

<div id = "Contact"></div>

## CONTACTS

AOCL BLIS is developed and maintained by AMD. You can contact us on the email-id <b>[aoclsupport@amd.com](mailto:aoclsupport@amd.com)</b>
