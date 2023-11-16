## Contents

* **[Contents](CMakeBuildSystem.md#contents)**
* **[Introduction](CMakeBuildSystem.md#introduction)**
* **[Step 1: Chose a framework configuration](CMakeBuildSystem.md#step-1-choose-a-framework-configuration)**
* **[Step 2: Configuring CMake](CMakeBuildSystem.md#step-2-configuring-cmake)**
* **[Step 3: Compilation](CMakeBuildSystem.md#step-3-compilation)**
* **[Step 4: Installation](CMakeBuildSystem.md#step-4-installation)**
* **[Compiling with BLIS](CMakeBuildSystem.md#compiling-with-blis)**
* **[Uninstalling](CMakeBuildSystem.md#uninstalling)**
* **[Available targets](CMakeBuildSystem.md#available-targets)**
* **[Adding configurations](CMakeBuildSystem.md#adding-configurations)**
* **[Some examples](CMakeBuildSystem.md#some-examples)**
* **[Final notes](CMakeBuildSystem.md#final-notes)**

## Introduction

This document describes how to use CMake to build and install a BLIS library to your local system.

The BLIS CMake system is based on the [Make build system](BuildSystem.md) and is designed for use with both Linux and Windows. Other requirements are:

  * CMake (3.15.0 or higher)
  * Python (3.4 or later for python3)
  * GNU `make` (3.81 or later) on Linux
  * Visual Studio 17 2022 on Windows
  * a working C99 compiler (gcc or clang on Linux and **only** clang-cl on Windows)

**_NOTE:_**
To get clang-cl on Visual Studio, one needs to choose "C++ Clang tools for Windows" when installing "Desktop development with C++" with Visual Studio.

Note that, on Windows, BLIS implements basic pthreads functionality automatically, so a POSIX threads library is not required. On Linux, the implementation is the same to the one of the Make system.

CMake is used to build out of source, so we need to start by creating a build directory from which we will do the configuration and build. Since there is a directory called blis/build, the build directory must have a different name. Here is an example of creating the directory:
```
$ mkdir build_blis
$ cd build_blis
```

## Step 1: Choose a framework configuration

The first step is to choose the appropriate BLIS configuration. As on the Make build system, the user must decide which configuration to use or whether automatic hardware detection should be used to determine the configuration. Currently only the following configurations are supported:

  * amdzen
  * zen
  * zen2
  * zen3
  * zen4
  * generic

Instructions on how to add a configuration on the CMake system, are provided in [Adding configurations](CMakeBuildSystem.md#adding-configurations).

### Multithreading

As in Make system, multithreading in BLIS is disabled by default. To configure cmake so that OpenMP is used, please use `-DTHREADING_MODEL=openmp`. All available options can be found if cmake-gui is used, or by running
```
cmake .. -DPRINT_CONFIGURE_HELP=ON
```

## Step 2: Configuring CMake

### Choosing a generator

This is a reminder on how to configure CMake to use a specific generator:
```
cmake -G <generator_of_choice>
```

On Linux "Unix Makefiles" is used by default and `-G <generator_of_choice>` can be omitted.

On Windows, specify Visual Studio generator using
```
cmake -G "Visual Studio 17 2022"
```

For the rest of this documentation, we will use the platform-agnostic commands to build the libraries, but the usual make commands can be used instead. On the following command snippets we ommit specifying the generator, but one can use their prefered way of building using common CMake practices. 

### Choosing a configuration 

This step is equivalent to running `./configure <confname>` using the Make system. In this case, simply run:
```
cmake .. -DBLIS_CONFIG_FAMILY=<confname>
```
If the provided configuration is not supported, an error will be thrown and a message with the available configurations will be printed.

To configure based on your hardware, you can configure using
```
cmake .. -DBLIS_CONFIG_FAMILY=auto
```
Please note that when `auto` is used as a configuration option, the `generic` configuration will be chosen by default on non-AMD hardware.

### Specifying a prefix path for installation

We remind users that to specify the installation prefix in cmake, one needs to configure using `CMAKE_INSTALL_PREFIX` variable:
```
cmake .. -DBLIS_CONFIG_FAMILY=auto -DCMAKE_INSTALL_PREFIX=<prefix>
```
This will cause libraries to eventually be installed to `<prefix>/lib` and headers will be installed to `<prefix>/include`.

Option to specify the library install and the header install separately, like in Make system, is not currently supported by the CMake equivalent.

## Step 3: Compilation

Once configuration is finished and the corresponding platform-dependent build files have been generated, you can proceed to building the library.
To build the library in a platform agnostic way use:
```
cmake --build . --config Release
```
For a verbose build, you can use:
```
cmake --build . --verbose --config Release
```
To build in parallel on a multicore system, you can use:
```
cmake --build . --config Release -j<n>
```
where `<n>` is the number of jobs allowed to run simultaneously by this command.

Note that on Linux, if Makefiles are used, the above is equivalent to running
```
make -j<n>
```

## Step 4: Installation

The BLIS library resides in your chosen build directory, say `blis/build_blis` and the generated header files are in `blis/build_blis/include/<confname>`. To install the library and the header files associated with it, you can use:
```
cmake --build . --target install
```
This will install the libraries and header files and create the corresponding symbolic links of the shared libraries in the path specified in `CMAKE_INSTALL_PREFIX`.

Note that on Linux, if Makefiles are used, the above is equivalent to running
```
make install
```

## Uninstalling

Please note that CMake does not provide functionality to uninstall targets.

## Available targets

The BLIS CMake system aims to be combatible with the current `make` system. For that reason, it implements the same targets for the generation of libraries and the tests. The table of avalable targets can be found below.

| target   | Description                                        |
|:----------------|:---------------------------------------------------|
| `all`           | Execute `libs` target.                             |
| `libs`          | Compile BLIS as a static and/or shared library (depending on CMake options). |
| `test`          | Execute `checkblis` and `checkblas` targets.         |
| `check`         | Execute `checkblis-fast` and `checkblas` targets.  |
| `checkblis`     | Execute `testblis` and characterize the results to `stdout`. |
| `checkblis-fast`| Execute `testblis-fast` and characterize the results to `stdout`. |
| `checkblis-md`  | Execute `testblis-md` and characterize the results to `stdout`. |
| `checkblis-salt`| Execute `testblis-salt` and characterize the results to `stdout`. |
| `checkblas`     | Execute `testblas` and characterize the results to `stdout`. |
| `testblis`      | Run the BLIS testsuite with default parameters (runs for 2-8 minutes). |
| `testblis-fast` | Run the BLIS testsuite with "fast" parameters (runs for a few seconds). |
| `testblis-md`   | Run the BLIS testsuite for `gemm` with full mixing of datatypes (runs for 10-30 seconds). |
| `testblis-salt` | Run the BLIS testsuite while simulating application-level threading (runs for a few seconds). |
| `testsuite`     | Same as `testblis`.                                |
| `testblas`      | Run the BLAS test drivers with default parameters (runs for a few seconds). |
| `checkbliscpp`  | Run the BLIS C++ tests (runs for a few seconds). |

**_NOTE:_** 
Using those targets sets the environment appropriately, so copying the input files and/or the DLL in case of Windows builds is not required.

### Running the testsuites
* On Linux all targets can be build and run in `build_blis` directory.
* On Windows, when Visual Studio has been used as a generator, one can build and run the blis API related tests from `build_blis/testsuite` directory and blas API tests from `build_blis/blastest` directory. To build and run the BLIS C++  interface tests, execute the target `checkbliscpp` in `build_blis/vendor/testcpp` directory. The targets `check` and `test` can be used in `build_blis` directory.
* On Windows, if Visual Studio is used to build the library and tests, note that only the high level targets will appear. All targets are available to build from the command prompt.

## Adding configurations

The CMake system is designed to closely relate to the BLIS Make system. Assuming that a user has followed the steps in [Configuration How To](ConfigurationHowTo.md), adding the new configuration on the CMake system requires the following steps:
* Add a `make_defs.cmake` file which is equivalent to `make_defs.mk`. One can see `blis/config/zen/make_defs.cmake` and `blis/config/zen/make_defs.mk` for an example.
* Update `blis/CMakeLists.txt` to remove the error for the particular new configuration and to add the option in `set_property()` so that it appears in cmake-gui. 

## Some examples

In this section we provide some examples for users that are familiar with the build system based in Makefiles and want to try the new CMake system.

**_NOTE:_** 
The CMake system generates the shared libraries by default. To build the static libraries, you need to specify the corresponding CMake variable below
```
cmake .. -DBUILD_SHARED_LIBS=OFF -DBLIS_CONFIG_FAMILY=amdzen
```
The same generated header `blis.h` can be used when using the library.

For shared libraries on Windows, one can easily import the symbols by defining the macro `-DBLIS_EXPORT=__declspec(dllimport)` while building the application,
but this is not necessary if static data symbols and objects are not used.

### Example 1: multi-threaded LP64 libraries for amdzen configuration using clang compiler

* With configure script:
```
CC=clang ./configure --enable-threading=openmp --int-size=32 --blas-int-size=32 amdzen
```

* With CMake on Linux:
```
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang -DENABLE_THREADING=openmp -DINT_SIZE=32 -DBLAS_INT_SIZE=32 -DBLIS_CONFIG_FAMILY=amdzen
```

* With CMake on Windows:
```
cmake .. -G "Visual Studio 17 2022" -TClangCl -DENABLE_THREADING=openmp -DINT_SIZE=32 -DBLAS_INT_SIZE=32 -DBLIS_CONFIG_FAMILY=amdzen -DOpenMP_libomp_LIBRARY="path_to_openmp_library"
```

### Example 2: single-threaded ILP64 libraries for amdzen configuration with aocl_gemm addon enabled and default compiler

**_NOTE:_** 
Addon functionality is currently available only on Linux.

* With configure script:
```
./configure --enable-threading=no --int-size=64 --blas-int-size=64 --enable-addon=aocl_gemm amdzen
```

* With CMake on Linux:
```
cmake .. -DENABLE_THREADING=no -DINT_SIZE=64 -DBLAS_INT_SIZE=64 -DENABLE_ADDON=aocl_gemm -DBLIS_CONFIG_FAMILY=amdzen
```

## Conclusion

The BLIS CMake system is developed and maintained by AMD. You can contact us on the email-id toolchainsupport@amd.com. You can also raise any issue/suggestion on the git-hub repository at https://github.com/amd/blis/issues.