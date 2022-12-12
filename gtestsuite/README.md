# Dependencies
* GoogleTest is used as the tool for testing. The library is fetched and build at configuration time. No installation necessary. 
* An installation location for BLIS needs to be passed in as an argument during cmake invocation.

# Build Tests
$ mkdir build\
$ cd build\
$ cmake .. -DBLIS_PATH = /home/username/blis_installation\
$ make -j

This will build all executables into an executable named `gtest_libblis`.

# Run Tests
./gtest_libblis

# Configuration Options
There are multiple configuration options to chose from when invoking CMake. 

## Compiler Options
* `-DCMAKE_CXX_COMPILER=path_to_preferred_compiler` can be used to specify the compiler.
* For example, to compile with Clang, use `-DCMAKE_CXX_COMPILER=clang++`.

## Threading Options (Linux Only)
* For single threaded BLIS, use `-DENABLE_THREADING=no`.
* For multithreaded BLIS that uses pthreads, use `-DENABLE_THREADING=pthreads`.
* For multithreaded BLIS that uses OpenMP, use `-DENABLE_THREADING=openmp`. [**Default**]
      * In addition, to use Intel OpenMP runtime, use `-DOpenMP_LIBRARY=Intel`.
      * For GNU OpenMP runtime, use `-DOpenMP_LIBRARY=GNU`. [**Default**]

## BLIS Linking Type (Linux Only) 
* To link static BLIS, use `-DBLIS_LINKING_TYPE=static`. [**Default**]
* To link shared BLIS, use `-DBLIS_LINKING_TYPE=shared`. [**Disabled**]

## Valgrind
* To build for a valgrind test, configure using `-DENABLE_VALGRIND=ON`. [**OFF by default**]
* Run the executable using
$ OMP_NUM_THREADS=1 valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes -s ./gtest_libblis

## Address Sanitizer
* To build using address sanitizer, configure using `-DENABLE_ASAN=ON`. [**OFF by default**]
* An installation to BLIS which was build with ASAN flags needs to be provided.

__Currently tested only with clang compiler.__