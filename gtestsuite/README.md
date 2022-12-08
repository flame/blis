Steps to build gtestsuite executable on linux
1. Set BLIS_PATH( blis installation path ) in gtestsuite Makefile.
2. To build executable
    GCC compiler  : type "make"
    AOCC compiler : type "make AOCL=1"
   executable will be generated in bin folder.
3. Set input parameters in input.general file and select api's from input.operations file.
4. Finally run the executable
      $./bin/libblis_gtest

Steps to build gtestsuite executable and run valdrind tool
1. In gtestsuite Makefile, define the macro/enable "__GTEST_VALGRIND_TEST__(-D__GTEST_VALGRIND_TEST__)"
   Note : Undefine the macro "__GTEST_VALGRIND_TEST__" when it is not built for valgrind test.
2. Generate the executable as mentioned above(Steps to build gtestsuite executable)
3. Set input parameters and select api's to test in input.general and input.operations respectively
4. Finally run the executable
      $ OMP_NUM_THREADS=1 valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes -s <executable>

Steps to build blis library and gtestsuite executable and run ASAN
1. Build blis library with the flag CFLAGS="-g -fsanitize=address"
   CC=clang ./configure -a aocl_gemm --enable-threading=openmp --enable-cblas CFLAGS="-g -fsanitize=address" auto
2. And in gtestsuite Makefile, define the macro/enable "__GTEST_VALGRIND_TEST__(-D__GTEST_VALGRIND_TEST__)"
   and even set/enable "CXXFLAGS += -fsanitize=address -static-libsan".
3. Generate the executable as mentioned above(Steps to build gtestsuite executable)
4. Set input parameters in input.general file and select api's from input.operations file.
5. Finally run the executable
      $./bin/libblis_gtest

Steps to build gtestsuite executable for mkl library(machine lib-daytonax-04)
1. Goto gtestsuite folder and export the following (paths depends on the machine, where intel package is placed)
     export LD_LIBRARY_PATH=/home/intel2019/update_05/intel64/:$LD_LIBRARY_PATH
     export MKLROOT=/home/intel2019/update_05/
     export MKL_DEBUG_CPU_TYPE=5
     export MKL_ENABLE_INSTRUCTIONS=AVX2
2. Type "make mkl=1" or "make AOCL=1 mkl=1(for clang compiler)", executable will be generated in bin folder
3. Set input parameters in input.general file and select api's from input.operations file.
4. Finally run the executable
      $./bin/gtest_mkl
