CWD=$(pwd)
echo $CWD
make clean
make blis CFLAGS+="-DFLOAT"
numactl -C 1 ./test_gemm_blis.x
numactl -C 1 ./test_trsm_blis.x
numactl -C 1 ./test_hemm_blis.x
numactl -C 1 ./test_symm_blis.x

make clean
make blis CFLAGS+="-DDOUBLE"
numactl -C 1 ./test_gemm_blis.x
numactl -C 1 ./test_trsm_blis.x
numactl -C 1 ./test_hemm_blis.x
numactl -C 1 ./test_symm_blis.x

make clean
make blis CFLAGS+="-DSCOMPLEX"
numactl -C 1 ./test_gemm_blis.x
numactl -C 1 ./test_trsm_blis.x
numactl -C 1 ./test_hemm_blis.x
numactl -C 1 ./test_symm_blis.x

make clean
make blis CFLAGS+="-DDCOMPLEX"
numactl -C 1 ./test_gemm_blis.x
numactl -C 1 ./test_trsm_blis.x
numactl -C 1 ./test_hemm_blis.x
numactl -C 1 ./test_symm_blis.x

