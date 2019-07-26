CWD=$(pwd)
echo $CWD
make clean
make blis CFLAGS+="-DFLOAT"
numactl -C 1 ./test_gemm1_blis.x
make clean
make blis CFLAGS+="-DDOUBLE"
numactl -C 1 ./test_gemm1_blis.x
make clean
make blis CFLAGS+="-DSCOMPLEX"
numactl -C 1 ./test_gemm1_blis.x
make clean
make blis CFLAGS+="-DDCOMPLEX"
numactl -C 1 ./test_gemm1_blis.x


cd ../test/
CWD=$(pwd)
echo $CWD
make clean
make blis CFLAGS+="-DFLOAT"
numactl -C 1 ./test_gemm_blis.x
make clean
make blis CFLAGS+="-DDOUBLE"
numactl -C 1 ./test_gemm_blis.x
make clean
make blis CFLAGS+="-DSCOMPLEX"
numactl -C 1 ./test_gemm_blis.x
make clean
make blis CFLAGS+="-DDCOMPLEX"
numactl -C 1 ./test_gemm_blis.x

