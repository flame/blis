source activate
export CC=clang
export CXX=clang++
export RANLIB=echo
./configure --disable-shared auto
make -j4 || make -j4
make check
