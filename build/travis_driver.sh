#!/usr/bin/env bash

set -e
set -x

CC="$1"
THREADING="$2"
TARGET="$3"
RUN_TEST="$4"

export BLIS_IC_NT=2
export BLIS_JC_NT=1
export BLIS_IR_NT=1
export BLIS_JR_NT=1

HARDWARE=0
SDE_OPTIONS="false"
case "$TARGET" in
    knl)
        if $(grep avx512f -q /proc/cpuinfo) && $(grep avx512pf -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        SDE_OPTIONS="-knl"
        ;;
    skx)
        if $(grep avx512f -q /proc/cpuinfo) && $(grep avx512vl -q /proc/cpuinfo) && $(grep avx512dq -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        SDE_OPTIONS="-skx"
        ;;
    haswell)
        if $(grep avx2 -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        SDE_OPTIONS="-hsw"
        ;;
    sandybridge)
        if $(grep avx -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        SDE_OPTIONS="-snb"
        ;;
    dunnington)
        if $(grep ssse3 -q /proc/cpuinfo) && $(grep sse4_1 -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        SDE_OPTIONS="-pnr"
        ;;
    auto)
        HARDWARE=1
        ;;
    reference)
        HARDWARE=1
        ;;
    *)
        ;;
esac

if [ "x$TARGET" == "xknl" ] ; then
    # older binutils do not support AVX-512 (need at least 2.25)
    wget https://ftp.gnu.org/gnu/binutils/binutils-2.28.tar.bz2
    tar -xaf binutils-2.28.tar.bz2
    cd binutils-2.28
    ./configure --prefix=/tmp/binutils-2.28
    make
    make install
    export PATH=/tmp/binutils-2.28/bin:$PATH
    export LD_LIBRARY_PATH=/tmp/binutils-2.28/bin:$LD_LIBRARY_PATH
    which ld
    # now configure
    ./configure -d sde -t $THREADING CC=$CC $TARGET
else
    ./configure        -t $THREADING CC=$CC $TARGET
fi

make

if [ "x${RUN_TEST}" == "x1" ] ; then
    make testsuite-bin
    # We make no attempt to run SDE_OPTIONS on Mac.  It is supported but requires elevated permissions.
    if [ "x${HARDWARE}" != "x1" ] && [ "${TRAVIS_OS_NAME}" == "linux" ] ; then
        wget -q ${SDE_LOCATION}
        tar -xaf sde-external-7.58.0-2017-01-23-lin.tar.bz2
        export PATH=${PWD}/sde-external-7.58.0-2017-01-23-lin:${PATH}
        if [ `uname -m` = x86_64 ] ; then SDE_BIN=sde64 ; else SDE_BIN=sde ; fi
        ${PWD}/sde-external-7.58.0-2017-01-23-lin/${SDE_BIN} ${SDE_OPTIONS} -- make BLIS_ENABLE_TEST_OUTPUT=yes testsuite-run
    else
        make BLIS_ENABLE_TEST_OUTPUT=yes testsuite-run
    fi
fi
