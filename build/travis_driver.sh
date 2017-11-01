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
        if [ "$TRAVIS_OS_NAME" = "linux" ] ; then
            if $(grep avx512f -q /proc/cpuinfo) && $(grep avx512pf -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        fi
        SDE_OPTIONS="-knl"
        ;;
    skx)
        if [ "$TRAVIS_OS_NAME" = "linux" ] ; then
            if $(grep avx512f -q /proc/cpuinfo) && $(grep avx512vl -q /proc/cpuinfo) && $(grep avx512dq -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        fi
        SDE_OPTIONS="-skx"
        ;;
    haswell)
        if [ "$TRAVIS_OS_NAME" = "linux" ] ; then
            if $(grep avx2 -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        elif [ "$TRAVIS_OS_NAME" = "osx" ] ; then
            #if $(sysctl -a | grep machdep.cpu.leaf7_features | grep -q AVX2) ; then HARDWARE=1 ; fi
            if [ $(sysctl -n hw.optional.avx2_0) -eq 1 ] ; then HARDWARE=1 ; fi
        fi
        SDE_OPTIONS="-hsw"
        ;;
    sandybridge)
        if [ "$TRAVIS_OS_NAME" = "linux" ] ; then
            if $(grep avx -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        elif [ "$TRAVIS_OS_NAME" = "osx" ] ; then
            #if $(sysctl -a | grep machdep.cpu.features | grep -q AVX1) ; then HARDWARE=1 ; fi
            if [ $(sysctl -n hw.optional.avx1_0) -eq 1 ] ; then HARDWARE=1 ; fi
        fi
        SDE_OPTIONS="-snb"
        ;;
    dunnington)
        if [ "$TRAVIS_OS_NAME" = "linux" ] ; then
            if $(grep ssse3 -q /proc/cpuinfo) && $(grep sse4_1 -q /proc/cpuinfo) ; then HARDWARE=1 ; fi
        elif [ "$TRAVIS_OS_NAME" = "osx" ] ; then
            #if $(sysctl -a | grep machdep.cpu.features | grep -q "SSE4.1") ; then HARDWARE=1 ; fi
            if [ $(sysctl -n hw.optional.sse4_1) -eq 1 ] ; then HARDWARE=1 ; fi
        fi
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
    pushd /tmp
    # older binutils do not support AVX-512 (need at least 2.25)
    wget https://ftp.gnu.org/gnu/binutils/binutils-2.28.tar.bz2
    tar -xaf binutils-2.28.tar.bz2
    cd binutils-2.28
    export BINUTILS_PATH=/tmp/mybinutils
    ./configure --prefix=$BINUTILS_PATH
    make
    make install
    export PATH=$BINUTILS_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$BINUTILS_PATH/lib:$LD_LIBRARY_PATH
    popd
    which ld
    # now configure BLIS
    ./configure -d sde -t $THREADING CC=$CC $TARGET
else
    ./configure        -t $THREADING CC=$CC $TARGET
fi

make

if [ "x${RUN_TEST}" == "x1" ] ; then
    make testsuite-bin
    # We make no attempt to run SDE_OPTIONS on Mac.  It is supported but requires elevated permissions.
    if [ "x${HARDWARE}" != "x1" ] && [ "${TRAVIS_OS_NAME}" == "linux" ] ; then
        set +x
        wget -q ${SDE_LOCATION}
        set -x
        tar -xaf sde-external-7.58.0-2017-01-23-lin.tar.bz2
        export PATH=${PWD}/sde-external-7.58.0-2017-01-23-lin:${PATH}
        if [ `uname -m` = x86_64 ] ; then SDE_BIN=sde64 ; else SDE_BIN=sde ; fi
        ${PWD}/sde-external-7.58.0-2017-01-23-lin/${SDE_BIN} ${SDE_OPTIONS} -- make BLIS_ENABLE_TEST_OUTPUT=yes testsuite-run
    else
        make BLIS_ENABLE_TEST_OUTPUT=yes testsuite-run
    fi
fi
