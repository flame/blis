#!/bin/bash

set -e
set -x

SDE_VERSION=sde-external-8.16.0-2018-01-30-lin
SDE_TARBALL=$SDE_VERSION.tar.bz2
SDE=$SDE_VERSION/sde64

curl --verbose --form accept_license=1 --form form_id=intel_licensed_dls_step_1 \
     --output /dev/null --cookie-jar jar.txt \
     --location https://software.intel.com/protected-download/267266/144917
curl --verbose --cookie jar.txt --output $SDE_TARBALL \
     https://software.intel.com/system/files/managed/2a/1a/$SDE_TARBALL

tar xvf $SDE_TARBALL

make -j2 testsuite-bin
cp $DIST_PATH/testsuite/input.general.fast input.general
cp $DIST_PATH/testsuite/input.operations.fast input.operations

TMP=`ldd ./test_libblis.x | grep ld | sed 's/^.*=> //'`
LD_SO=${TMP%% *}
TMP=`ldd ./test_libblis.x | grep libc | sed 's/^.*=> //'`
LIBC_SO=${TMP%% *}
TMP=`ldd ./test_libblis.x | grep libm | sed 's/^.*=> //'`
LIBM_SO=${TMP%% *}
for LIB in $LD_SO $LIBC_SO $LIBM_SO; do
    $DIST_PATH/travis/patch-ld-so.py $LIB .tmp
    chmod a+x .tmp
    sudo mv .tmp $LIB
done

for ARCH in penryn sandybridge haswell skx knl piledriver steamroller excavator zen; do
    if [ "$ARCH" = "knl" ]; then
        $SDE -knl -- ./test_libblis.x > output.testsuite
    else
        $SDE -cpuid_in $DIST_PATH/travis/cpuid/$ARCH.def -- ./test_libblis.x > output.testsuite
    fi
    $DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
    TMP=`grep "active sub-configuration" output.testsuite`
    CONFIG=${TMP##* }
    if [ "$CONFIG" != "$ARCH" ]; then
        echo "Wrong configuration chosen:"
        echo "    Expected: $ARCH"
        echo "    Got: $CONFIG"
        exit 1
    fi
done

