#!/bin/bash

set -e
set -x

SDE_VERSION=sde-external-8.69.1-2021-07-18-lin
SDE_TARBALL=$SDE_VERSION.tar.bz2
SDE=$SDE_VERSION/sde64

#
# This doesn't seem to be necessary anymore
#
#curl --verbose --form accept_license=1 --form form_id=intel_licensed_dls_step_1 \
#     --output /dev/null --cookie-jar jar.txt \
#     --location https://software.intel.com/protected-download/267266/144917
#curl --verbose --cookie jar.txt --output $SDE_TARBALL \
#     https://software.intel.com/system/files/managed/2a/1a/$SDE_TARBALL

#curl --verbose --output $SDE_TARBALL \
#     https://software.intel.com/content/dam/develop/external/us/en/documents/downloads/$SDE_TARBALL

CI_UTILS=ci-utils
CI_UTILS_URL=https://github.com/flame/${CI_UTILS}.git
CI_UTILS_SDE_DIR=sde
SDE_DIRPATH=$CI_UTILS/$CI_UTILS_SDE_DIR

git clone $CI_UTILS_URL
mv $SDE_DIRPATH/$SDE_TARBALL .

tar xvf $SDE_TARBALL

make -j2 testsuite-bin blastest-bin

TMP=`ldd ./test_libblis.x | grep ld | sed 's/^.*=> //'`
LD_SO=${TMP%% *}
TMP=`ldd ./test_libblis.x | grep libc | sed 's/^.*=> //'`
LIBC_SO=${TMP%% *}
TMP=`ldd ./test_libblis.x | grep libm | sed 's/^.*=> //'`
LIBM_SO=${TMP%% *}
for LIB in $LD_SO $LIBC_SO $LIBM_SO; do
    $DIST_PATH/ci/patch-ld-so.py $LIB .tmp
    chmod a+x .tmp
    sudo mv .tmp $LIB
done

for ARCH in sandybridge haswell skx knl piledriver steamroller excavator zen; do
#for ARCH in penryn sandybridge haswell skx knl piledriver steamroller excavator zen; do
    if [ "$ARCH" = "knl" ]; then
        TESTSUITE_WRAPPER="$SDE -knl --"
    else
        TESTSUITE_WRAPPER="$SDE -cpuid_in $DIST_PATH/ci/cpuid/$ARCH.def --"
    fi

    make TESTSUITE_WRAPPER="$TESTSUITE_WRAPPER" check

    TMP=`grep "active sub-configuration" output.testsuite`
    CONFIG=${TMP##* }
    if [ "$CONFIG" != "$ARCH" ]; then
        echo "Wrong configuration chosen:"
        echo "    Expected: $ARCH"
        echo "    Got: $CONFIG"
        exit 1
    fi
done

