#!/bin/bash

set -e
set -x

SDE_VERSION=sde-external-8.16.0-2018-01-30-lin
SDE_TARBALL=$SDE_VERSION.tar.bz2
SDE=$SDE_VERSION/sde64

set +x
curl -s -X POST https://content.dropboxapi.com/2/files/download -H "Authorization: Bearer $DROPBOX_TOKEN" -H "Dropbox-API-Arg: {\"path\": \"/$SDE_TARBALL\"}" > $SDE_TARBALL
set -x
tar xvf $SDE_TARBALL

make -j2 testsuite-bin
cp $DIST_PATH/testsuite/input.general.fast input.general
cp $DIST_PATH/testsuite/input.operations.fast input.operations

$DIST_PATH/travis/patch-ld-so.py /lib64/ld-linux-x86-64.so.2 ld.so.nohwcap
chmod a+x ld.so.nohwcap

for ARCH in penryn sandybridge haswell skx knl piledriver steamroller excavator; do
    if [ "$ARCH" = "knl" ]; then
        $SDE -knl -- ./ld.so.nohwcap ./test_libblis.x > output.testsuite
    else
        $SDE -cpuid_in $DIST_PATH/travis/cpuid/$ARCH.def -- ./ld.so.nohwcap ./test_libblis.x > output.testsuite
    fi
    $DIST_PATH/build/check-blistest.sh ./output.testsuite
    TMP=`grep "active sub-configuration" output.testsuite`
    CONFIG=${TMP##* }
    if [ "$CONFIG" != "$ARCH" ]; then
        echo "Wrong configuration chosen:"
        echo "    Expected: $ARCH"
        echo "    Got: $CONFIG"
        exit 1
    fi
done

