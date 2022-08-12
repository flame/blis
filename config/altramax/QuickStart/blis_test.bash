#!/bin/bash

if [ "$1" = "quiet" ]; then
    quiet_blistest="quiet"
else
    quiet_blistest=""
fi

# We don't want to quiet this part:
echo "#################################################################"
echo "Simple testing of BLIS - use testsuite for more extensive tests."
echo "#################################################################"

. ./blis_setenv.bash $quiet_blistest
# It's critical to unset parallelism parameters before
# running the test code!
. ./blis_unset_par.bash quiet
echo "Switching to directory $BLIS_HOME"
pushd $BLIS_HOME > /dev/null
make check -j
popd > /dev/null

