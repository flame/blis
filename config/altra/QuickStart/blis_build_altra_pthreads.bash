#!/bin/bash
echo "#######################################################"
echo "Building pThreads version of BLIS..."
echo "#######################################################"
. ./blis_setenv.bash quiet
echo "##########################################################"
echo "Configuring BLIS for Altra using pThreads for parallelism..."
echo "##########################################################"
. ./blis_configure_altra_pthreads.bash quiet
echo "Switching to directory $BLIS_HOME"
pushd $BLIS_HOME > /dev/null
make -j
popd > /dev/null
if [ "$1" != "notest" ]; then
    . ./blis_test.bash quiet
fi
. ./blis_setenv.bash
echo "##########################################################"
echo "...done"
echo "##########################################################"
