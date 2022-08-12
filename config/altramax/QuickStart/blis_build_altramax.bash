#!/bin/bash
echo "#######################################################"
echo "Building standard OpenMP BLIS..."
echo "#######################################################"
. ./blis_setenv.bash quiet
echo "#############################################################"
echo "Configuring BLIS for Altramax using OpenMP for parallelism..."
echo "#############################################################"
. ./blis_configure_altramax.bash quiet
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
