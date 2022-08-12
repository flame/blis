#!/bin/bash

if [ "$1" = "quiet" ]; then
    quiet_confopenmp="quiet"
else
    quiet_confopenmp=""
fi

if [ "$quiet_confopenmp" = "" ]; then
    echo "##########################################################"
    echo "Configuring BLIS for Altra using OpenMP for parallelism..."
    echo "##########################################################"
fi

. ./blis_setenv.bash $quiet_confopenmp
pushd $BLIS_HOME > /dev/null
make distclean
./configure -t openmp --disable-pba-pools altra
popd > /dev/null

