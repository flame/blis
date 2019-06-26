#!/bin/bash

cd ..
module load gcc/8.2
./configure power9
make -j
sleep 5
cd test
make blis -j
sleep 5
./runme.sh
