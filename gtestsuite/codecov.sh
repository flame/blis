#!/bin/bash

echo "Code Coverage for BLIS"
echo "obj_dir_path : $1"
echo "out_dir_name : $2"

#$1 : obj_dir_path
#$2 : out_dir_name

lcov --capture --directory $1 --output-file $2.info
lcov --remove $2.info -o $2_filtered.info '/usr/*' '/*/_deps/*'
genhtml $2_filtered.info --output-directory $2
