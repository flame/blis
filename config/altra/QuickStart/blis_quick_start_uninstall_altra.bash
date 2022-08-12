#!/bin/bash
# This utility will remove all the configuration
# Specific QuickStart files from the blis directory.
# This is very useful when switching configurations!
#
if [[ -n "$BLIS_HOME" ]]; then
  echo "REMOVING ALL ALTRA QUICKSTART FILES FROM $BLIS_HOME"
    
  rm $BLIS_HOME/blis_build_altra_pthreads.bash
	rm $BLIS_HOME/blis_build_altra.bash
	rm $BLIS_HOME/blis_build_both_libraries.bash
	
	rm $BLIS_HOME/blis_configure_altra_pthreads.bash
	rm $BLIS_HOME/blis_configure_altra.bash
	
	rm $BLIS_HOME/blis_quick_start_altra.txt
	rm $BLIS_HOME/blis_setenv.bash
	
  rm $BLIS_HOME/blis_unset_par.bash
  rm $BLIS_HOME/blis_test.bash

	rm $BLIS_HOME/TimeDGEMM.c
	rm $BLIS_HOME/time_gemm.x
	
	rm $BLIS_HOME/blis_quick_start_uninstall_altra.bash

else
  echo "ONLY USE THIS SCRIPT FROM THE BLIS HOME DIRECTORY!"
  echo "BLIS_HOME is not set!"
fi
