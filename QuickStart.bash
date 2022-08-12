#!/bin/bash
# Created by Oracle Labs
clear
echo "##########################################"
echo "Running Quickstart.bash..."
echo "##########################################"
# Get list of config directories...
if [[ "$1" != "" ]]; then
	{
	# if [ $# .gt 1 ] then;
  echo "checking for confguration ${1}..."
  check_dir=./config/$1
  if [ -d "$check_dir" ]
	then
    echo "Great!  Found platform $1! Checking for Quickstart info..."
    quick_dir=${check_dir}/QuickStart
    if [ -d "$quick_dir" ]
    then
      echo "QuickStart info exists for platform $1!  Fetching..."
 			echo "##########################################"
      ls -l $quick_dir/*
      
      cp $quick_dir/* .

			# Rename C Test File:
      mv ./TimeDGEMM.cfile ./TimeDGEMM.c
      
    QuickStartFile=./blis_quick_start_$1.txt

 	  echo "##########################################"
      echo "$1 scripts and info successfully fetched!"
      echo "You can now view the contents of $QuickStartFile here:"
 	  echo "##########################################"
      ls -l $QuickStartFile
      
      # sleep 4
      # This was too verbose to show at the time
      # cat ./$QuickStartFile
    else
      echo "Sorry - no QuickStart info exists for platform $1."
      #echo "Fetching generic QuickStart info from ref platform."
    fi
	else
    echo "Error: platform $1 not found."
    echo "You can try a related platform from the list instead:"
    sleep 3
    cat ./config_registry
	fi
  }
else
	{
echo
echo "Building and using blis takes a few steps:"
echo "##########################################"
echo "(1) Configure blis appropriately for your platform."
echo "(2) Make and test blis."
echo "(3) Set runtime environment variables."
echo "(4) Launch your blis enabled application"
echo "    with the correct NUMA settings"
echo "##########################################"
echo "...6 sec"
sleep 6
echo "Enter the closest platform from the architecture list below"
echo "to get detailed instructions and helpful scripts"
echo "for that architecture:"
echo " (These are all listed in file ./config_registry)"
echo "... 6 sec"
echo "##########################################"
sleep 6
echo " Browse and choose an architecture as input to QuickStart.bash..."
echo " The list is sorted by processor family, e.g. ARM NEON64."
echo "------------------------------------------"
echo " For example, for help using BLIS on an Altra, enter:"
echo " source ./QuickStart.bash altra"
echo " If you want help using BLIS on an AltraMax, enter:"
echo " source ./QuickStart.bash altramax"
echo "------------------------------------------"
echo " Helpful info and scripts will be copied into this"
echo " directory named by your platform, e.g."
echo " blis_quick_start_altra.txt"
echo "##########################################"
echo "Here's the supported list of BLIS platforms..."
sleep 8
cat ./config_registry
echo "##########################################"
	}
fi
