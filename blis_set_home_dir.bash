#!/bin/bash
##########################################
# This file sets the value of BLIS_HOME
# In your environment to the current local
# blis directory this script resides in.
##########################################
# The use case we envision is a person may
# Have multiple blis directories on his
# machine, and needs to be sure WHICH blis
# library he is actually using.  You can
# call this from any location, but scripts
# that use it must remain in their blis
# directories.
##########################################
# If needed, you may hard code BLIS_HOME in
# blis_setenv.bash, but be careful!
##########################################
# Try to be clever and find the current
# directory path regardless of where it's
# called from:

if [ "$1" = "quiet" ]; then
    quiet_sethome="quiet"
else
    quiet_sethome=""
fi

# Use realpath on Unix, less robust ways on Mac, etc.
#
if type realpath &> /dev/null; then

   # Use realpath if it exists to get the actual path with no soft links
   if [[ -n "$BASH_VERSION" ]] ; then
      export BLIS_HOME="$( realpath $( dirname ${BASH_SOURCE[0]} ) )"
   else
      export BLIS_HOME="$( realpath $( dirname $0 ) )"
   fi

else

   # Otherwise, this should work on any system, but the path might include soft links
   if [[ -n "$BASH_VERSION" ]] ; then
      export BLIS_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
   else
      export BLIS_HOME="$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"
   fi

fi

if [ "$quiet_sethome" = "" ]; then
    echo "BLIS HOME set to $BLIS_HOME"
fi
