::
::
::  BLIS    
::  An object-based framework for developing high-performance BLAS-like
::  libraries.
::
::  Copyright (C) 2014, The University of Texas at Austin
::
::  Redistribution and use in source and binary forms, with or without
::  modification, are permitted provided that the following conditions are
::  met:
::   - Redistributions of source code must retain the above copyright
::     notice, this list of conditions and the following disclaimer.
::   - Redistributions in binary form must reproduce the above copyright
::     notice, this list of conditions and the following disclaimer in the
::     documentation and/or other materials provided with the distribution.
::   - Neither the name of The University of Texas at Austin nor the names
::     of its contributors may be used to endorse or promote products
::     derived from this software without specific prior written permission.
::
::  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
::  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
::  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
::  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
::  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
::  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
::  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
::  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
::  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
::  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
::  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
::
::

@echo off

echo. 
echo  Makefile
echo. 
echo  Field G. Van Zee
echo.  
echo  nmake Makefile for building BLIS for Microsoft Windows. nmake targets
echo  may be invoked after running the configure.cmd script. Valid targets are:
echo. 
echo    all          - Invoke the lib and dll targets.
echo    lib          - Build BLIS as a static library.
echo    dll          - Build BLIS as a dynamically-linked library.
echo    help         - Output help and usage information.
echo    clean        - Invoke clean-log and clean-build targets.
echo    clean-log    - Remove any log files present.
echo    clean-config - Remove all products of configure.cmd. Namely, remove the
echo                   config, include, and src directories.
echo    clean-build  - Remove all products of the compilation portion of the build
echo                   process. Namely, remove the obj, lib, and dll directories.
echo    distclean    - Invoke clean-log, clean-config, and clean-build targets.
echo.
echo  The Makefile also recognizes configuration options corresponding to the
echo  following Makefile variables:
echo.
echo    VERBOSE               - When defined, nmake outputs the actual commands
echo                            executed instead of more concise one-line progress
echo                            indicators. (Undefined by default.)
echo.
echo  Typically, these options are specified by commenting or uncommenting the
echo  corresponding lines in the Makefile. However, if the Makefile currently does
echo  not define one of the options, and you wish to enable the corresponding
echo  feature without editing the Makefile, you may define the variable at the
echo  command line when nmake is invoked. For example, you may enable verboseness
echo  while invoking the lib target as follows:
echo.
echo    nmake lib VERBOSE=1
echo.
