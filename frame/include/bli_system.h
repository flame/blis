/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BLIS_SYSTEM_H
#define BLIS_SYSTEM_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include <float.h>

// Determine if we are on a 64-bit or 32-bit architecture
#if defined(_M_X64) || defined(__x86_64) || defined(__aarch64__) || \
    defined(_ARCH_PPC64)
#define BLIS_ARCH_64
#else
#define BLIS_ARCH_32
#endif

// Determine the target operating system
#if defined(_WIN32) || defined(__CYGWIN__)
#define BLIS_OS_WINDOWS 1
#elif defined(__APPLE__) || defined(__MACH__)
#define BLIS_OS_OSX 1
#elif defined(__ANDROID__)
#define BLIS_OS_ANDROID 1
#elif defined(__linux__)
#define BLIS_OS_LINUX 1
#elif defined(__bgq__)
#define BLIS_OS_BGQ 1
#elif defined(__bg__)
#define BLIS_OS_BGP 1
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || \
      defined(__bsdi__) || defined(__DragonFly__)
#define BLIS_OS_BSD 1
#else
#error "Cannot determine operating system"
#endif

#if BLIS_OS_WINDOWS

  // Include Windows header file.
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #include <windows.h>

  // Undefine attribute specifiers in Windows.
  #define __attribute__(x)

  // Undefine restrict.
  #define restrict

#endif

// gettimeofday() needs this.
#if BLIS_OS_WINDOWS
  #include <time.h>
#elif BLIS_OS_OSX
  #include <mach/mach_time.h>
#else
  #include <sys/time.h>
  #include <time.h>
#endif

#endif
