#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas at Austin nor the names
#     of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

# Only include this block of code once.
ifndef MAKE_DEFS_MK_INCLUDED
MAKE_DEFS_MK_INCLUDED := yes



#
# --- Development tools definitions --------------------------------------------
#

# --- Determine the C compiler and related flags ---
ifeq ($(CC),)
CC             := gcc
CC_VENDOR      := gcc
endif

# Enable IEEE Standard 1003.1-2004 (POSIX.1d). 
# NOTE: This is needed to enable posix_memalign().
CPPROCFLAGS    := -D_POSIX_C_SOURCE=200112L
CMISCFLAGS     := -std=c99 -m64
CPICFLAGS      := -fPIC
CWARNFLAGS     := -Wall

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
COPTFLAGS      := -O0
else
COPTFLAGS      := -O3
endif

ifeq ($(DEBUG_TYPE),sde)
CPPROCFLAGS    += -DBLIS_NO_HBWMALLOC
endif

CKOPTFLAGS     := $(COPTFLAGS)

ifeq ($(CC_VENDOR),gcc)
CVECFLAGS      := -mavx512f -mavx512pf -mfpmath=sse -march=knl
else
ifeq ($(CC_VENDOR),icc)
CVECFLAGS      := -xMIC-AVX512
else
ifeq ($(CC_VENDOR),clang)
CVECFLAGS      := -mavx512f -mavx512pf -mfpmath=sse -march=knl
else
$(error gcc, icc, or clang is required for this configuration.)
endif
endif
endif

# The assembler on OS X won't recognize AVX512 without help
ifneq ($(CC_VENDOR),icc)
ifeq ($(OS_NAME),Darwin)
CVECFLAGS      += -Wa,-march=knl
endif
endif

# --- Determine the archiver and related flags ---
AR             := ar
ARFLAGS        := cr

# --- Determine the linker and related flags ---
LINKER         := $(CC)
SOFLAGS        := -shared

ifneq ($(DEBUG_TYPE),sde)
LDFLAGS        := -lmemkind
else
LDFLAGS        :=
endif

ifneq ($(CC_VENDOR),icc)
LDFLAGS        += -lm
endif 



# end of ifndef MAKE_DEFS_MK_INCLUDED conditional block
endif
