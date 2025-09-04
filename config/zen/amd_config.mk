#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2021 - 2025, Advanced Micro Devices, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
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

# All the common flags for AMD architectures will be added here

# NOTE: The build system will append these variables with various
# general-purpose/configuration-agnostic flags in common.mk. You
# may specify additional flags here as needed.

CPPROCFLAGS    :=
CMISCFLAGS     :=
CPICFLAGS      :=
CWARNFLAGS     :=

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
  COPTFLAGS      := -O0
else
  COPTFLAGS      := -O3
endif

# Flags specific to optimized kernels.
# NOTE: The -fomit-frame-pointer option is needed for some kernels because
# they make explicit use of the rbp register.
CKOPTFLAGS     := $(COPTFLAGS) -fomit-frame-pointer
# Additional flag which is required for lpgemm kernels
CKLPOPTFLAGS     :=

ifeq ($(CC_VENDOR),gcc)
  CKVECFLAGS     := -mavx2 -mfpmath=sse -mfma
  ifeq ($(shell test $(CC_MAJOR) -ge 15; echo $$?),0)
    # gcc 15.1.0 fails to compile SUP kernels if -ftree-slp-vectorize
    # is enabled, which is default in -O2 and higher
    CKOPTFLAGS += -fno-tree-slp-vectorize
  endif
else ifeq ($(CC_VENDOR),clang)
  CKVECFLAGS     := -mavx2 -mfpmath=sse -mfma -mno-fma4 -mno-tbm -mno-xop -mno-lwp
  ifeq ($(strip $(shell $(CC) -v |&head -1 |grep -c 'AOCC.LLVM')),1)
    CKVECFLAGS += -mllvm -disable-licm-vrp
  endif
else
  $(error gcc or clang are required for this configuration.)
endif

# Flags specific to reference kernels.
CROPTFLAGS     := $(CKOPTFLAGS)
ifeq ($(CC_VENDOR),gcc)
  CRVECFLAGS     := $(CKVECFLAGS) -funsafe-math-optimizations -ffp-contract=fast
else ifeq ($(CC_VENDOR),clang)
  CRVECFLAGS     := $(CKVECFLAGS) -funsafe-math-optimizations -ffp-contract=fast
else
  CRVECFLAGS     := $(CKVECFLAGS)
endif

