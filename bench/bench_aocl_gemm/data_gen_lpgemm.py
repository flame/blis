#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
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

# Initializing global mnk_array.This array will be used to store all mnk values
mnk_array = []

max_elem = 2600;
out_file_name = "accuracy_test_data_lpgemm.txt"
# Important mnk generator function.This will generate all possible combinations
# of m,n,k values using formula m(t+1)=ROUND(m(t)*Base,0)+offset
def mnk_generator():
    k_1 = 1
    incr_k = 500
    while (k_1 <= max_elem):
        n_1 = 1
        incr_n = 200
        while (n_1 <= max_elem):
            m_1 = 1
            incr_m = 100
            while (m_1 <= max_elem):
                mnk_array.append([m_1, n_1, k_1])
                if (m_1 == 1):
                    m_1 = m_1 + 9
                else:
                    m_1 = m_1 + incr_m
            if (n_1 == 1):
                n_1 = n_1 + 9
            else:
                n_1 = n_1 + incr_n
        if (k_1 == 1):
            k_1 = k_1 + 9
        else:
            k_1 = k_1 + incr_k

def data_gen():
    mnk_generator()

    fout = open(out_file_name, "w")

    for ele in mnk_array:
        fout.write("r n n n r " + str(ele[0]) + " " + str(ele[1]) + " " + str(ele[2]) + " " +\
                str(ele[2]) + " " + str(ele[1]) + " " + str(ele[1]) + " u8s8s32os32:none" + "\n")

    fout.truncate(fout.tell() - 1)
    fout.close()

##__main__
data_gen()
