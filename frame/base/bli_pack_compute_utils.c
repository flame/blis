/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

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
#include "blis.h"

// Utility function to compute the offset for K dimension traversal
// such that it is a multiple of NR.
dim_t get_Bpanel_width_for_kdim_traversal
     (
       dim_t jc,
       dim_t n,
       dim_t NC,
       dim_t NR
     )
{
    dim_t n_mod_NR = n % NR;
    dim_t n_sub_updated = NC;

    if ( ( n % NC ) != 0 )
    {
        // Only applicable to final NC part of jc loop where jc + remaining
        // elements is less than NC; or when n < NC in which case panel width
        // is atmost n.
        dim_t n_last_loop = ( n / NC ) * NC;
        if ( jc >= n_last_loop )
        {
            n_sub_updated = n - n_last_loop;
            if ( n_mod_NR != 0 )
            {
                n_sub_updated += ( NR - n_mod_NR );
            }
        }
    }

    return n_sub_updated;
}

void get_B_panel_reordered_start_offset_width
     (
       dim_t  jc,
       dim_t  n,
       dim_t  NC,
       dim_t  NR,
       dim_t* panel_start,
       dim_t* panel_offset,
       dim_t* panel_width,
       dim_t* panel_width_kdim_trav
     )
{
    // Since n dimension is split across threads in units of NR blocks,
    // it could happen that B matrix chunk for a thread may be part of
    // two separate NCxKC panels. In this case nc0 is updated such that
    // the jr loop only accesses the remaining portion of current NCxKC
    // panel, with the next jc iteration taking care of the other panel.
    // This ensures that jr loop does not cross panel boundaries.
    ( *panel_start ) = ( jc / NC ) * NC;
    ( *panel_offset ) = jc - ( *panel_start );

    // Check if jc + current_panel_width (nc0) crosses panel boundaries.
    if ( ( jc + ( *panel_width ) ) > ( ( *panel_start ) + NC ) )
    {
        ( *panel_width ) = NC - ( *panel_offset );
    }

    ( *panel_width_kdim_trav ) = get_Bpanel_width_for_kdim_traversal
                                 (
                                   jc, n, NC, NR
                                 );
}

void adjust_B_panel_reordered_jc( dim_t* jc, dim_t panel_start )
{
    // Since n dimension is split across threads in units of NR blocks,
    // it could happen that B matrix chunk for a thread may be part of
    // two separate NCxKC panels. In this case jc is reset to immediate
    // previous panel offset so that in the next iteration, the
    // following panel belonging to the B chunk is accessed. This
    // ensures that jr loop does not cross panel boundaries.
    ( *jc ) = panel_start;
}


