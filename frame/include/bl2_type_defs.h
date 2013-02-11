/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#ifndef BLIS_TYPE_DEFS_H
#define BLIS_TYPE_DEFS_H

// -- Basic types --------------------------------------------------------------

// Boolean type

typedef   signed long int bool_t;

// Integer types

typedef unsigned long int dim_t;  // dimension type
typedef unsigned long int inc_t;  // increment/stride type
typedef   signed long int doff_t; // diagonal offset type
typedef unsigned long int siz_t;  // byte size type

// Complex types

typedef struct scomplex_s
{
	float  real;
	float  imag;
} scomplex;

typedef struct dcomplex_s
{
	double real;
	double imag;
} dcomplex;

// Memory object type

typedef struct mem_s
{
	void*  buf;
	dim_t  m;
	dim_t  n;
} mem_t;

// Blocksize object type

typedef struct blksz_s
{
	dim_t v[BLIS_NUM_FP_TYPES];
} blksz_t;

// Atom type
// Note: atom types are used to hold "bufferless" scalar object values. Note
// that it needs to be as large as the largest possible scalar value we might
// want to hold. Thus, for now, it is a dcomplex.

typedef dcomplex atom_t;


// -- BLIS object type definitions ---------------------------------------------

// Object information bit field

typedef unsigned long int info_t;

#define BLIS_DOMAIN_BIT           0x01
#define BLIS_PRECISION_BIT        0x02
#define BLIS_SPECIAL_BIT          0x04
#define BLIS_DATATYPE_BITS        0x07
#define BLIS_TRANS_BIT            0x08
#define BLIS_CONJ_BIT             0x10
#define BLIS_CONJTRANS_BITS       0x18
#define BLIS_UPPER_BIT            0x20
#define BLIS_DIAG_BIT             0x40
#define BLIS_LOWER_BIT            0x80
#define BLIS_UPLO_BITS            0xE0
#define BLIS_UNIT_DIAG_BIT        0x100
#define BLIS_TARGET_DT_BITS       0xE00
#define BLIS_EXECUTION_DT_BITS    0x7000
#define BLIS_PACK_BITS            0x38000
#define BLIS_STRUC_BITS           0xC0000

#define BLIS_BITVAL_REAL              0x00
#define BLIS_BITVAL_COMPLEX           0x01
#define BLIS_BITVAL_SINGLE_PREC       0x00
#define BLIS_BITVAL_DOUBLE_PREC       0x02
#define   BLIS_BITVAL_FLOAT_TYPE      0x00
#define   BLIS_BITVAL_SCOMPLEX_TYPE   0x01
#define   BLIS_BITVAL_DOUBLE_TYPE     0x02
#define   BLIS_BITVAL_DCOMPLEX_TYPE   0x03
#define   BLIS_BITVAL_INT_TYPE        0x04
#define   BLIS_BITVAL_CONST_TYPE      0x05
#define BLIS_BITVAL_NO_TRANS          0x0
#define BLIS_BITVAL_TRANS             0x08
#define BLIS_BITVAL_NO_CONJ           0x0
#define BLIS_BITVAL_CONJ              0x10
#define BLIS_BITVAL_ZEROS             0x0 
#define BLIS_BITVAL_UPPER             0x60 
#define BLIS_BITVAL_LOWER             0xC0
#define BLIS_BITVAL_DENSE             0xE0
#define BLIS_BITVAL_NONUNIT_DIAG      0x0
#define BLIS_BITVAL_UNIT_DIAG         0x100
#define BLIS_BITVAL_NOT_PACKED        0x0
#define BLIS_BITVAL_PACKED_UNSPEC     0x8000
#define BLIS_BITVAL_PACKED_VECTOR     0x10000
#define BLIS_BITVAL_PACKED_ROWS       0x18000
#define BLIS_BITVAL_PACKED_COLUMNS    0x20000
#define BLIS_BITVAL_PACKED_ROW_PANELS 0x28000
#define BLIS_BITVAL_PACKED_COL_PANELS 0x30000
#define BLIS_BITVAL_GENERAL           0x0
#define BLIS_BITVAL_HERMITIAN         0x40000
#define BLIS_BITVAL_SYMMETRIC         0x80000
#define BLIS_BITVAL_TRIANGULAR        0xC0000

#define BLIS_TARGET_DT_SHIFT          9
#define BLIS_EXECUTION_DT_SHIFT       12

/*
  info field description

  13 12 11 10 F E D C B A 9 8 7 6 5 4 3 2 1 0

  bit(s)   purpose
  ------   -------
   2 ~ 0   Stored numerical datatype
           - 0: domain    (0 == real, 1 == complex)
           - 1: precision (0 == single, 1 == double)
           - 2: special   (100 = int; 101 = const)
       3   Transposition required [during pack]?
       4   Conjugation required [during pack]?
   7 ~ 5   Part of matrix stored:
           - 5: strictly upper triangular
           - 6: diagonal
           - 7: strictly lower triangular
       8   Implicit unit diagonal?
   B ~ 9   Target numerical datatype
           - 9: domain    (0 == real, 1 == complex)
           - A: precision (0 == single, 1 == double)
           - B: unused
   E ~ C   Execution numerical datatype
           - C: domain    (0 == real, 1 == complex)
           - D: precision (0 == single, 1 == double)
           - E: unused
  11 ~ F   Packed type/status
           - 0 == not packed
           - 1 == packed (unspecified; row or column)
           - 2 == packed vector
           - 3 == packed by rows
           - 4 == packed by columns
           - 5 == packed by row panels
           - 6 == packed by column panels
           - 7 == unused
  13 ~ 12  Structure type
           - (00 == general, 01 == Hermitian)
           - (10 == symmetric, 11 == triangular)
*/

typedef struct obj_s
{
	// Basic fields
	struct obj_s* root;

	dim_t         offm;
	dim_t         offn;
	dim_t         m;
	dim_t         n;
	doff_t        diag_off;

	info_t        info;
	siz_t         elem_size;

	void*         buffer;
	inc_t         rs;
	inc_t         cs;

	// Bufferless scalar storage
	atom_t        scalar;

	// Pack-related fields
	mem_t         pack_mem; // cached memory region for packing
	mem_t         cast_mem; // cached memory region for casting
	inc_t         ps;       // panel stride (distance to next panel)

} obj_t;


// Define these macros here since they must be updated if contents of
// obj_t changes.
#define bl2_obj_init_as_copy_of( a, b ) \
{ \
	(b).root      = (a).root; \
\
	(b).offm      = (a).offm; \
	(b).offn      = (a).offn; \
	(b).m         = (a).m; \
	(b).n         = (a).n; \
	(b).diag_off  = (a).diag_off; \
\
	(b).info      = (a).info; \
	(b).elem_size = (a).elem_size; \
\
	(b).buffer    = (a).buffer; \
	(b).rs        = (a).rs; \
	(b).cs        = (a).cs; \
\
	/* We must NOT copy pack_mem field since this macro forms the basis of
	   bl2_obj_alias_to(), which is used in packm. There, we want to copy
	   over the basic fields of the obj_t but PRESERVE the pack_mem field
	   of the destination object since it holds the cached mem_t buffer
	   (and dimensions). */ \
}

#define bl2_obj_init_subpart_from( a, b ) \
{ \
	(b).root      = (a).root; \
\
	(b).offm      = (a).offm; \
	(b).offn      = (a).offn; \
\
\
	(b).diag_off  = (a).diag_off; \
\
	(b).info      = (a).info; \
	(b).elem_size = (a).elem_size; \
\
	(b).buffer    = (a).buffer; \
	(b).rs        = (a).rs; \
	(b).cs        = (a).cs; \
\
	/* We want to copy the pack_mem field here because this macro is used
	   when creating subpartitions, including those of packed objects. In
	   those situations, we want the subpartition to inherit the pack_mem
	   field of its parent. */ \
	(b).pack_mem  = (a).pack_mem; \
	(b).cast_mem  = (a).cast_mem; \
	(b).ps        = (a).ps; \
}


// Operational parameter types

typedef enum
{
	BLIS_NO_TRANSPOSE      = 0x0,
	BLIS_TRANSPOSE         = BLIS_TRANS_BIT,
	BLIS_CONJ_NO_TRANSPOSE = BLIS_CONJ_BIT,
	BLIS_CONJ_TRANSPOSE    = BLIS_CONJ_BIT | BLIS_TRANS_BIT
} trans_t;

typedef enum
{
	BLIS_NO_CONJUGATE      = 0x0,
	BLIS_CONJUGATE         = BLIS_CONJ_BIT
} conj_t;

typedef enum
{
	BLIS_ZEROS             = BLIS_BITVAL_ZEROS,
	BLIS_LOWER             = BLIS_LOWER_BIT | BLIS_DIAG_BIT,
	BLIS_UPPER             = BLIS_UPPER_BIT | BLIS_DIAG_BIT,
	BLIS_DENSE             = BLIS_UPPER_BIT | BLIS_DIAG_BIT | BLIS_LOWER_BIT
} uplo_t;

typedef enum
{
	BLIS_LEFT,
	BLIS_RIGHT
} side_t;

typedef enum
{
	BLIS_NONUNIT_DIAG      = 0x0,
	BLIS_UNIT_DIAG         = BLIS_BITVAL_UNIT_DIAG
} diag_t;

typedef enum
{
	BLIS_GENERAL           = BLIS_BITVAL_GENERAL,
	BLIS_HERMITIAN         = BLIS_BITVAL_HERMITIAN,
	BLIS_SYMMETRIC         = BLIS_BITVAL_SYMMETRIC,
	BLIS_TRIANGULAR        = BLIS_BITVAL_TRIANGULAR
} struc_t;


// Data type

typedef enum
{
	BLIS_FLOAT             = BLIS_BITVAL_FLOAT_TYPE,
	BLIS_DOUBLE            = BLIS_BITVAL_DOUBLE_TYPE,
	BLIS_SCOMPLEX          = BLIS_BITVAL_SCOMPLEX_TYPE,
	BLIS_DCOMPLEX          = BLIS_BITVAL_DCOMPLEX_TYPE,
	BLIS_INT               = BLIS_BITVAL_INT_TYPE,
	BLIS_CONSTANT          = BLIS_BITVAL_CONST_TYPE,
} num_t;

typedef enum
{
	BLIS_REAL              = BLIS_BITVAL_REAL,
	BLIS_COMPLEX           = BLIS_BITVAL_COMPLEX
} dom_t;

typedef enum
{
	BLIS_SINGLE_PREC       = BLIS_BITVAL_SINGLE_PREC,
	BLIS_DOUBLE_PREC       = BLIS_BITVAL_DOUBLE_PREC
} prec_t;


// Pack schema type

typedef enum
{
	BLIS_NOT_PACKED        = BLIS_BITVAL_NOT_PACKED,
	BLIS_PACKED_UNSPEC     = BLIS_BITVAL_PACKED_UNSPEC,
	BLIS_PACKED_VECTOR     = BLIS_BITVAL_PACKED_VECTOR,
	BLIS_PACKED_ROWS       = BLIS_BITVAL_PACKED_ROWS,
	BLIS_PACKED_COLUMNS    = BLIS_BITVAL_PACKED_COLUMNS,
	BLIS_PACKED_ROW_PANELS = BLIS_BITVAL_PACKED_ROW_PANELS,
	BLIS_PACKED_COL_PANELS = BLIS_BITVAL_PACKED_COL_PANELS
} pack_t;


// Subpartition type

typedef enum
{
	BLIS_SUBPART0,
	BLIS_SUBPART1,
	BLIS_SUBPART2,
	BLIS_SUBPART00,
	BLIS_SUBPART10,
	BLIS_SUBPART20,
	BLIS_SUBPART01,
	BLIS_SUBPART11,
	BLIS_SUBPART21,
	BLIS_SUBPART02,
	BLIS_SUBPART12,
	BLIS_SUBPART22,
	BLIS_SUBPART10B,
	BLIS_SUBPART12B
} subpart_t;


// Machine parameter types

typedef enum
{
    BLIS_MACH_EPS = 0,
    BLIS_MACH_SFMIN,
    BLIS_MACH_BASE,
    BLIS_MACH_PREC,
    BLIS_MACH_NDIGMANT,
    BLIS_MACH_RND,
    BLIS_MACH_EMIN,
    BLIS_MACH_RMIN,
    BLIS_MACH_EMAX,
    BLIS_MACH_RMAX,
    BLIS_MACH_EPS2
} machval_t;

#define BLIS_NUM_MACH_PARAMS   11
#define BLIS_MACH_PARAM_FIRST  BLIS_MACH_EPS
#define BLIS_MACH_PARAM_LAST   BLIS_MACH_EPS2


// Error types

typedef enum
{
	BLIS_NO_ERROR_CHECKING = 0,
	BLIS_FULL_ERROR_CHECKING,
} errlev_t;

typedef enum
{
	// Generic error codes
	BLIS_SUCCESS                               = (  -1),
	BLIS_FAILURE                               = (  -2),

	BLIS_ERROR_CODE_MIN                        = (  -9),

	// General errors
	BLIS_INVALID_ERROR_CHECKING_LEVEL          = ( -10),
	BLIS_UNDEFINED_ERROR_CODE                  = ( -11),
	BLIS_NULL_POINTER                          = ( -12),
	BLIS_NOT_YET_IMPLEMENTED                   = ( -13),

	// Parameter-specific errors
	BLIS_INVALID_SIDE                          = ( -20),
	BLIS_INVALID_UPLO                          = ( -21),
	BLIS_INVALID_TRANS                         = ( -22),
	BLIS_INVALID_CONJ                          = ( -23),
	BLIS_INVALID_DIAG                          = ( -24),
	BLIS_INVALID_MACHVAL                       = ( -25),

	// Datatype-specific errors
	BLIS_INVALID_DATATYPE                      = ( -30),
	BLIS_EXPECTED_FLOATING_POINT_DATATYPE      = ( -31),
	BLIS_EXPECTED_NONINTEGER_DATATYPE          = ( -32),
	BLIS_EXPECTED_NONCONSTANT_DATATYPE         = ( -33),
	BLIS_EXPECTED_REAL_DATATYPE                = ( -34),
	BLIS_INCONSISTENT_DATATYPES                = ( -35),
	BLIS_EXPECTED_REAL_PROJ_OF                 = ( -36),
	BLIS_EXPECTED_REAL_VALUED_OBJECT           = ( -37),

	// Dimension-specific errors
	BLIS_NONCONFORMAL_DIMENSIONS               = ( -40),
	BLIS_EXPECTED_SCALAR_OBJECT                = ( -41),
	BLIS_EXPECTED_VECTOR_OBJECT                = ( -42),
	BLIS_UNEQUAL_VECTOR_LENGTHS                = ( -43),
	BLIS_EXPECTED_SQUARE_OBJECT                = ( -44),
	BLIS_UNEXPECTED_OBJECT_LENGTH              = ( -45),
	BLIS_UNEXPECTED_OBJECT_WIDTH               = ( -46),
	BLIS_UNEXPECTED_VECTOR_DIM                 = ( -47),
	BLIS_UNEXPECTED_DIAG_OFFSET                = ( -48),

	// Stride-specific errors
	BLIS_INVALID_ROW_STRIDE                    = ( -50),
	BLIS_INVALID_COL_STRIDE                    = ( -51),
	BLIS_INVALID_DIM_STRIDE_COMBINATION        = ( -52),

	// Structure-specific errors    
	BLIS_EXPECTED_GENERAL_OBJECT               = ( -60),
	BLIS_EXPECTED_HERMITIAN_OBJECT             = ( -61),
	BLIS_EXPECTED_SYMMETRIC_OBJECT             = ( -62),
	BLIS_EXPECTED_TRIANGULAR_OBJECT            = ( -63),

	// Storage-specific errors    
	BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT        = ( -70),

	// Partitioning-specific errors
	BLIS_INVALID_3x1_SUBPART                   = ( -80),
	BLIS_INVALID_1x3_SUBPART                   = ( -81),
	BLIS_INVALID_3x3_SUBPART                   = ( -82),

	// Control tree-specific errors
	BLIS_UNEXPECTED_NULL_CONTROL_TREE          = ( -90),

	// Packing-specific errors
	BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK  = (-100),

	// Memory allocator errors
	BLIS_EXHAUSTED_STATIC_MEMORY_POOL          = (-110),

	BLIS_ERROR_CODE_MAX                        = (-120)
} err_t;

#endif
