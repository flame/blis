### Low Precision POWER10 Kernels

This is a special BLIS Sandbox that allows users to call low precision POWER10 `gemm` kernels. 

#### Introduction

This document describes how the low precision POWER10 `gemm` kernels are implemented. The document will also demonstrate how to call the `gemm` kernels. 

**Important: This sandbox does not have the full functionality of BLIS. This sandbox can only perform single threaded, no transpose, GEMM. At this time, full functioning POWER10 hardware has not be released. Once hardware has been released, the kernels will be further optimized in areas such as prefetching and cache blocksizes.**

#### Implementation

The kernels are implemented in `generic_gemm.c`. They are instantiated with macro templates. The main template is called `GENERIC_GEMM`. This template is used to create the 5-loop `gemm` function.

The API points are created in `gemm_api.c`. In this file, the API points are wrappers for the functions that are created by the templates in `generic_gemm.c`.

#### Kernels

The following low precision datatypes have POWER10 `gemm` kernels: `IEEE float16, bfloat16, int16, int8, int4`. 

#### Low Precision Types

| BLIS type  | BLIS char | Type definition                        | Used to represent...                 |
|:-----------|:----------|:---------------------------------------|:-------------------------------------|
| `float16`    | `h`    | `typedef union { uint16_t v; struct { uint16_t m:10; uint16_t e:5; uint16_t s:1} bits; }` | IEEE half-precision real numbers        |
| `bfloat16`   | `b`    | `typedef union { uint16_t v; struct { uint16_t m:7; uint16_t e:8; uint16_t s:1; } bits; }` | Google's half-precision real numbers    |
| `int16`    | `i16`     | `int16_t`    | 16 bit integers |
| `int8`     | `i8`       | `int8_t`  | 8 bit integers |
| `int4`     | `i4`       | `typedef union{ uint8_t v; struct { uint8_t nib1:4; uint8_t nib2:4; } bits; }` | 4 bit integers |

#### Low Precision API

The API that is used for the low precision POWER10 `gemm` kernels is similar to the existing [BLIS basic typed API](https://github.com/flame/blis/blob/master/docs/BLISTypedAPI.md). The main difference between the two is that in the existing BLIS typed API, there is only one type for the input and output matrices. However in the low precision API, there is a input and output type.

Thus the new `gemm` call looks like the following:

```
void bli_??gemm
     (
       trans_t transa,
       trans_t transb,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       ctype_out*  alpha,
       ctype_in*   a, inc_t rsa, inc_t csa,
       ctype_in*   b, inc_t rsb, inc_t csb,
       ctype_out*  beta,
       ctype_out*  c, inc_t rsc, inc_t csc
     );
```

The first `?` is for the output type. The second `?` is for the input type. 

At this time for IEEE float16 and bfloat16, the only output type is single precision float. For int16, int8, and int4, the only output type is 32 bit int.


#### How To Build The Sandbox

Add the following flags when running the configure script to build BLIS correctly.

`CFLAGS="-fPIC -std=c99 -D_ISOC11_SOURCE -D_POSIX_C_SOURCE=200112L" -s power10`

Ensure that you have GCC 10.2 or greater.



#### References

* [bfloat16 wiki]
* [IEEE float16 wiki]
