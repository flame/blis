/*===================================================================
 * File Name :  aoclos.c
 * 
 * Description : Abstraction for os services used by DTL.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/

#ifndef _AOCL_OS_H_
#define _AOCL_OS_H_

#include "aocltpdef.h"
#include "malloc.h"

/* The OS Services function declaration */

/* Alias for memory mangement functions. */
#define AOCL_malloc malloc
#define AOCL_free free

uint32 AOCL_gettid(void);
pid_t  AOCL_getpid(void);
uint64 AOCL_getTimestamp(void);

#endif /* _AOCL_OS_H_ */

/* --------------- End of aoclOS.h ----------------- */
