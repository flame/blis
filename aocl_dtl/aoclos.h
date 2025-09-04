/*===================================================================
 * File Name :  aoclos.c
 * 
 * Description : Abstraction for os services used by DTL.
 *
 * Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.
 * 
 *==================================================================*/

#ifndef _AOCL_OS_H_
#define _AOCL_OS_H_

#include "aocltpdef.h"
#include "stdlib.h"

/* The OS Services function declaration */

/* Alias for memory mangement functions. */
#define AOCL_malloc malloc
#define AOCL_free free

AOCL_TID AOCL_gettid(void);
pid_t  AOCL_getpid(void);
uint64 AOCL_getTimestamp(void);

#endif /* _AOCL_OS_H_ */

/* --------------- End of aoclOS.h ----------------- */
