/*===================================================================
 * File Name :  aoclflist.h
 *
 * Description : Linked list of open files assocaited with
 *               each thread. This is used to log the deta
 *               to correct file as per the current thread id.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 *
 *==================================================================*/

#ifndef _AOCL_FLIST_H_
#define _AOCL_FLIST_H_

#include "aocltpdef.h"
#include "aoclfal.h"

typedef struct AOCL_FLIST_Node_t
{
    AOCL_TID tid;
    AOCL_FAL_FILE *fp;
    uint64 u64SavedTimeStamp;
    struct AOCL_FLIST_Node_t *pNext;
} AOCL_FLIST_Node;

Bool AOCL_FLIST_IsEmpty(
    AOCL_FLIST_Node *plist);

AOCL_FLIST_Node * AOCL_FLIST_GetNode(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid);

AOCL_FAL_FILE *AOCL_FLIST_GetFile(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid);

AOCL_FAL_FILE *AOCL_FLIST_AddFile(
    const int8 *pchFilePrefix,
    AOCL_FLIST_Node **plist,
    AOCL_TID tid);

void AOCL_FLIST_CloseFile(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid);

void AOCL_FLIST_CloseAll(
    AOCL_FLIST_Node *plist);

#endif /* _AOCL_FLIST_H_ */

/* --------------- End of aoclfist.h ----------------- */
