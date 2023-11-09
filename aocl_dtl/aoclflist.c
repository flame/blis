/*===================================================================
 * File Name :  aoclflist.c
 *
 * Description : Linked list of open files assocaited with
 *               each thread. This is used to log the data
 *               to correct file as per the current thread id.
 *
 * Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/

#include "blis.h"
#include "aocltpdef.h"
#include "aocldtl.h"
#include "aoclfal.h"
#include "aoclflist.h"
#include "aoclos.h"


/* Disable instrumentation for following function, since they are called from
 * Auto Generated execution trace handlers. */
Bool AOCL_FLIST_IsEmpty(
    AOCL_FLIST_Node *plist) __attribute__((no_instrument_function));

AOCL_FAL_FILE *AOCL_FLIST_GetFile(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid) __attribute__((no_instrument_function));

AOCL_FAL_FILE *AOCL_FLIST_AddFile(
    const int8 *pchFilePrefix,
    AOCL_FLIST_Node **plist,
    AOCL_TID tid) __attribute__((no_instrument_function));

void AOCL_FLIST_CloseFile(
    AOCL_FLIST_Node *plist,
    AOCL_TID tid) __attribute__((no_instrument_function));

void AOCL_FLIST_CloseAll(
    AOCL_FLIST_Node *plist) __attribute__((no_instrument_function));



Bool AOCL_FLIST_IsEmpty(AOCL_FLIST_Node *plist)
{
    return (plist == NULL);

} /* AOCL_FLIST_IsEmpty */

AOCL_FLIST_Node * AOCL_FLIST_GetNode(AOCL_FLIST_Node *plist, AOCL_TID tid)
{
    AOCL_FLIST_Node *temp;

    if (AOCL_FLIST_IsEmpty(plist) == 1)
    {
        return NULL;
    }

    temp = plist;

    /* if list is not empty search for the file handle in all nodes */
    while (temp != NULL)
    {
        if (temp->tid == tid)
        {
            if (temp->fp == NULL)
            {
#ifdef BLIS_ENABLE_PTHREADS
                AOCL_DEBUGPRINT("Could not get saved time stamp for thread = %ld", tid);
#else
                AOCL_DEBUGPRINT("Could not get saved time stamp for thread = %d", tid);
#endif
            }
            return temp;
        }
        temp = temp->pNext;
    }

    return NULL;

} /* AOCL_FLIST_GetNode */

AOCL_FAL_FILE *AOCL_FLIST_GetFile(AOCL_FLIST_Node *plist, AOCL_TID tid)
{
    AOCL_FLIST_Node *temp;

    if (AOCL_FLIST_IsEmpty(plist) == 1)
    {
        return NULL;
    }

    temp = plist;

    /* if list is not empty search for the file handle in all nodes */
    while (temp != NULL)
    {
        if (temp->tid == tid)
        {
            if (temp->fp == NULL)
            {
#ifdef BLIS_ENABLE_PTHREADS
                AOCL_DEBUGPRINT("File associated with this thread id %ld does not exists or closed", tid);
#else
                AOCL_DEBUGPRINT("File associated with this thread id %d does not exists or closed", tid);
#endif
            }
            return temp->fp;
        }
        temp = temp->pNext;
    }

    return NULL;

} /* AOCL_FLIST_GetFile */

AOCL_FAL_FILE *AOCL_FLIST_AddFile(const int8 *pchFilePrefix, AOCL_FLIST_Node **plist, AOCL_TID tid)
{
    AOCL_FLIST_Node *newNode = NULL, *temp = NULL;
    AOCL_FAL_FILE *file = NULL;
    int8 pchFileName[255];

    /* We don't want duplicates so we will check if the file already opened for this thread */
    file = AOCL_FLIST_GetFile(*plist, tid);
    if (file != NULL)
    {
        AOCL_DEBUGPRINT("Open file alread exits for this key.");
        return file;
    }

    /* We don't have exiting file, lets try to open new one */
#ifdef BLIS_ENABLE_PTHREADS
    sprintf(pchFileName, "P%d_T%lu_%s", AOCL_getpid(), tid, pchFilePrefix);
#else
    sprintf(pchFileName, "P%d_T%u_%s", AOCL_getpid(), tid, pchFilePrefix);
#endif
    file = AOCL_FAL_Open(pchFileName, "wb");
    if (file == NULL)
    {
        return NULL;
    }

    /* Now allocate new node as we are sure we will need it */
    newNode = AOCL_malloc(sizeof(AOCL_FLIST_Node));
    if (newNode == NULL)
    {
        AOCL_FAL_Close(file);
        AOCL_DEBUGPRINT("Out of memory while opening new log file");
        return NULL;
    }

    newNode->pNext = NULL;
    newNode->tid = tid;
    newNode->u64SavedTimeStamp = AOCL_getTimestamp();
    newNode->fp = file;

    if (AOCL_FLIST_IsEmpty(*plist) == 1)
    {
        *plist = newNode;
    }
    else
    {
        /* go to the end of the list */
        for (temp = *plist; temp->pNext != NULL; temp = temp->pNext)
            ;

        temp->pNext = newNode;
    }

    return newNode->fp;

} /* AOCL_FLIST_AddFile */

void AOCL_FLIST_CloseFile(AOCL_FLIST_Node *plist, AOCL_TID tid)
{
    AOCL_FAL_FILE *pfile = AOCL_FLIST_GetFile(plist, tid);
    AOCL_FAL_Close(pfile);

    return;

} /* AOCL_FLIST_CloseFile */

void AOCL_FLIST_CloseAll(AOCL_FLIST_Node *plist)
{

    AOCL_FLIST_Node *temp;

    if (AOCL_FLIST_IsEmpty(plist) == 1)
    {
        return;
    }

    temp = plist;

    /* if list is not iterate over all nodes and close the assocaited files*/
    while (temp != NULL)
    {
        AOCL_FAL_Close(temp->fp);
        temp = temp->pNext;
    }

} /* AOCL_FLIST_CloseAll */

/* ------------------- End of aoclflist.c ----------------------- */
