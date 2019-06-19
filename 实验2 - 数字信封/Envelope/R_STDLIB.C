/* R_STDLIB.C - platform-specific C library routines for RSAREF
 */

/* Copyright (C) RSA Laboratories, a division of RSA Data Security,
     Inc., created 1991. All rights reserved.
 */

#include <string.h>
#include "global.h"
#include "rsaref.h"

void R_memset (POINTER output, int value, unsigned int len)
{
  if (len)
    memset (output, value, len);
}

void R_memcpy (POINTER output, POINTER input, unsigned int len)
{
  if (len)
    memcpy (output, input, len);
}

int R_memcmp (POINTER firstBlock, POINTER secondBlock, unsigned int len)
{
  if (len)
    return (memcmp (firstBlock, secondBlock, len));
  else
    return (0);
}
