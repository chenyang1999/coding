/* MD2.H - header file for MD2C.C
 */

/* Copyright (C) 1990-2, RSA Data Security, Inc. Created 1990. All
   rights reserved.

   License to copy and use this software is granted for
   non-commercial Internet Privacy-Enhanced Mail provided that it is
   identified as the "RSA Data Security, Inc. MD2 Message Digest
   Algorithm" in all material mentioning or referencing this software
   or this function.

   RSA Data Security, Inc. makes no representations concerning either
   the merchantability of this software or the suitability of this
   software for any particular purpose. It is provided "as is"
   without express or implied warranty of any kind.

   These notices must be retained in any copies of any part of this
   documentation and/or software.  
 */

#ifndef _MD2_H_
#define _MD2_H_ 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned char state[16];                                 /* state */
  unsigned char checksum[16];                           /* checksum */
  unsigned int count;                 /* number of bytes, modulo 16 */
  unsigned char buffer[16];                         /* input buffer */
} MD2_CTX;

void MD2Init PROTO_LIST ((MD2_CTX *));
void MD2Update PROTO_LIST
  ((MD2_CTX *, unsigned char *, unsigned int));
void MD2Final PROTO_LIST ((unsigned char [16], MD2_CTX *));

#ifdef __cplusplus
}
#endif

#endif
