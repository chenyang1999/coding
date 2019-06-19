// HashTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string.h>
#include <stdlib.h>
#include "MD5C.C"

//以十六进制字符串形式输出变量digest中保存的MD5散列值
void showhash(unsigned char *digest)
{
     for (int i=0; i<16; i++)
	   printf("%x", digest[i]);
    printf("\n");
}

int main(int argc, char* argv[])
{
	MD5_CTX context;  //上下文变量 
   	unsigned char digest[16]; //用于保存最终的散列值
   

	//*******在下面填写你的代码*************
	 
   
}

