// HashTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string.h>
#include <stdlib.h>
#include "MD5C.C"

//��ʮ�������ַ�����ʽ�������digest�б����MD5ɢ��ֵ
void showhash(unsigned char *digest)
{
     for (int i=0; i<16; i++)
	   printf("%x", digest[i]);
    printf("\n");
}

int main(int argc, char* argv[])
{
	MD5_CTX context;  //�����ı��� 
   	unsigned char digest[16]; //���ڱ������յ�ɢ��ֵ
   

	//*******��������д��Ĵ���*************
	 
   
}

