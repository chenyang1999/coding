// Envelope.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <string.h>
#include <stdlib.h>
#include "R_STDLIB.C"
#include "R_RANDOM.C"
#include "NN.C"
#include "RSA.C"
#include "DIGIT.C"
#include "MD5C.C"
#include "PRIME.C"
#include "R_KEYGEN.C"
#include "DESC.C"

#define TEXT_LEN  16  //�����ĳ���

//������������
void seed_randomStruct (unsigned char *seed, R_RANDOM_STRUCT *randomStruct)
{
    unsigned int bytesNeeded = 256;  //�ṹ���������ӳ���

    R_RandomInit (randomStruct);	
    while (bytesNeeded > 0)
    { 
       R_RandomUpdate (randomStruct, seed, 
                                       strlen((char *)seed));
	   R_GetRandomBytesNeeded (&bytesNeeded, randomStruct);
	}
}

// ��ʮ��������ʽ��ʾoutput�е�����(����len��ʾoutput�ĳ���)
void shows (unsigned char *output, unsigned int len)
{  
   for (unsigned int i=0; i<len; i++)
	    printf("%x", output[i]);
   printf("\n");
}

int main(int argc, char* argv[])
{ 
	
R_RANDOM_STRUCT   randomStruct; //���������
unsigned char seed[] = "3adqwe1212asd"; // ����
unsigned char iv[8+1] = "13wedfgr";     // IV
unsigned char input[TEXT_LEN+1] = "12345678abcdefgh"; // ����

seed_randomStruct (seed, &randomStruct);  // ������������������

printf ("plaintext: %s\n", input);  // ��ʾ����

//*****��������ÿһ��������д��Ĵ���**************

//����1���������RSA��Կ��������Կ��˽Կ��


//����2�� �����ŷ�ķ�װ(����)
  // (1) ��������ԳƻỰ��Կ
  // (2) �ù�Կ���ܸûỰ��Կ
  // (3) �ûỰ��Կ������DES-CBC�������ģ���ʼ����iv�������ϣ�


//����3����ʾ��װ���
  // ���ú���shows��ʾ����
  // ���ú���shows��ʾ���ܺ�ĶԳ���Կ


//����4�� �����ŷ�Ľ��(����)
  //(1) ��˽Կ���ܳ��Ự��Կ
  //(2) �ûỰ��Կ������DES-CBC�������ģ���ʹ��ͬ���ĳ�ʼ����iv��


//����5�����ַ�����ʽ��ʾ�ָ���������


return 0; 
}



