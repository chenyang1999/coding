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

#define TEXT_LEN  16  //明密文长度

//填充随机数变量
void seed_randomStruct (unsigned char *seed, R_RANDOM_STRUCT *randomStruct)
{
    unsigned int bytesNeeded = 256;  //结构体所需种子长度

    R_RandomInit (randomStruct);	
    while (bytesNeeded > 0)
    { 
       R_RandomUpdate (randomStruct, seed, 
                                       strlen((char *)seed));
	   R_GetRandomBytesNeeded (&bytesNeeded, randomStruct);
	}
}

// 以十六进制形式显示output中的内容(参数len表示output的长度)
void shows (unsigned char *output, unsigned int len)
{  
   for (unsigned int i=0; i<len; i++)
	    printf("%x", output[i]);
   printf("\n");
}

int main(int argc, char* argv[])
{ 
	
R_RANDOM_STRUCT   randomStruct; //保存随机数
unsigned char seed[] = "3adqwe1212asd"; // 种子
unsigned char iv[8+1] = "13wedfgr";     // IV
unsigned char input[TEXT_LEN+1] = "12345678abcdefgh"; // 明文

seed_randomStruct (seed, &randomStruct);  // 用种子填充随机数变量

printf ("plaintext: %s\n", input);  // 显示明文

//*****请在下面每一步后面填写你的代码**************

//步骤1：产生随机RSA密钥（包括公钥和私钥）


//步骤2： 数字信封的封装(加密)
  // (1) 产生随机对称会话密钥
  // (2) 用公钥加密该会话密钥
  // (3) 用会话密钥，采用DES-CBC加密明文（初始向量iv定义如上）


//步骤3：显示封装结果
  // 调用函数shows显示密文
  // 调用函数shows显示加密后的对称密钥


//步骤4： 数字信封的解封(解密)
  //(1) 用私钥解密出会话密钥
  //(2) 用会话密钥，采用DES-CBC解密密文（需使用同样的初始向量iv）


//步骤5：以字符串形式显示恢复出的明文


return 0; 
}



