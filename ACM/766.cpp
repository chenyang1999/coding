//
//题目描述:
//请寻找并输出1至1000000之间的数m，它满足m、m^2和m^3均为回文数。回文数大家都知道吧，就是各位数字左右对称的整数，例如121、676、123321等。满足上述条件的数如m=11，m^2=121,m^3=1331皆为回文数。
//输入描述:
//没有输入
//输出描述:
//输出1至1000000之间满足要求的全部回文数，每两个数之间用空格隔开，每行输出五个数
#include <iostream>
#include<string>
#include<cstdlib>
#include<cstdio>
#include<string>
using namespace std;
bool pd(long long x){
	long long a[100];
	for(long long i=0;i<100;i++)a[i]=0;
	long long cnt=0,i=0;
	while(x>0){
		a[cnt++]=x%10;
		x=x/10;
	}
	bool bj=true;
	cnt--;
	for(long long i=0;i<cnt;i++)
		if(a[i]!=a[cnt-i])bj=false;
	return bj;
}
int main(int argc, char *argv[]) {
//	打表不解释了;
	long long list[10000],cnt=1;
	for(long long i=1;i<=1000000;i++)
	if(pd(i)&&pd(i*i)&&pd(i*i*i))list[cnt++]=i;
	for(int i=1;i<cnt;i++){
		cout<<list[i]<<" ";
		if(i%5==0)cout<<endl;}
	return 0;
}