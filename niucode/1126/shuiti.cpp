#include<iostream>
//链接：https://ac.nowcoder.com/acm/contest/275/H
//来源：牛客网
//
//给定 n，求一对整数 (i,j)，在满足 1 ≤ i ≤ j ≤ n 且  的前提下，要求最大化 i+j 的值
using namespace std;
int main(){
	long long n;
	cin>>n;
	long long a=1;
	if(n==1)cout<<2;else cout<<2*n-1;
	return 0;
}