//链接：https://ac.nowcoder.com/acm/problem/17450
//来源：牛客网
//q次询问，每次给一个x，问1到x的因数个数的和。
//输入描述:
//第一行一个正整数q ；
//接下来q行，每行一个正整数 x
#include <iostream>
#include <cmath>
using namespace std;
int main(int argc, char *argv[]) {
	long long n;
	cin>>n;
	while (n--) {
		long long x,ans=0;
		cin>>x;
		long long t=sqrt(x+1);
		for (long long i=1;i<=t;i++) {
			ans+=(x/i);
	
		}
		cout<<ans*2-t*t<<endl;
	}
	return 0;
}