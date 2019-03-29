#include<cstdio>
const long long mod=1e9+7;
//快速幂 (a^b)%mod
//如果求逆元，则b = mod-2;
long long pow_quick(long long a,long long b){
	long long r = 1,base = a;
	while(b){
		if(b & 1){
			r *= base;
			r %= mod;
		}
		base *= base;
		base %= mod;
		b >>= 1;
	}
	return r;
}
long long gcd(long long a,long long b)
{
	long long c;
	while(b!=0)
	{
		c=a%b;
		a=b;
		b=c;
	}
	return a;
}
int main()
{
	long long n;
	long long a,b;
	long long A,B,temp;
	scanf("%lld",&n);
	A=B=1;
	while(n--)
	{
		scanf("%lld%lld",&a,&b);
		A*=(b-a);
		B*=b;
		A%=mod;
		B%=mod;
	}
	printf("%lld\n",(((B-A+mod)%mod)*pow_quick(B,mod-2))%mod);
	return 0;
}