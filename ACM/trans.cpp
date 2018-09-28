#include<iostream>
#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#define ll long long
using namespace std;
ll prime[25] = {97,89,83,79,73,71,67,61,59,53,47,43,41,37,31,29,23,19,17,13,11,7,5,3,2};

ll pow_mod(ll a, ll n, ll mod)
{
	ll ret = 1;
	while (n)
	{
		if (n&1)
			ret = ret * a % mod;
		a = a * a % mod;
		n >>= 1;
	}
	return ret;
}

int isPrime(ll n)
{
	if (n < 2 || (n != 2 && !(n&1)))
		return 0;
	ll s = n - 1;
	while (!(s&1))
		s >>= 1;
	for (int i = 0; i <25 ; ++i) 
	{
		if (n == prime[i])
			return 1;
		ll t = s, m = pow_mod(prime[i], s, n);
		while (t != n-1 && m != 1 && m != n-1)
		{
			m = m * m % n;
			t <<= 1;
		}
		if (m != n-1 && !(t&1))
			return 0;
	}
	return 1;
}
int main(){
    long long n,s,z,m;
	cin>>n;
    s=n;z=0;
    int bj=0;
    if(!isPrime(n))bj=1;
    while (s){
     m=s%10;
        if(m==3||m==4||m==7)bj=1;
        s=s/10;
        if(m==0)z=z*10;
        if(m==1)z=z*10+1;
        if(m==2)z=z*10+2;
        if(m==5)z=z*10+5;
        if(m==6)z=z*10+9;
        if(m==9)z=z*10+6;
    }
    if(!isPrime(z))bj=1;

	if(bj==1)cout<<"no";else cout<<"yes";
    return 0;
}