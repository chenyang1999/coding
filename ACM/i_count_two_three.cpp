#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace std;
#define ll long long
long long ans[100000];
long long pow (int  x,int  y)
{
	x=x;
	y=y;
	ll ans=1;
	while(y)
	{
		if(y&1)
			ans*=x;
		x*=x;
		y>>=1;
	}
	return ans;
}
int main(){
	int t;
	long long a,b,c,d;
	cin>>t;
	int bj=0;
	for(int i=0; i<30; i++)
		{
			a=pow(2,i);
			for(int j=0; j<30; j++)
			{
				b=pow(3,j);
				for(int k=0; k<30; k++)
				{
					c=pow(5,k);
					for(int m=0; m<30; m++)
					{
						d=pow(7,m);
						if(a*b*c*d<=0||a*b*c*d>1000000000)
							break;
						else
						{
							ans[bj++]=a*b*c*d;
	 
						}
					}
				}
			}
		}
	sort(ans,ans+bj);

//for(int i=0;i<=100;i++)cout<<ans[i]<<endl;
	for(int i=1;i<=t;i++){
		scanf("%ld",&a);
		printf("%ld\n",*lower_bound(ans, ans+bj, a));
	}
	
	return 0;
}