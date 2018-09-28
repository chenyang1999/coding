#include <iostream>
#include <cstdio> 
using namespace std;
int n,bj,d;

int a[10000];
int main ()
{
	while (cin>>n)
	{
		for(int i=0;i<n;i++) cin>>a[i];
		if (n <= 2)cout<<1<<endl;
		else
		{
			for(int i=0;i<n-1;i++)
			{
				d = a[i + 1] - a[i];
				bj = 1;
				for(int j=i+2;j<n;j++) if (a[j] - a[j - 1] != d)bj=0;
				if (bj)
				{
					cout<<i+1<<endl;
					break;
				}
			}
		}
	}
	return 0;
}