#include <iostream>
using namespace std;
int main()
{
	int fool;
	int f[100];
	int i;
	int k;
	int sum;
	cout <<"请输入你想去的楼层"<<endl;
	cin >>fool;
	for (i=0;i<fool;i++)
	{
		cin >>f[i];
	}
	for(sum=f[0]*6+5,i=0;i<fool-1;i++)
	{
		if(fool==1)
		{cout <<"一共需要的时间为"<<sum<<endl;
			cout <<f[0]<<"*6+5";
			break;
		}
		if(f[i]>f[i+1])
		{   k=f[i]-f[i+1];
			sum=sum+4*k+5;
			
			
		}
		else
		{
			k=f[i+1]-f[i];
			sum=sum+6*k+5;
			
		}
	}
	cout <<sum<<endl;
	int q;
	cout <<f[0]<<"*6+5"<<'+';
	for(i=0;i<fool-1;i++)
	{
		if(f[i]>f[i+1])
		{   q=f[i]-f[i+1];
			cout <<q<<"*4+5"<<'+';
		}
		else{
			q=f[i+1]-f[i];
			cout <<q<<"*6+5"<<'+';
		}
	}
	return 0;
	
	
	
	
}