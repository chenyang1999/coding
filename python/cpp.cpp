#include<iostream>
#include <cmath>
using namespace std;
int main()
{
	int n,m;
	int x,y;
	int a[10100][101];
	int b[10100][101];
	cin>>n>>m>>x>>y;
	for(int i=0;i<n;i++)
		for(int j=0;j<x;j++)
			cin>>a[i][j];
	for(int i=0;i<m;i++)
		for(int j=0;j<x;j++)
			cin>>b[i][j];
	for(int i=0;i<n;i++)
		//for(int j=0;j<m;j++)c[j]=a[i][j];
		for(int j=0;j<m;j++)
			for(int k=0;k<x;k++)
			if(a[i][k]>y-b[j][k])a[i][k]=y-b[j][k];
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<x;j++)cout<<a[i][j]<<" ";
		cout<<endl;
	}
	return 0;
}
