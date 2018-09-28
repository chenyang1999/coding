#include <iostream>
#include <cmath>
#define N 2018
using namespace std;
int a[N][N];
int bx[N];
int by[N];
int main(int argc, char *argv[]) {

	int x,y,n,m,ax,ay,cn=0;
	while (cin>>x>>y) {
        for(int i=0;i<=x+1;i++)
            for(int j=0;j<=y+1;j++)
                a[i][j]=10000;
		cn=0;
		cin>>n;
		for(int i=1;i<=n;i++){
			cin>>ax>>ay;
//			cn++;
			bx[i]=ax;
			by[i]=ay;
		}
		for(int i=1;i<=x;i++)
			for(int j=1;j<=y;j++)
			for(int k=1;k<=n;k++)
			a[i][j]=min(a[i][j],abs(i-bx[k])+abs(j-by[k]));
		int bj=0;
		for(int i=x;i>=1;i--)
			for(int j=y;j>=1;j--)
			if(a[i][j]>=bj){
				bj=a[i][j];
				ax=i;
				ay=j;
			}
	cout<<ax<<" "<<ay<<endl;
	}
	return 0;
	
}